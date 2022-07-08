import copy
import warnings

import torch
import numpy as np
from torch.utils.data import Dataset

import pddlgym
from pddlgym.structs import Predicate

from .planning import PlanningFailure, PlanningTimeout


class GraphDictDataset(Dataset):
    def __init__(self, graph_dicts_input, graph_dicts_target):
        self.graph_dicts_input = graph_dicts_input
        self.graph_dicts_target = graph_dicts_target

    def __len__(self):
        return len(self.graph_dicts_input)

    def __getitem__(self, idx):
        sample = {
            "graph_input": self.graph_dicts_input[idx],
            "graph_target": self.graph_dicts_target[idx],
        }
        return sample


def collect_training_data(
    train_env_name, train_planner, num_train_problems, timeout=60
):
    inputs = []
    labels = []
    env = pddlgym.make(f"PDDLEnv{train_env_name}-v0")
    assert env.operators_as_actions
    print(f"{len(env.problems)} problems found!")
    # for idx in range(len(env.problems)):
    for idx in range(min(num_train_problems, len(env.problems))):
        print(f"Collecting training data for problem {idx}")
        env.fix_problem_index(idx)
        state, _ = env.reset()
        try:
            train_planner.reset_statistics()
            plan = train_planner(env.domain, state, timeout=timeout)
        except (PlanningTimeout, PlanningFailure):
            warnings.warn(
                f"Planning failed. Skipping problem {idx}, env.problems[idx].fname"
            )
            continue
        inputs.append(state)
        objects_in_plan = {o for act in plan for o in act.variables}
        labels.append(objects_in_plan)
    return inputs, labels


def wrap_goal_literal(x):
    """Convert a state to a required input representation. """
    if isinstance(x, Predicate):
        return Predicate(
            "WANT" + x.name,
            x.arity,
            var_types=x.var_types,
            is_negative=x.is_negative,
            is_anti=x.is_anti,
        )
    new_predicate = wrap_goal_literal(x.predicate)
    return new_predicate(*x.variables)


def reverse_binary_literal(x):
    if isinstance(x, Predicate):
        assert x.arity == 2
        return Predicate(
            "REV" + x.name,
            x.arity,
            var_types=x.var_types,
            is_negative=x.is_negative,
            is_anti=x.is_anti,
        )
    new_predicate = reverse_binary_literal(x.predicate)
    variables = [v for v in x.variables]
    assert len(variables) == 2
    return new_predicate(*variables[::-1])


def state_to_graph(state, graph_metadata):
    """Create a graph from a pddlgym.State object. """
    all_objects = sorted(state.objects)
    node_to_objects = dict(enumerate(all_objects))
    objects_to_node = {v: k for k, v in node_to_objects.items()}
    num_objects = len(all_objects)

    G = wrap_goal_literal
    R = reverse_binary_literal

    graph_input = {}

    # Nodes: one per object
    graph_input["n_node"] = np.array(num_objects)
    input_node_features = np.zeros((num_objects, graph_metadata["num_node_features"]))

    # Add features for types
    for obj_idx, obj in enumerate(all_objects):
        type_idx = graph_metadata["node_feature_to_idx"][obj.var_type]
        input_node_features[obj_idx, type_idx] = 1

    # Add features for unary state literals
    for lit in state.literals:
        if lit.predicate.arity != 1:
            continue
        lit_idx = graph_metadata["node_feature_to_idx"][lit.predicate]
        assert len(lit.variables) == 1
        obj_idx = objects_to_node[lit.variables[0]]
        input_node_features[obj_idx, lit_idx] = 1

    # Add features for unary goal literals
    for lit in state.goal.literals:
        if lit.predicate.arity != 1:
            continue
        lit_idx = graph_metadata["node_feature_to_idx"][G(lit.predicate)]
        assert len(lit.variables) == 1
        obj_idx = objects_to_node[lit.variables[0]]
        input_node_features[obj_idx, lit_idx] = 1

    graph_input["nodes"] = input_node_features

    # Edges
    all_edge_features = np.zeros(
        (num_objects, num_objects, graph_metadata["num_edge_features"])
    )

    # Add edge features for binary state literals
    for bin_lit in state.literals:
        if bin_lit.predicate.arity != 2:
            continue
        for lit in [bin_lit, R(bin_lit)]:
            pred_idx = graph_metadata["edge_feature_to_idx"][lit.predicate]
            assert len(lit.variables) == 2
            obj0_idx = objects_to_node[lit.variables[0]]
            obj1_idx = objects_to_node[lit.variables[1]]
            all_edge_features[obj0_idx, obj1_idx, pred_idx] = 1

    # Add edge features for binary goal literals
    for bin_lit in state.goal.literals:
        if bin_lit.predicate.arity != 2:
            continue
        for lit in [G(bin_lit), G(R(bin_lit))]:
            pred_idx = graph_metadata["edge_feature_to_idx"][lit.predicate]
            assert len(lit.variables) == 2
            obj0_idx = objects_to_node[lit.variables[0]]
            obj1_idx = objects_to_node[lit.variables[1]]
            all_edge_features[obj0_idx, obj1_idx, pred_idx] = 1

    # Organize into expected representation
    adjmat = np.any(all_edge_features, axis=2)
    receivers, senders, edges = [], [], []
    for sender, receiver in np.argwhere(adjmat):
        edge = all_edge_features[sender, receiver]
        senders.append(sender)
        receivers.append(receiver)
        edges.append(edge)

    num_edges = len(edges)
    edges = np.reshape(edges, [num_edges, graph_metadata["num_edge_features"]])
    receivers = np.reshape(receivers, [num_edges]).astype(np.int64)
    senders = np.reshape(senders, [num_edges]).astype(np.int64)
    num_edges = np.reshape(num_edges, [1]).astype(np.int64)

    graph_input["receivers"] = receivers
    graph_input["senders"] = senders
    graph_input["n_edge"] = num_edges
    graph_input["edges"] = edges

    # Globals
    graph_input["globals"] = None

    return graph_input, node_to_objects


def create_graph_dataset(training_data):

    _unary_types = set()
    _unary_predicates, _binary_predicates = set(), set()
    _node_feature_to_idx, _edge_feature_to_idx = {}, {}

    inputs, labels = training_data

    for state in inputs:
        types = {o.var_type for o in state.objects}
        _unary_types.update(types)
        for lit in set(state.literals) | set(state.goal.literals):
            arity = lit.predicate.arity
            assert arity == len(lit.variables)
            assert arity <= 2, "Arity >2 predicates not yet supported"
            if arity == 0:
                continue
            elif arity == 1:
                _unary_predicates.add(lit.predicate)
            elif arity == 2:
                _binary_predicates.add(lit.predicate)
    _unary_types = sorted(_unary_types)
    _unary_predicates = sorted(_unary_predicates)
    _binary_predicates = sorted(_binary_predicates)

    G = wrap_goal_literal
    R = reverse_binary_literal

    # Initialize node features
    idx = 0
    for unary_type in _unary_types:
        _node_feature_to_idx[unary_type] = idx
        idx += 1
    for unary_predicate in _unary_predicates:
        _node_feature_to_idx[unary_predicate] = idx
        idx += 1
    for unary_predicate in _unary_predicates:
        _node_feature_to_idx[G(unary_predicate)] = idx
        idx += 1

    # Initialize edge features
    idx = 0
    for binary_predicate in _binary_predicates:
        _edge_feature_to_idx[binary_predicate] = idx
        idx += 1
    for binary_predicate in _binary_predicates:
        _edge_feature_to_idx[R(binary_predicate)] = idx
        idx += 1
    for binary_predicate in _binary_predicates:
        _edge_feature_to_idx[G(binary_predicate)] = idx
        idx += 1
    for binary_predicate in _binary_predicates:
        _edge_feature_to_idx[G(R(binary_predicate))] = idx
        idx += 1

    _num_node_features = len(_node_feature_to_idx)
    nnf = _num_node_features
    assert max(_node_feature_to_idx.values()) == nnf - 1
    _num_edge_features = len(_edge_feature_to_idx)
    nef = _num_edge_features
    assert max(_edge_feature_to_idx.values()) == nef - 1

    # Process data
    num_training_examples = len(inputs)
    graphs_input, graphs_target = [], []

    graph_metadata = {
        "num_node_features": _num_node_features,
        "num_edge_features": _num_edge_features,
        "node_feature_to_idx": _node_feature_to_idx,
        "edge_feature_to_idx": _edge_feature_to_idx,
        "unary_types": _unary_types,
        "unary_predicates": _unary_predicates,
        "binary_predicates": _binary_predicates,
    }

    for i in range(num_training_examples):
        state = inputs[i]
        target_object_set = labels[i]
        graph_input, node_to_objects = state_to_graph(state, graph_metadata)
        graph_target = {
            "n_node": graph_input["n_node"],
            "n_edge": graph_input["n_edge"],
            "edges": graph_input["edges"],
            "senders": graph_input["senders"],
            "receivers": graph_input["receivers"],
            "globals": graph_input["globals"],
        }

        # Target nodes
        objects_to_node = {v: k for k, v in node_to_objects.items()}
        object_mask = np.zeros((len(node_to_objects), 1), dtype=np.int64)
        for o in target_object_set:
            obj_idx = objects_to_node[o]
            object_mask[obj_idx] = 1
        graph_target["nodes"] = object_mask

        graphs_input.append(graph_input)
        graphs_target.append(graph_target)

    return graphs_input, graphs_target, graph_metadata


def state_to_graph_hierarchical(state, graph_metadata, target_object_set=None):
    """Create a graph from a pddlgym.State object. """
    all_objects = sorted(state.objects)
    node_to_objects = dict(enumerate(all_objects))
    objects_to_node = {v: k for k, v in node_to_objects.items()}
    num_objects = len(all_objects)

    G = wrap_goal_literal
    R = reverse_binary_literal

    graph_input = {}

    # Nodes: one per object
    graph_input["n_node"] = np.array(num_objects)
    input_node_features = np.zeros((num_objects, graph_metadata["num_node_features"]))

    # Add features for types
    for obj_idx, obj in enumerate(all_objects):
        type_idx = graph_metadata["node_feature_to_idx"][obj.var_type]
        input_node_features[obj_idx, type_idx] = 1

    # Add features for unary state literals
    for lit in state.literals:
        if lit.predicate.arity != 1:
            continue
        lit_idx = graph_metadata["node_feature_to_idx"][lit.predicate]
        assert len(lit.variables) == 1
        obj_idx = objects_to_node[lit.variables[0]]
        input_node_features[obj_idx, lit_idx] = 1

    # Add features for unary goal literals
    for lit in state.goal.literals:
        if lit.predicate.arity != 1:
            continue
        lit_idx = graph_metadata["node_feature_to_idx"][G(lit.predicate)]
        assert len(lit.variables) == 1
        obj_idx = objects_to_node[lit.variables[0]]
        input_node_features[obj_idx, lit_idx] = 1

    graph_input["nodes"] = input_node_features

    # Edges
    all_edge_features = np.zeros(
        (num_objects, num_objects, graph_metadata["num_edge_features"])
    )

    # Add edge features for binary state literals
    for bin_lit in state.literals:
        if bin_lit.predicate.arity != 2:
            continue
        for lit in [bin_lit, R(bin_lit)]:
            pred_idx = graph_metadata["edge_feature_to_idx"][lit.predicate]
            assert len(lit.variables) == 2
            obj0_idx = objects_to_node[lit.variables[0]]
            obj1_idx = objects_to_node[lit.variables[1]]
            all_edge_features[obj0_idx, obj1_idx, pred_idx] = 1

    # Add edge features for binary goal literals
    for bin_lit in state.goal.literals:
        if bin_lit.predicate.arity != 2:
            continue
        for lit in [G(bin_lit), G(R(bin_lit))]:
            pred_idx = graph_metadata["edge_feature_to_idx"][lit.predicate]
            assert len(lit.variables) == 2
            obj0_idx = objects_to_node[lit.variables[0]]
            obj1_idx = objects_to_node[lit.variables[1]]
            all_edge_features[obj0_idx, obj1_idx, pred_idx] = 1

    # Organize into expected representation
    adjmat = np.any(all_edge_features, axis=2)
    receivers, senders, edges = [], [], []
    for sender, receiver in np.argwhere(adjmat):
        edge = all_edge_features[sender, receiver]
        senders.append(sender)
        receivers.append(receiver)
        edges.append(edge)

    num_edges = len(edges)
    edges = np.reshape(edges, [num_edges, graph_metadata["num_edge_features"]])
    receivers = np.reshape(receivers, [num_edges]).astype(np.int64)
    senders = np.reshape(senders, [num_edges]).astype(np.int64)
    num_edges = np.reshape(num_edges, [1]).astype(np.int64)

    graph_input["receivers"] = receivers
    graph_input["senders"] = senders
    graph_input["n_edge"] = num_edges
    graph_input["edges"] = edges

    # Globals
    graph_input["globals"] = None

    """
    Build room graph
    """
    num_room_nodes = 0  # Number of nodes in the room graph
    num_room_edges = 0  # Number of edges in the room graph
    room_inds_global = (
        []
    )  # Indices of rooms within the full object set (i.e., global object ids of room nodes)
    room_edges_global = []  # Edges in the room graph (global)
    room_edges_local = []  # Edges in the room graph (local)
    room_subnodes_global = (
        {}
    )  # Dict of indices of places and objects contained within each room (global wrt the entire object set)
    room_subnodes_local = None  # Dict of indices of places and objects contained within each room (local to the room graph)
    parent_object_dict = (
        {}
    )  # Dict to store the parent object of each object (i.e., dict[hash_of_object] = hash_of_parent_object)
    room_place_dict = (
        {}
    )  # Dict containing the "place" (i.e., door) associated with each room
    place_location_dict = (
        {}
    )  # Dict mapping each location to its entry point (i.e., virtual door-equivalent for each place)
    location_place_dict = (
        {}
    )  # Inverse dict for the place_location_dict. Maps a location to its place.
    object_location_dict = (
        {}
    )  # Dict mapping each object (receptacle / item / agent) to its location
    object_place_dict = {}  # Dict mapping each object to its place
    location_to_placelocation_dict = (
        {}
    )  # Dict mapping each location to its "placelocation" location (i.e., the location that allows accessing all co-located objcts)

    # Determine num rooms
    for obj_idx, obj in enumerate(all_objects):
        type_idx = graph_metadata["node_feature_to_idx"][obj.var_type]
        # print(obj.var_type)
        if obj.var_type == "room":
            num_room_nodes += 1
            room_inds_global.append(obj_idx)
    # Map room node indices from original object set to reduced room-specific set
    # i.e., if originally rooms are 121, 137, 149, etc., relabel them to 0, 1, 2, etc.
    room_node_global_to_local = {k: v for v, k in enumerate(room_inds_global)}
    # Initialize dictionary of subnodes of each room
    room_subnodes_global = {k: [] for k in room_inds_global}

    # Determine room connectivities
    for bin_lit in state.literals:
        if bin_lit.predicate.arity != 2:
            continue
        for lit in [bin_lit, R(bin_lit)]:
            pred_idx = graph_metadata["edge_feature_to_idx"][lit.predicate]
            assert len(lit.variables) == 2
            obj0_idx = objects_to_node[lit.variables[0]]
            obj1_idx = objects_to_node[lit.variables[1]]
            if lit.predicate == "roomsconnected":
                num_room_edges += 1
                room_edges_global.append((obj0_idx, obj1_idx))
                room_edges_local.append(
                    (
                        room_node_global_to_local[obj0_idx],
                        room_node_global_to_local[obj1_idx],
                    )
                )

    # Determine the places and locations present in each room and store them
    # This is useful for subsequent pruning operations (i.e., prune out all places and locations in unimportant rooms)
    # First, check "placeinroom" predicates (i.e., assign all places to rooms)
    for bin_lit in state.literals:
        if bin_lit.predicate.arity != 2:
            continue
        pred_idx = graph_metadata["edge_feature_to_idx"][bin_lit.predicate]
        if bin_lit.predicate in ["placeinroom", "roomplace"]:
            # print(bin_lit.predicate)
            # First object in predicate denotes "place" (for "placeinroom")
            obj0_idx = objects_to_node[bin_lit.variables[0]]
            # Second object in predicate denotes "room" (for "placeinroom")
            obj1_idx = objects_to_node[bin_lit.variables[1]]
            # Add the place to the room
            room_subnodes_global[obj1_idx].append(obj0_idx)
            # Add the room as the parent of the place
            parent_object_dict[bin_lit.variables[0]] = copy.deepcopy(
                bin_lit.variables[1]
            )
            # If this is a "roomplace", add it to the room place dict
            if bin_lit.predicate in ["roomplace"]:
                # There's only one place per room with roomplace set to True
                assert bin_lit.variables[1] not in room_place_dict.keys()
                room_place_dict[bin_lit.variables[1]] = copy.deepcopy(
                    bin_lit.variables[0]
                )

    # Second, check "locationinplace" predicates (i.e., assign all locations to rooms)
    for bin_lit in state.literals:
        if bin_lit.predicate.arity != 2:
            continue
        pred_idx = graph_metadata["edge_feature_to_idx"][bin_lit.predicate]
        if bin_lit.predicate in ["locationinplace", "placelocation"]:
            # print(bin_lit.predicate)
            # First object in predicate denotes "location" (for "locationinplace")
            obj0_idx = objects_to_node[bin_lit.variables[0]]
            # Second object in predicate denotes "place" (for "locationinplace")
            obj1_idx = objects_to_node[bin_lit.variables[1]]
            # Parent of this location is set to this place
            parent_object_dict[bin_lit.variables[0]] = copy.deepcopy(
                bin_lit.variables[1]
            )
            # Add the location to the room that contains the place (room --contains-> place --contains-> location)
            for k in room_subnodes_global:
                if obj1_idx in room_subnodes_global[k]:
                    # assert obj0_idx not in room_subnodes_global[k]
                    room_subnodes_global[k].append(obj0_idx)
                    break
            # If this is a "placelocation", add it to the placelocation dict
            if bin_lit.predicate in ["placelocation"]:
                # There's only one location per place with placelocation set to True
                assert bin_lit.variables[1] not in place_location_dict.keys()
                place_location_dict[bin_lit.variables[1]] = copy.deepcopy(
                    bin_lit.variables[0]
                )
            # Also store the mapping from this location to the place it is in
            # Aside: This assertion may not hold for all (since there may be the same locations with
            # both "locationinplace" and "placelocation" predicates)
            # assert bin_lit.variables[0] not in location_place_dict.keys()
            location_place_dict[bin_lit.variables[0]] = copy.deepcopy(
                bin_lit.variables[1]
            )
    # # Flip the place->location dict to form a location->place dict
    # location_to_place_dict = {v: k for k, v in place_location_dict.items()}

    # Next, check "receptacleatlocation" or "itematlocation" predicates (i.e., assign all receptacles and items to rooms)
    for bin_lit in state.literals:
        if bin_lit.predicate.arity != 2:
            continue
        pred_idx = graph_metadata["edge_feature_to_idx"][bin_lit.predicate]
        if bin_lit.predicate in ["receptacleatlocation", "itematlocation"]:
            # First object in predicate denotes "receptacle" or "item" (for "receptacleatlocation" or "itematlocation")
            obj0_idx = objects_to_node[bin_lit.variables[0]]
            # Second object in predicate denotes "location" (for "receptacleatlocation" and "itematlocation")
            obj1_idx = objects_to_node[bin_lit.variables[1]]
            # Parent of this receptacle / item is set to this location
            parent_object_dict[bin_lit.variables[0]] = bin_lit.variables[1]
            # Add the item or receptacle to the room that contains the location
            for k in room_subnodes_global:
                if obj1_idx in room_subnodes_global[k]:
                    # print(f"Found: {bin_lit.variables[0].var_type}, {bin_lit.variables[1].var_type}")
                    room_subnodes_global[k].append(obj0_idx)
                    # Map the current object to its location
                    assert bin_lit.variables[0] not in place_location_dict.keys()
                    object_location_dict[bin_lit.variables[0]] = copy.deepcopy(
                        bin_lit.variables[1]
                    )
                    # Map each object to its 'place'
                    object_place_dict[bin_lit.variables[0]] = copy.deepcopy(
                        location_place_dict[bin_lit.variables[1]]
                    )
                    # break

    # Next, add the agent to the room it is currently in
    for bin_lit in state.literals:
        if bin_lit.predicate.arity != 2:
            continue
        pred_idx = graph_metadata["edge_feature_to_idx"][bin_lit.predicate]
        if bin_lit.predicate in ["inroom"]:
            # print(bin_lit.predicate)
            # First object in predicate denotes "agent"
            obj0_idx = objects_to_node[bin_lit.variables[0]]
            # Second object in predicate denotes "room" that the "agent" is currently in
            obj1_idx = objects_to_node[bin_lit.variables[1]]
            # Add agent to room
            room_subnodes_global[obj1_idx].append(obj0_idx)
    # Next, add the agent to the location it is currently in
    for bin_lit in state.literals:
        if bin_lit.predicate.arity != 2:
            continue
        pred_idx = graph_metadata["edge_feature_to_idx"][bin_lit.predicate]
        if bin_lit.predicate in ["atlocation"]:
            # variables[0] -> agent; variables[1] -> location
            parent_object_dict[bin_lit.variables[0]] = copy.deepcopy(
                bin_lit.variables[1]
            )
            # Map the agent object to its corresponding location object
            object_location_dict[bin_lit.variables[0]] = copy.deepcopy(
                bin_lit.variables[1]
            )
            # Map the agent object to its corresponding place object
            object_place_dict[bin_lit.variables[0]] = copy.deepcopy(
                location_place_dict[bin_lit.variables[1]]
            )

    # Build the location to placelocation dict
    for l, p in location_place_dict.items():
        location_to_placelocation_dict[l] = copy.deepcopy(place_location_dict[p])

    for bin_lit in state.literals:
        if bin_lit.predicate.arity != 2:
            continue
        pred_idx = graph_metadata["edge_feature_to_idx"][bin_lit.predicate]
        if bin_lit.predicate in ["inreceptacle"]:
            # inreceptacle ?i - item ?r - receptacle
            item_idx = objects_to_node[bin_lit.variables[0]]
            item_obj = bin_lit.variables[0]
            receptacle_idx = objects_to_node[bin_lit.variables[1]]
            receptacle_obj = bin_lit.variables[1]
            # Parent of this item is set to the receptacle
            parent_object_dict[item_obj] = copy.deepcopy(receptacle_obj)

    # # Test case (assert that each object has been added to at least one room)
    # # This won't necessarily be true for all domains (e.g., in the bagslots domains,
    # # we don't need the bagslots to be connected to anything else)
    # for obj_idx, obj in enumerate(all_objects):
    #     type_idx = graph_metadata["node_feature_to_idx"][obj.var_type]
    #     # print(obj.var_type)
    #     found = False
    #     for k in room_subnodes_global:
    #         if type_idx in room_subnodes_global[k]:
    #             found = True
    #     assert found, f"Object idx {type_idx} not found in any room"

    room_graph = {}
    room_graph["n_node"] = num_room_nodes
    room_graph["n_edge"] = num_room_edges
    room_graph["senders"] = np.array([u for (u, v) in room_edges_local])
    room_graph["receivers"] = np.array([v for (u, v) in room_edges_local])
    efeats = []
    room_graph["edge_feat_inds_global"] = []
    for curedge in room_edges_global:
        # # curedge = room_edges_global[i]
        # Look up room_edges_global; iterate over each edge. E.g., first edge may be (121, 122)
        # Find all outgoing vertices from 121 (i.e., all senders with idx 121) and all incoming
        # vertices into 122 (i.e., all receivers with idx 122). Intersecting them will give one
        # unique edge going from 121 to 122. If there's multiple, pick the first of them
        _edgeid = (
            np.intersect1d(
                np.where(graph_input["senders"] == curedge[0]),
                np.where(graph_input["receivers"] == curedge[1]),
            )
        )[0]
        # Look up the edge id of this edge in the corresponding graph
        efeats.append(graph_input["edges"][_edgeid])
        room_graph["edge_feat_inds_global"].append(_edgeid)
        # print(graph_input["senders"][_edgeid] == curedge[0], graph_input["receivers"][_edgeid] == curedge[1])

    # Cast efeats to np array and store in room_graph
    room_graph["edges"] = np.array(efeats)

    # Populate node features for room graph
    nfeats = []
    for node_idx in room_inds_global:
        # Pool features from room's subnodes to build node feature vector for room
        subnfeats = []
        subfeats_from_edges_of_subnodes = []
        room_has_receptacle = False
        room_has_goal_predicate = False
        room_has_agent = False
        for subnode_idx in room_subnodes_global[node_idx]:

            # Determine whether the room has any receptacles
            subnodefeat = graph_input["nodes"][subnode_idx]
            room_has_receptacle = (subnodefeat[4] == 1) or room_has_receptacle
            room_has_agent = (subnodefeat[0] == 1) or room_has_agent
            # room_has_goal_predicate = (subnodefeat[8] == 1) or (subnodefeat[9] == 1) or room_has_goal_predicate

            # print(subnode_idx, np.where(graph_input["senders"] == subnode_idx))
            sender_indices = np.where(graph_input["senders"] == subnode_idx)
            for sender_idx in sender_indices[0]:
                # subfeats_from_edges_of_subnodes.append(graph_input["edges"][sender_idx])
                subedgefeat = graph_input["edges"][sender_idx]
                # Indices 22, 23, 24, 25, 26 correspond to goal binary literals (i.e., G())
                # for the relations atlocation, inplace, inreceptacle, inroom, itematlocation
                # Indices 33, 34, 35, 36, 37 correspond to the goal reversed binary literals
                # (i.e., G(R())) for the same relations.
                room_has_goal_predicate = (
                    room_has_goal_predicate
                    or (subedgefeat[[22, 23, 24, 25, 26]].sum() >= 1)
                    or (subedgefeat[[33, 34, 35, 36, 37]].sum() >= 1)
                )
            receiver_indices = np.where(graph_input["receivers"] == subnode_idx)
            for receiver_idx in receiver_indices[0]:
                # subfeats_from_edges_of_subnodes.append(graph_input["edges"][receiver_idx])
                subedgefeat = graph_input["edges"][receiver_idx]
                # # print(graph_input["edges"][receiver_idx].shape)

            subnodefeat_extended = np.zeros((subnodefeat.shape[0] + 3))
            subnodefeat_extended[:-3] = subnodefeat.copy()
            if room_has_receptacle:
                subnodefeat_extended[-3] = 1.0
            if room_has_agent:
                subnodefeat_extended[-2] = 1.0
            if room_has_goal_predicate:
                subnodefeat_extended[-1] = 1.0
            # extended_subnodefeat = np.zeros_like()

            subnfeats.append(subnodefeat_extended)
            # subnfeats.append(subnodefeat)

        subfeats_from_nodes_of_subnodes = np.stack(subnfeats).max(0)
        # subfeats_from_edges_of_subnodes = np.stack(subfeats_from_edges_of_subnodes).max(0)
        nfeats.append(subfeats_from_nodes_of_subnodes)
        # nfeats.append(np.concatenate((subfeats_from_nodes_of_subnodes, subfeats_from_edges_of_subnodes)))
        # nfeats.append(graph_input["nodes"][node_idx])  # Naive node feature (doesn't work -- identical feat for all rooms)
    # Drop the first 5 dims (cause they denote object types, and in the room graph, all nodes are of type "room")
    nfeats = np.array(nfeats)[:, 5:]
    room_graph["nodes"] = nfeats

    # Populate other metadata for room graph
    room_graph["room_inds_global"] = room_inds_global
    room_graph["room_edges_global"] = room_edges_global
    room_graph["room_edges_local"] = room_edges_local
    room_graph["room_subnodes_global"] = room_subnodes_global
    room_graph["parent_object_dict"] = parent_object_dict
    room_graph["room_place_dict"] = room_place_dict
    room_graph["place_location_dict"] = place_location_dict
    room_graph["location_place_dict"] = location_place_dict
    room_graph["location_to_placelocation_dict"] = location_to_placelocation_dict
    room_graph["object_location_dict"] = object_location_dict
    room_graph["object_place_dict"] = object_place_dict

    graph_input["room_graph"] = room_graph

    """
    Generate labels
    """

    # Labels for overall scene graph (i.e., entire object set)
    if target_object_set is not None:
        objects_to_node = {v: k for k, v in node_to_objects.items()}
        object_mask = np.zeros((len(node_to_objects), 1), dtype=np.int64)
        for o in target_object_set:
            obj_idx = objects_to_node[o]
            object_mask[obj_idx] = 1

        # Labels for room graph
        room_mask = np.zeros((room_graph["n_node"], 1), dtype=np.int64)
        for o in target_object_set:
            obj_idx = objects_to_node[o]
            if obj_idx in room_inds_global:
                room_idx_local = room_inds_global.index(obj_idx)
                room_mask[room_idx_local] = 1
                # print(obj_idx, room_inds_global, room_inds_global.index(obj_idx))

        # Copy the input graph into a label graph
        graph_label = copy.deepcopy(graph_input)
        graph_label["nodes"] = object_mask
        graph_label["room_graph"]["nodes"] = room_mask
    else:
        graph_label = {}
        graph_label["room_graph"] = {}

    # Set global attributes to none in both the input and label room graphs
    graph_input["room_graph"]["globals"] = None
    graph_label["room_graph"]["globals"] = None

    return graph_input, graph_label, node_to_objects


def create_graph_dataset_hierarchical(training_data):

    _unary_types = set()
    _unary_predicates, _binary_predicates = set(), set()
    _node_feature_to_idx, _edge_feature_to_idx = {}, {}

    inputs, labels = training_data

    for state in inputs:
        types = {o.var_type for o in state.objects}
        _unary_types.update(types)
        for lit in set(state.literals) | set(state.goal.literals):
            arity = lit.predicate.arity
            assert arity == len(lit.variables)
            assert arity <= 2, "Arity >2 predicates not yet supported"
            if arity == 0:
                continue
            elif arity == 1:
                _unary_predicates.add(lit.predicate)
            elif arity == 2:
                _binary_predicates.add(lit.predicate)
    _unary_types = sorted(_unary_types)
    _unary_predicates = sorted(_unary_predicates)
    _binary_predicates = sorted(_binary_predicates)

    G = wrap_goal_literal
    R = reverse_binary_literal

    # Initialize node features
    idx = 0
    for unary_type in _unary_types:
        # print(unary_type, idx)
        _node_feature_to_idx[unary_type] = idx
        idx += 1
    for unary_predicate in _unary_predicates:
        # print(unary_predicate, idx)
        _node_feature_to_idx[unary_predicate] = idx
        idx += 1
    for unary_predicate in _unary_predicates:
        # print(G(unary_predicate), idx)
        _node_feature_to_idx[G(unary_predicate)] = idx
        idx += 1

    # Initialize edge features
    idx = 0
    for binary_predicate in _binary_predicates:
        # print(binary_predicate, idx)
        _edge_feature_to_idx[binary_predicate] = idx
        idx += 1
    for binary_predicate in _binary_predicates:
        # print(R(binary_predicate), idx)
        _edge_feature_to_idx[R(binary_predicate)] = idx
        idx += 1
    for binary_predicate in _binary_predicates:
        # print(G(binary_predicate), idx)
        _edge_feature_to_idx[G(binary_predicate)] = idx
        idx += 1
    for binary_predicate in _binary_predicates:
        # print(G(R(binary_predicate)), idx)
        _edge_feature_to_idx[G(R(binary_predicate))] = idx
        idx += 1

    _num_node_features = len(_node_feature_to_idx)
    nnf = _num_node_features
    assert max(_node_feature_to_idx.values()) == nnf - 1
    _num_edge_features = len(_edge_feature_to_idx)
    nef = _num_edge_features
    assert max(_edge_feature_to_idx.values()) == nef - 1

    # Process data
    num_training_examples = len(inputs)
    graphs_input, graphs_target = [], []

    graph_metadata = {
        "num_node_features": _num_node_features,
        "num_edge_features": _num_edge_features,
        "node_feature_to_idx": _node_feature_to_idx,
        "edge_feature_to_idx": _edge_feature_to_idx,
        "unary_types": _unary_types,
        "unary_predicates": _unary_predicates,
        "binary_predicates": _binary_predicates,
    }

    for i in range(num_training_examples):
        state = inputs[i]
        target_object_set = labels[i]
        graph_input, graph_label, _ = state_to_graph_hierarchical(
            state, graph_metadata, target_object_set
        )

        for k in ["edges", "senders", "receivers", "nodes"]:
            graph_input[k] = torch.from_numpy(graph_input[k])
            graph_input["room_graph"][k] = torch.from_numpy(
                graph_input["room_graph"][k]
            )
            graph_label[k] = torch.from_numpy(graph_label[k])
            graph_label["room_graph"][k] = torch.from_numpy(
                graph_label["room_graph"][k]
            )
            if k in ["nodes", "edges"]:
                graph_input[k] = graph_input[k].float()
                graph_input["room_graph"][k] = graph_input["room_graph"][k].float()
                graph_label[k] = graph_label[k].float()
                graph_label["room_graph"][k] = graph_label["room_graph"][k].float()

        graphs_input.append(graph_input)
        graphs_target.append(graph_label)

    return graphs_input, graphs_target, graph_metadata
