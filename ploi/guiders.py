from .datautils import state_to_graph, state_to_graph_hierarchical
from .guidance import BaseSearchGuidance
from .traineval import (
    predict_graph_with_graphnetwork,
    predict_graph_with_graphnetwork_hierarchical,
)


class PLOIGuidance(BaseSearchGuidance):
    """Minimal PLOI guidance class. """

    def __init__(self, model, graph_metadata, train_mode=False):
        super().__init__()
        self.model = model
        self.graph_metadata = graph_metadata
        self._last_processed_state = None
        self._last_object_scores = None
        self.train_mode = train_mode

    def score_object(self, obj, state):
        if state != self._last_processed_state:
            # Create input graph from state
            graph, node_to_objects = state_to_graph(state, self.graph_metadata)
            # Predict graph
            prediction = predict_graph_with_graphnetwork(self.model, graph)
            # Derive object scores
            object_scores = {
                o: prediction["nodes"][n][0] for n, o in node_to_objects.items()
            }
            self._last_object_scores = object_scores
            self._last_processed_state = state
        return self._last_object_scores[obj]


class HierarchicalGuidance(object):
    """Minimal hierarchical guidance class. """

    def __init__(self, room_model, object_model, graph_metadata, train_mode=False):
        super().__init__()
        self.room_model = room_model
        self.object_model = object_model
        self.graph_metadata = graph_metadata
        self._last_processed_state = None
        self._last_object_scores = None
        self.train_mode = train_mode

    def compute_scores(self, state):
        # Create input graph from state
        graph, _, node_to_objects = state_to_graph_hierarchical(
            state, self.graph_metadata
        )
        # Room and object level predictions
        room_scores, object_scores = predict_graph_with_graphnetwork_hierarchical(
            self.room_model, self.object_model, graph
        )

        room_to_objects_in_room_dict = (
            {}
        )  # Dictionary to map a room object to a list containing all objects within the room

        obj_to_room_score_dict = {}
        obj_to_obj_score_dict = {}
        for n, o in node_to_objects.items():
            obj_to_obj_score_dict[o] = object_scores[n].item()
            if o.var_type == "room":
                # Get the index of this node in the room graph. For that, simply
                # look up the global node idx (i.e., node idx in the object graph)
                # and determine its *index* in the `room_inds_global` list.
                room_idx_local = graph["room_graph"]["room_inds_global"].index(n)
                obj_to_room_score_dict[o] = room_scores[room_idx_local].item()
                # Form a list of all objects in the room
                # Store it in a dictionary, indexed by the room object
                objects_in_this_room = []
                for obj_in_room_idx in graph["room_graph"]["room_subnodes_global"][n]:
                    objects_in_this_room.append(node_to_objects[obj_in_room_idx])
                room_to_objects_in_room_dict[o] = objects_in_this_room

        return (
            obj_to_room_score_dict,
            obj_to_obj_score_dict,
            room_to_objects_in_room_dict,
            graph["room_graph"],
        )


class SceneGraphGuidance(object):
    """Minimal hierarchical guidance class. """

    def __init__(self, graph_metadata, train_mode=False):
        super().__init__()
        self.graph_metadata = graph_metadata

    def compute_scores(self, state):
        # Create input graph from state
        graph, _, node_to_objects = state_to_graph_hierarchical(
            state, self.graph_metadata
        )

        # Dictionary to map a room object to a list containing all objects within the room
        room_to_objects_in_room_dict = {}

        for n, o in node_to_objects.items():
            if o.var_type == "room":
                # Form a list of all objects in the room
                # Store it in a dictionary, indexed by the room object
                objects_in_this_room = []
                for obj_in_room_idx in graph["room_graph"]["room_subnodes_global"][n]:
                    objects_in_this_room.append(node_to_objects[obj_in_room_idx])
                room_to_objects_in_room_dict[o] = objects_in_this_room

        graph["room_graph"][
            "room_to_objects_in_room_dict"
        ] = room_to_objects_in_room_dict

        return graph["room_graph"]
