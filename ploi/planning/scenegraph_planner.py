"""
Our scenegraph-based planner that sparsifies a scenegraph ensuring a valid
scenegraph at each point.
"""

import shutil
import tempfile
import time
import warnings

import numpy as np
from pddlgym.parser import PDDLProblemParser
from pddlgym.spaces import LiteralSpace
from pddlgym.structs import State

from .planner import Planner, PlanningFailure
from .validate import validate_strips_plan


class SceneGraphPlanner(Planner):
    """Sample objects by incrementally lowering a score threshold."""

    def __init__(
        self, is_strips_domain, base_planner, guidance,
    ):
        super().__init__()
        assert isinstance(base_planner, Planner)
        self._is_strips_domain = is_strips_domain
        self._planner = base_planner
        self._guidance = guidance

    def __call__(self, domain, state, timeout, domain_file_global=None):
        self._planner.reset_statistics()
        act_preds = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_preds, type_to_parent_types=domain.type_to_parent_types
        )
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
        if domain_file_global is not None:
            shutil.copy(domain_file_global, dom_file)
        else:
            domain.write(dom_file)
        lits = set(state.literals)
        if not domain.operators_as_actions:
            lits |= set(act_space.all_ground_literals(state, valid_only=False))
        PDDLProblemParser.create_pddl_file(
            prob_file,
            state.objects,
            lits,
            "myproblem",
            domain.domain_name,
            state.goal,
            fast_downward_order=True,
        )

        cur_objects = set()

        # Always start off considering objects in the goal.
        for lit in state.goal.literals:
            cur_objects |= set(lit.variables)

        # Always include agent, bagslot, room
        for o in state.objects:
            if o.var_type in ["agent", "bagslot", "room"]:
                cur_objects.add(o)

        objs_to_add_tmp = set()
        for bin_lit in state.literals:
            if bin_lit.predicate.arity != 2:
                continue
            if bin_lit.predicate == "classrelation":
                # variables[0] -> iclass, variables[1] -> rclass
                if (
                    bin_lit.variables[0] in cur_objects
                    or bin_lit.variables[1] in cur_objects
                ):
                    objs_to_add_tmp.add(bin_lit.variables[0])
                    objs_to_add_tmp.add(bin_lit.variables[1])
        cur_objects |= objs_to_add_tmp

        # Get scores once.
        room_graph = self._guidance.compute_scores(state)

        # Variable to long number of replanning attempts
        num_replanning_steps = 0

        start_time = time.time()

        # Get the parent object of each object
        parent_object_dict = room_graph["parent_object_dict"]
        room_place_dict = room_graph["room_place_dict"]
        place_location_dict = room_graph["place_location_dict"]
        location_place_dict = room_graph["location_place_dict"]
        location_to_placelocation_dict = room_graph["location_to_placelocation_dict"]
        object_location_dict = room_graph["object_location_dict"]
        object_place_dict = room_graph["object_place_dict"]
        # room_to_objects_in_room_dict = room_graph["room_to_objects_in_room_dict"]

        receptacles_in_lifted_goal = (
            set()
        )  # All receptacles that have a receptacleclass predicate true
        items_in_lifted_goal = set()  # All items that have an itemcass predicate true
        objs_to_add_tmp = set()
        for bin_lit in state.literals:
            if bin_lit.predicate.arity != 2:
                continue
            if bin_lit.variables[1] in cur_objects:
                if bin_lit.predicate == "receptacleclass":
                    # variables[0] -> receptacle; variables[1] -> rclass
                    receptacles_in_lifted_goal.add(bin_lit.variables[0])
                elif bin_lit.predicate == "itemclass":
                    # variables[0] -> item; variables[1] -> rclass
                    items_in_lifted_goal.add(bin_lit.variables[0])
            # if bin_lit.predicate == "classrelation":
            #     # variables[0] -> iclass, variables[1] -> rclass
            #     objs_to_add_tmp.add(bin_lit.variables[0])
            #     objs_to_add_tmp.add(bin_lit.variables[1])
        cur_objects |= receptacles_in_lifted_goal
        cur_objects |= items_in_lifted_goal

        objs_to_add_tmp = set()
        for o in cur_objects:
            # print(o.var_type)
            if o.var_type in ["room"]:
                objs_to_add_tmp.add(room_place_dict[o])
            if o.var_type in ["place"]:
                objs_to_add_tmp.add(place_location_dict[o])
            if o.var_type in ["receptacle", "item"]:
                objs_to_add_tmp.add(object_place_dict[o])
                objs_to_add_tmp.add(object_location_dict[o])
                l = object_location_dict[o]
                objs_to_add_tmp.add(location_to_placelocation_dict[l])
            if o.var_type in ["location"]:
                objs_to_add_tmp.add(location_place_dict[o])
                objs_to_add_tmp.add(place_location_dict[location_place_dict[o]])
                objs_to_add_tmp.add(location_to_placelocation_dict[o])
            if o.var_type in ["agent"]:
                objs_to_add_tmp.add(object_place_dict[o])
                objs_to_add_tmp.add(object_location_dict[o])
                l = object_location_dict[o]
                objs_to_add_tmp.add(location_to_placelocation_dict[l])
            # Loop over iclass and add all parents to the object set
            # Loop over rclass and add all parents ot the object set
        cur_objects |= objs_to_add_tmp

        # For each object, add its location to the current object set
        objs_to_add_tmp = set()
        for o in cur_objects:
            # if o.var_type in ["agent", "bagslot", "receptacle", "item"]:
            if o.var_type in ["place"]:
                objs_to_add_tmp.add(place_location_dict[o])
            else:
                if o.var_type not in [
                    "location",
                    "room",
                    "bagslot",
                    "rclass",
                    "iclass",
                ]:
                    objs_to_add_tmp.add(object_location_dict[o])
                    objs_to_add_tmp.add(object_place_dict[o])
                if o.var_type not in [
                    "room",
                    "location",
                    "bagslot",
                    "rclass",
                    "iclass",
                ]:
                    l = object_location_dict[o]
                    objs_to_add_tmp.add(location_to_placelocation_dict[l])
        cur_objects |= objs_to_add_tmp

        ancestors_of_cur_objects = set()
        for no in cur_objects:
            # For each added object, make sure to add all of its ancestors (so it's reachable)
            if no in parent_object_dict.keys():
                par1 = parent_object_dict[no]  # parent of the cur obj
                ancestors_of_cur_objects.add(par1)
                # print("Lvl 1")
                if par1 in parent_object_dict.keys():
                    par2 = parent_object_dict[par1]  # parent of parent of cur obj
                    ancestors_of_cur_objects.add(par2)
                    # print("Lvl 2")
                    if par2 in parent_object_dict.keys():
                        par3 = parent_object_dict[par2]
                        ancestors_of_cur_objects.add(par3)
                        # print("Lvl 3")
                        if par3 in parent_object_dict.keys():
                            par4 = parent_object_dict[par3]
                            ancestors_of_cur_objects.add(par4)
                            # print("Lvl 4")
                            if par4 in parent_object_dict.keys():
                                print("NEED LEVEL 5!!!")
                                # quit()
        cur_objects |= ancestors_of_cur_objects

        # Keep only literals referencing currently considered objects.
        cur_lits = set()
        for lit in state.literals:
            if all(
                var in cur_objects for var in lit.variables
            ):  # or (lit.predicate in ["roomplace", "placelocation", "roomsconnected", "itematlocation", "inroom", "inplace", "receptacleatlocation", "inreceptacle", "inanyreceptacle", "receptacleopeningtype", "receptacleopened", "holds", "holdsany"]):
                cur_lits.add(lit)
        dummy_state = State(cur_lits, cur_objects, state.goal)

        # Try planning with only this object set.
        print(
            f"[Trying to plan with {len(cur_objects)} objects of {len(state.objects)} total.]",
            flush=True,
        )
        try:
            time_elapsed = time.time() - start_time
            # Get a plan from base planner & validate it.
            plan = self._planner(domain, dummy_state, timeout - time_elapsed)
            if not validate_strips_plan(
                domain_file=dom_file, problem_file=prob_file, plan=plan
            ):
                warnings.warn("Invalid plan")
        except PlanningFailure:
            # Try again with more objects.
            print("Plan failed with the current object set.")
            return None, None
            # quit()
        planner_stats = self._planner.get_statistics().copy()
        planner_stats["objects_used"] = len(cur_objects)
        planner_stats["objects_total"] = len(state.objects)
        planner_stats["num_replanning_steps"] = num_replanning_steps
        planner_stats["neural_net_time"] = 0.0

        return plan, planner_stats
