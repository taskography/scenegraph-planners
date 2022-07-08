"""
An incremental hierarchical planner that samples more and more objects along
multiple levels of the scene graph hierarchicy until it finds a plan.
"""

import shutil
import tempfile
import time

import torch
import numpy as np
from pddlgym.parser import PDDLProblemParser
from pddlgym.spaces import LiteralSpace
from pddlgym.structs import State

from .planner import Planner, PlanningFailure
from .validate import validate_strips_plan


class SeekPlanner(Planner):
    """One class to implement all SEEK experiments. """

    def __init__(
        self,
        is_strips_domain,
        base_planner,
        search_guider,
        seed,
        gamma=0.9,  # parameter for incrementing by score
        threshold_mode="geometric",  # geometric vs linear thresholding
        max_iterations=1000,
        force_include_goal_objects=True,
        scoring_mode="ploi",
        use_seek=True,
        seek_threshold=0.5,
    ):
        super().__init__()
        assert isinstance(base_planner, Planner)
        print(
            "Initializing {} with base planner {}, "
            "guidance {}".format(
                self.__class__.__name__,
                base_planner.__class__.__name__,
                search_guider.__class__.__name__,
            )
        )
        self._is_strips_domain = is_strips_domain
        self._gamma = gamma
        self._max_iterations = max_iterations
        self._planner = base_planner
        self._guidance = search_guider
        self._rng = np.random.RandomState(seed=seed)
        self._force_include_goal_objects = force_include_goal_objects
        self._threshold_mode = threshold_mode
        self._scoring_mode = scoring_mode
        self._use_seek = use_seek
        self._seek_threshold = seek_threshold

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

        if self._force_include_goal_objects:
            # Always start off considering objects in the goal.
            for lit in state.goal.literals:
                cur_objects |= set(lit.variables)
        start_time = time.time()
        # Get scores once.
        (
            obj_to_room_level_score,
            obj_to_object_level_score,
            room_to_objects_in_room_dict,
            room_graph,
        ) = self._guidance.compute_scores(state)
        neural_net_time = time.time() - start_time

        # Get the parent object of each object
        parent_object_dict = room_graph["parent_object_dict"]

        # object_to_score = {
        #     obj: self._guidance.score_object(obj, state)
        #     for obj in state.objects
        #     if obj not in cur_objects
        # }

        # Log number of replanning attempts
        num_replanning_steps = 0

        start_time = time.time()

        # Initialize threshold.
        room_threshold = 0.9  # self._gamma
        object_threshold = self._gamma

        for iter in range(self._max_iterations):

            objects_to_ignore = set()
            if self._scoring_mode == "hierarchical":
                # Threshold on rooms
                room_threshold = room_threshold - 0.1  # * self._gamma

                # Compute the set of rooms that can be ignored or discarded
                rooms_to_ignore = set()
                for r, roomscore in obj_to_room_level_score.items():
                    if roomscore < room_threshold:
                        rooms_to_ignore.add(r)

                # Get the set of objects in the rooms to be ignored. We ignore these objects
                objects_to_ignore = set()
                for r in rooms_to_ignore:
                    for so in room_to_objects_in_room_dict[r]:
                        if obj_to_object_level_score[so] < object_threshold:
                            objects_to_ignore.add(so)

            # Find new objects by incrementally lower threshold.
            unused_objs = sorted(
                list((state.objects - cur_objects) - objects_to_ignore)
            )
            # print(len(state.objects - cur_objects), len((state.objects - cur_objects) - objects_to_ignore), len(objects_to_ignore))
            # unused_objs = sorted(list(state.objects - cur_objects))
            new_objs = set()
            while unused_objs:

                # Relax threshold.
                if self._threshold_mode == "geometric":
                    object_threshold *= self._gamma
                elif self._threshold_mode == "linear":
                    object_threshold -= 0.1 * self._gamma

                # See if there are any new objects.
                if self._scoring_mode in ["ploi", "hierarchical"]:
                    new_objs = {
                        o
                        for o in unused_objs
                        if obj_to_object_level_score[o] > object_threshold
                    }
                elif self._scoring_mode in ["random"]:
                    new_objs = set()
                    for o in unused_objs:
                        if torch.rand(1).item() > self._seek_threshold:
                            new_objs.add(o)

                ancestors_of_new_objs = set()
                if self._use_seek:
                    for no in new_objs:
                        # For each added object, make sure to add all of its ancestors (so it's reachable)
                        if no in parent_object_dict.keys():
                            par1 = parent_object_dict[no]  # parent of the cur obj
                            ancestors_of_new_objs.add(par1)
                            # print("Lvl 1")
                            if par1 in parent_object_dict.keys():
                                par2 = parent_object_dict[
                                    par1
                                ]  # parent of parent of cur obj
                                ancestors_of_new_objs.add(par2)
                                # print("Lvl 2")
                                if par2 in parent_object_dict.keys():
                                    par3 = parent_object_dict[par2]
                                    ancestors_of_new_objs.add(par3)
                                    # print("Lvl 3")
                                    if par3 in parent_object_dict.keys():
                                        par4 = parent_object_dict[par3]
                                        ancestors_of_new_objs.add(par4)
                                        # print("Lvl 4")
                                        if par4 in parent_object_dict.keys():
                                            print("NEED LEVEL 5!!!")
                                            # quit()
                new_objs |= ancestors_of_new_objs
                # unreachable_objects = set()
                # for no in new_objs:
                #     # For each added object, determine which ones are unreachable
                #     if no in parent_object_dict.keys():
                #         par = parent_object_dict[no]
                #         if par not in new_objs:
                #             unreachable_objects.add(par)
                #         if par in parent_object_dict.keys():
                #             par_par = parent_object_dict[par]
                #             if par_par not in new_objs:
                #                 unreachable_objects.add(par_par)
                #             if par_par in parent_object_dict.keys():
                #                 par_par_par = parent_object_dict[par_par]
                #                 if par_par_par not in new_objs:
                #                     unreachable_objects.add(par_par_par)
                # new_objs = new_objs - unreachable_objects

                # If there are new objects, try planning with them.
                if new_objs:
                    # if len(new_objs) > 10:
                    break
            cur_objects |= new_objs
            # Keep only literals referencing currently considered objects.
            cur_lits = set()
            for lit in state.literals:
                if all(var in cur_objects for var in lit.variables):
                    cur_lits.add(lit)
            dummy_state = State(cur_lits, cur_objects, state.goal)
            # Try planning with only this object set.
            print(
                "[Trying to plan with {} objects of {} total, "
                "threshold is {}...]".format(
                    len(cur_objects), len(state.objects), object_threshold
                ),
                flush=True,
            )
            try:
                time_elapsed = time.time() - start_time
                # Get a plan from base planner & validate it.
                plan = self._planner(domain, dummy_state, timeout - time_elapsed)
                if not validate_strips_plan(
                    domain_file=dom_file, problem_file=prob_file, plan=plan
                ):
                    raise PlanningFailure("Invalid plan")
            except PlanningFailure:
                num_replanning_steps += 1
                # Try again with more objects.
                continue
            planner_stats = self._planner.get_statistics().copy()
            planner_stats["objects_used"] = len(cur_objects)
            planner_stats["objects_total"] = len(state.objects)
            planner_stats["neural_net_time"] = neural_net_time
            planner_stats["num_replanning_steps"] = num_replanning_steps
            return plan, planner_stats
        raise PlanningFailure("Plan not found! Reached max_iterations.")
