"""An incremental planner that samples more and more objects until
it finds a plan.
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


class IncrementalPlanner(Planner):
    """Sample objects by incrementally lowering a score threshold."""

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

        num_replanning_steps = 0

        start_time = time.time()

        # Get scores once.
        object_to_score = {
            obj: self._guidance.score_object(obj, state)
            for obj in state.objects
            if obj not in cur_objects
        }

        neural_net_time = time.time() - start_time

        # Initialize threshold.
        threshold = self._gamma
        for _ in range(self._max_iterations):
            # Find new objects by incrementally lower threshold.
            unused_objs = sorted(list(state.objects - cur_objects))
            new_objs = set()
            while unused_objs:
                # Geometrically lower threshold.
                if self._threshold_mode == "geometric":
                    threshold *= self._gamma
                elif self._threshold_mode == "linear":
                    threshold -= 0.01 * self._gamma
                # Linearly lower threshold
                # See if there are any new objects.
                new_objs = {o for o in unused_objs if object_to_score[o] > threshold}
                # If there are, try planning with them.
                if new_objs:
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
                    len(cur_objects), len(state.objects), threshold
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


class TopKPlanner(Planner):
    """Sample objects by taking the top K objects and gradually incrementing K."""

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
        K=50,
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
        self.K = K

    def __call__(self, domain, state, timeout):
        act_preds = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_preds, type_to_parent_types=domain.type_to_parent_types
        )
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
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
        start_time = time.time()
        if self._force_include_goal_objects:
            # Always start off considering objects in the goal.
            for lit in state.goal.literals:
                cur_objects |= set(lit.variables)
        # Get scores once.
        object_to_score = {
            obj: self._guidance.score_object(obj, state)
            for obj in state.objects
            if obj not in cur_objects
        }
        # Initialize threshold.
        threshold = self._gamma
        for _ in range(self._max_iterations):
            # Find new objects by incrementally lower threshold.
            unused_objs = sorted(list(state.objects - cur_objects))
            new_objs = set()
            while unused_objs:
                # Geometrically lower threshold.
                if self._threshold_mode == "geometric":
                    threshold *= self._gamma
                elif self._threshold_mode == "linear":
                    threshold -= 0.01 * self._gamma
                scores_for_unused_objects = np.array(
                    [object_to_score[o] for o in unused_objs]
                )
                scores_indices_low_to_high = np.argsort(scores_for_unused_objects)
                scores_indices_high_to_low = scores_indices_low_to_high[::-1]
                # Retain top K objects
                new_objs = {
                    unused_objs[scores_indices_high_to_low[i]]
                    for i in range(min(self.K, len(unused_objs)))
                }
                self.K = self.K + self.K

                # # See if there are any new objects.
                # new_objs = {o for o in unused_objs if object_to_score[o] > threshold}
                # If there are, try planning with them.
                if new_objs:
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
                    len(cur_objects), len(state.objects), threshold
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
                # Try again with more objects.
                continue
            return plan
        raise PlanningFailure("Plan not found! Reached max_iterations.")


class DistillationPlanner(Planner):
    """Sample objects by incrementally lowering a score threshold. Return PLOI logits for distillation objectives."""

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

    def __call__(self, domain, state, timeout):
        act_preds = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_preds, type_to_parent_types=domain.type_to_parent_types
        )
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
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
        start_time = time.time()
        if self._force_include_goal_objects:
            # Always start off considering objects in the goal.
            for lit in state.goal.literals:
                cur_objects |= set(lit.variables)
        object_score_logits = None
        for (
            obj
        ) in (
            state.objects
        ):  # Hack (i.e., passing a dummy object to reuse _guidance.score_object)
            _, object_score_logits = self._guidance.score_object(
                obj, state, return_logits=True
            )
            break
        # Get scores once.
        object_to_score = {
            obj: self._guidance.score_object(obj, state)
            for obj in state.objects
            if obj not in cur_objects
        }
        # Initialize threshold.
        threshold = self._gamma
        for _ in range(self._max_iterations):
            # Find new objects by incrementally lower threshold.
            unused_objs = sorted(list(state.objects - cur_objects))
            new_objs = set()
            while unused_objs:
                # Geometrically lower threshold.
                if self._threshold_mode == "geometric":
                    threshold *= self._gamma
                elif self._threshold_mode == "linear":
                    threshold -= 0.01 * self._gamma
                # Linearly lower threshold
                # See if there are any new objects.
                new_objs = {o for o in unused_objs if object_to_score[o] > threshold}
                # If there are, try planning with them.
                if new_objs:
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
                    len(cur_objects), len(state.objects), threshold
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
                # Try again with more objects.
                continue
            return plan, object_score_logits
        raise PlanningFailure("Plan not found! Reached max_iterations.")


class TrainWithPlanner(Planner):
    """Sample objects by incrementally lowering a score threshold. Use this for planner in-the-loop training"""

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

    def __call__(self, domain, state, timeout):
        act_preds = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_preds, type_to_parent_types=domain.type_to_parent_types
        )
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
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
        start_time = time.time()
        if self._force_include_goal_objects:
            # Always start off considering objects in the goal.
            for lit in state.goal.literals:
                cur_objects |= set(lit.variables)
        # Get scores once.
        object_score_logits, node_to_object_dict = self._guidance.score_object(
            state, return_logits=True
        )
        object_scores = torch.sigmoid(object_score_logits.detach())
        object_scores.requires_grad = False
        object_to_node_dict = {v: k for k, v in node_to_object_dict.items()}

        with torch.no_grad():

            # Log object indices which've been added this iteration (used in supervision)
            objects_added_this_iter = []

            # Initialize threshold.
            threshold = self._gamma
            num_objects_to_sample = min(20, len(list(state.objects)))
            for _ in range(self._max_iterations):
                # Find new objects by incrementally lower threshold.
                unused_objs = sorted(list(state.objects - cur_objects))
                new_objs = set()
                while unused_objs:
                    num_objects_to_sample = 30
                    if num_objects_to_sample >= len(unused_objs):
                        num_objects_to_sample = len(unused_objs)
                    unused_obj_inds = [object_to_node_dict[o] for o in unused_objs]
                    scores_for_unused_objects = object_scores[unused_obj_inds].view(-1)
                    categorical_dist = torch.distributions.categorical.Categorical(
                        logits=scores_for_unused_objects
                    )
                    probs2d = categorical_dist.probs.reshape(
                        -1, categorical_dist._num_events
                    )
                    # If code fails here, double check dims. Last dim of categorical_dist.probs must be > num_objects_to_sample.
                    # By default, categorical_dist.probs is of shape (1, N) -- so transposing it for torch.multinomial to work.
                    sampled_objects = torch.multinomial(
                        categorical_dist.probs, num_objects_to_sample
                    )
                    scores_for_sampled_objects = scores_for_unused_objects[
                        sampled_objects
                    ]
                    log_probs = categorical_dist.log_prob(sampled_objects)

                    new_objs = {
                        unused_objs[sampled_objects[idx]]
                        for idx in range(sampled_objects.numel())
                    }

                    # Geometrically lower threshold.
                    if self._threshold_mode == "geometric":
                        threshold *= self._gamma
                    elif self._threshold_mode == "linear":
                        threshold -= 0.01 * self._gamma

                    # Linearly lower threshold
                    # See if there are any new objects.
                    # new_objs = {o for o in unused_objs if object_to_score[o] > threshold}
                    # If there are, try planning with them.
                    if new_objs:
                        break
                cur_objects |= new_objs
                # Keep only literals referencing currently considered objects.
                cur_lits = set()
                for lit in state.literals:
                    if all(var in cur_objects for var in lit.variables):
                        cur_lits.add(lit)
                dummy_state = State(cur_lits, cur_objects, state.goal)
                # Try planning with only this object set.
                # print(
                #     "[Trying to plan with {} objects of {} total, "
                #     "threshold is {}...]".format(
                #         len(cur_objects), len(state.objects), threshold
                #     ),
                #     flush=True,
                # )
                try:
                    time_elapsed = time.time() - start_time
                    # Get a plan from base planner & validate it.
                    plan = self._planner(domain, dummy_state, timeout - time_elapsed)
                    if not validate_strips_plan(
                        domain_file=dom_file, problem_file=prob_file, plan=plan
                    ):
                        raise PlanningFailure("Invalid plan")
                    else:
                        # If plan was successful, log these objects as "critical"
                        objects_added_this_iter = [
                            object_to_node_dict[o] for o in new_objs
                        ]
                except PlanningFailure:
                    # Try again with more objects.
                    continue
                return plan, object_score_logits, objects_added_this_iter
            raise PlanningFailure("Plan not found! Reached max_iterations.")


# class TrainWithPlanner(Planner):
#     """Sample objects by incrementally lowering a score threshold. Use this for planner in-the-loop training"""

#     def __init__(
#         self,
#         is_strips_domain,
#         base_planner,
#         search_guider,
#         seed,
#         gamma=0.9,  # parameter for incrementing by score
#         threshold_mode="linear",  # geometric vs linear thresholding
#         max_iterations=1000,
#         force_include_goal_objects=True,
#     ):
#         super().__init__()
#         assert isinstance(base_planner, Planner)
#         print(
#             "Initializing {} with base planner {}, "
#             "guidance {}".format(
#                 self.__class__.__name__,
#                 base_planner.__class__.__name__,
#                 search_guider.__class__.__name__,
#             )
#         )
#         self._is_strips_domain = is_strips_domain
#         self._gamma = gamma
#         self._max_iterations = max_iterations
#         self._planner = base_planner
#         self._guidance = search_guider
#         self._rng = np.random.RandomState(seed=seed)
#         self._force_include_goal_objects = force_include_goal_objects
#         self._threshold_mode = threshold_mode

#     def __call__(self, domain, state, timeout):
#         act_preds = [domain.predicates[a] for a in list(domain.actions)]
#         act_space = LiteralSpace(
#             act_preds, type_to_parent_types=domain.type_to_parent_types
#         )
#         dom_file = tempfile.NamedTemporaryFile(delete=False).name
#         prob_file = tempfile.NamedTemporaryFile(delete=False).name
#         domain.write(dom_file)
#         lits = set(state.literals)
#         if not domain.operators_as_actions:
#             lits |= set(act_space.all_ground_literals(state, valid_only=False))
#         PDDLProblemParser.create_pddl_file(
#             prob_file,
#             state.objects,
#             lits,
#             "myproblem",
#             domain.domain_name,
#             state.goal,
#             fast_downward_order=True,
#         )
#         cur_objects = set()
#         start_time = time.time()
#         if self._force_include_goal_objects:
#             # Always start off considering objects in the goal.
#             for lit in state.goal.literals:
#                 cur_objects |= set(lit.variables)
#         # Get scores once.
#         object_scores, node_to_object_dict = self._guidance.score_object(state)
#         object_to_node_dict = {v: k for k, v in node_to_object_dict.items()}

#         # for k, v in node_to_object_dict.items():  # Sanity check for created dictionaries (skip this loop)
#         #     assert node_to_object_dict[k] == v
#         #     assert k == object_to_node_dict[v]

#         # object_to_score = {
#         #     obj: self._guidance.score_object(obj, state)
#         #     for obj in state.objects
#         #     if obj not in cur_objects
#         # }

#         R = 0
#         policy_saved_log_probs = []
#         policy_rewards = []

#         # Initialize threshold.
#         threshold = self._gamma
#         num_objects_to_sample = min(20, len(list(state.objects)))
#         for _ in range(self._max_iterations):
#             # Find new objects by incrementally lower threshold.
#             unused_objs = sorted(list(state.objects - cur_objects))
#             new_objs = set()
#             while unused_objs:
#                 num_objects_to_sample += 10
#                 if num_objects_to_sample >= len(unused_objs):
#                     num_objects_to_sample = len(unused_objs)
#                 unused_obj_inds = [object_to_node_dict[o] for o in unused_objs]
#                 scores_for_unused_objects = object_scores[unused_obj_inds].view(-1)
#                 categorical_dist = torch.distributions.categorical.Categorical(logits=scores_for_unused_objects)
#                 probs2d = categorical_dist.probs.reshape(-1, categorical_dist._num_events)
#                 # If code fails here, double check dims. Last dim of categorical_dist.probs must be > num_objects_to_sample.
#                 # By default, categorical_dist.probs is of shape (1, N) -- so transposing it for torch.multinomial to work.
#                 sampled_objects = torch.multinomial(categorical_dist.probs, num_objects_to_sample)
#                 scores_for_sampled_objects = scores_for_unused_objects[sampled_objects]
#                 log_probs = categorical_dist.log_prob(sampled_objects)

# #                 new_objs = {unused_objs[sampled_objects[idx]] for idx in range(sampled_objects.numel())}

# #                 # Geometrically lower threshold.
# #                 if self._threshold_mode == "geometric":
# #                     threshold *= self._gamma
# #                 elif self._threshold_mode == "linear":
# #                     threshold -= 0.01 * self._gamma

# #                 # Linearly lower threshold
# #                 # See if there are any new objects.
# #                 # new_objs = {o for o in unused_objs if object_to_score[o] > threshold}
# #                 # If there are, try planning with them.
# #                 if new_objs:
# #                     policy_saved_log_probs.append(log_probs)
# #                     break
# #             cur_objects |= new_objs
# #             # Keep only literals referencing currently considered objects.
# #             cur_lits = set()
# #             for lit in state.literals:
# #                 if all(var in cur_objects for var in lit.variables):
# #                     cur_lits.add(lit)
# #             dummy_state = State(cur_lits, cur_objects, state.goal)
# #             # Try planning with only this object set.
# #             # print(
# #             #     "[Trying to plan with {} objects of {} total, "
# #             #     "threshold is {}...]".format(
# #             #         len(cur_objects), len(state.objects), threshold
# #             #     ),
# #             #     flush=True,
# #             # )
# #             try:
# #                 time_elapsed = time.time() - start_time
# #                 # Get a plan from base planner & validate it.
# #                 plan = self._planner(domain, dummy_state, timeout - time_elapsed)
# #                 if not validate_strips_plan(
# #                     domain_file=dom_file, problem_file=prob_file, plan=plan
# #                 ):
# #                     raise PlanningFailure("Invalid plan")
# #                 policy_rewards.append(10)
# #             except PlanningFailure:
# #                 # Try again with more objects.
# #                 policy_rewards.append(-3)
# #                 continue
# #             return plan, policy_saved_log_probs, policy_rewards
# #         raise PlanningFailure("Plan not found! Reached max_iterations.")
