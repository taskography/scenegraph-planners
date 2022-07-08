"""
Use SCRUB to generate reduced 3D scene graph problem files for use with planning problems.
"""

import argparse
import json
import os
import pickle
import time
import warnings

import pddlgym
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from ploi.argparsers import get_ploi_argument_parser
from ploi.datautils import (
    collect_training_data,
    create_graph_dataset,
    create_graph_dataset_hierarchical,
    GraphDictDataset,
)
from ploi.guiders import SceneGraphGuidance
from ploi.modelutils import GraphNetwork
from ploi.planning import FD

# from ploi.planning.incremental_hierarchical_planner import IncrementalHierarchicalPlanner
from ploi.planning.scenegraph_planner import SceneGraphPlanner
from ploi.planning.scrubber import SCRUBber
from ploi.traineval import test_planner


def _create_planner(planner_name):
    if planner_name == "fd-lama-first":
        return FD(alias_flag="--alias lama-first")
    if planner_name == "fd-opt-lmcut":
        return FD(alias_flag="--alias seq-opt-lmcut")
    raise ValueError(f"Uncrecognized planner name {planner_name}")


def scrub_problems_in_domain(
    planner,
    domain_name,
    num_problems,
    timeout,
    savedir,
    mode="test",
    debug_mode=False,
    all_problems=False,
    no_scrub=False,
    write_domain_file=False,
):
    print(f"Writing scrubbed domains with prune = {not no_scrub}")
    # In debug mode, use train problems for testing too (False by default)
    env = None
    if mode == "test":
        env = pddlgym.make("PDDLEnv{}Test-v0".format(domain_name))
    else:
        env = pddlgym.make("PDDLEnv{}-v0".format(domain_name))
    if debug_mode:
        warnings.warn(
            "WARNING: Running in debug mode (i.e., testing on train problems)"
        )
        env = pddlgym.make("PDDLEnv{}-v0".format(domain_name))
    num_problems = min(num_problems, len(env.problems))
    # If `all_problems` is set to True, override num_problems
    if all_problems:
        num_problems = len(env.problems)

    if write_domain_file:
        domain_file_name = env.domain.domain_fname
        new_domain_name = env.domain.domain_name.lower() + "scrub"
        print(domain_file_name, new_domain_name)
        fp = open(domain_file_name, "r")
        lines = fp.readlines()
        try:
            lines[1] = lines[1].replace(env.domain.domain_name.lower(), new_domain_name)
        except Exception as e:
            pass
        fp.close()
        parent_of_savedir = os.path.dirname(savedir)
        fp = open(os.path.join(parent_of_savedir, f"{new_domain_name}.pddl"), "w")
        for l in lines:
            fp.write(l)
        fp.close()

    stats_to_log = ["num_node_expansions", "plan_length", "search_time", "total_time"]
    num_timeouts = 0
    num_failures = 0
    num_invalidated_plans = 0
    run_stats = []
    for problem_idx in trange(num_problems):
        env.fix_problem_index(problem_idx)
        state, _ = env.reset()

        domain_file = env.domain.domain_fname
        problem_file = env.problems[problem_idx].problem_fname
        problem_unique_name = env.problems[problem_idx].problem_name

        planner(
            env.domain,
            state,
            timeout,
            problem_file,
            savedir,
            domain_file_global=env._domain_file,
            no_scrub=no_scrub,
            problem_unique_name=problem_unique_name,
        )


def _scrub_domain(args):
    # Seed RNG
    torch.manual_seed(args.seed)

    # Create dir to log files to
    args.expdir = os.path.join(args.logdir, args.expid)
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir, exist_ok=True)

    # Directory to write the domain files to
    args.savedir = f"{args.logdir}/scrubbed_domains/{args.domain.lower()}scrub"
    if args.mode == "test":
        args.savedir = args.savedir + "_test"
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir, exist_ok=True)
    print(f"Writing scrubbed problems to {args.savedir}")

    # Capitalize the first letter of the domain name
    args.domain = args.domain.capitalize()

    # This datafile is the same for ploi and hierarchical variants
    args.datafile = os.path.join(args.logdir, f"ploi_{args.domain}.pkl")

    print(f"Domain: {args.domain}")
    print(f"Train planner: {args.train_planner_name}")
    print(f"Test planner: {args.eval_planner_name}")

    eval_planner = _create_planner(args.eval_planner_name)
    is_strips_domain = True

    train_planner = _create_planner(args.train_planner_name)

    training_data = None
    print("Collecting training data")
    if not os.path.exists(args.datafile) or args.force_collect_data:
        training_data = collect_training_data(
            args.domain, train_planner, num_train_problems=args.num_train_problems
        )
        with open(args.datafile, "wb") as f:
            pickle.dump(training_data, f)
    else:
        print("Training data already found on disk")
    with open(args.datafile, "rb") as f:
        print("Loading training data from file")
        training_data = pickle.load(f)

    graphs_inp, graphs_tgt, graph_metadata = create_graph_dataset(training_data)

    scenegraph_guidance = SceneGraphGuidance(graph_metadata)
    planner_to_eval = SCRUBber(
        is_strips_domain=is_strips_domain,
        base_planner=eval_planner,
        guidance=scenegraph_guidance,
    )
    no_scrub = False
    write_domain_file = False
    if args.mode == "train":
        no_scrub = True  # Do not prune in train mode (we need to retain extraneous objects to train PLOI variants)
        write_domain_file = True
    else:
        write_domain_file = False
        no_scrub = False
    scrub_problems_in_domain(
        planner_to_eval,
        args.domain,
        args.num_test_problems,
        args.timeout,
        args.savedir,
        args.mode,
        all_problems=args.all_problems,
        no_scrub=no_scrub,
        write_domain_file=write_domain_file,
    )


if __name__ == "__main__":

    parser = get_ploi_argument_parser()

    parser.add_argument(
        "--all-problems", action="store_true", help="SCRUB all problems in domain"
    )

    args = parser.parse_args()

    domains_to_scrub = [
        "taskographyv2tiny1",  # 'taskographyv2medium1', 'taskographyv2tiny2', 'taskographyv2medium2',
        # 'taskographyv2tiny10', 'taskographyv2medium10', 'taskographyv3tiny10bagslots10', 'taskographyv3medium10bagslots10',
        # 'taskographyv3tiny10bagslots3', 'taskographyv3medium10bagslots3', 'taskographyv3tiny10bagslots5', 'taskographyv3medium10bagslots5',
        # 'taskographyv3tiny10bagslots7', 'taskographyv3medium10bagslots7',
        # 'taskographyv4tiny5', 'taskographyv4medium5', 'taskographyv5tiny5bagslots5', 'taskographyv5medium5bagslots5'
    ]

    # (Un)comment this to generate scrubbed domains for all entries in domains_to_scrub
    for d in domains_to_scrub:
        args.domain = d
        for mode in ["train", "test"]:
            args.mode = mode
            _scrub_domain(args)

    # (Un)comment this to generate entries for __init__.py in pddlgym (to register these envs)
    for d in domains_to_scrub:
        name = d.lower() + "scrub"
        PDDL_GYM_STRING = f"(\n    \"{name}\",\n    {{\n        'operators_as_actions': True,\n        'dynamic_action_space':True,\n    }}\n),"
        print(PDDL_GYM_STRING)
