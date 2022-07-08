import argparse
import json
import os
import pickle
import warnings

import torch
from torch.utils.data import DataLoader
from ploi.argparsers import get_ploi_argument_parser
from ploi.datautils import (
    collect_training_data,
    create_graph_dataset,
    create_graph_dataset_hierarchical,
    GraphDictDataset,
)
from ploi.guiders import HierarchicalGuidance, PLOIGuidance, SceneGraphGuidance
from ploi.modelutils import GraphNetwork
from ploi.planning import FD, IncrementalPlanner
from ploi.planning.incremental_hierarchical_planner import (
    IncrementalHierarchicalPlanner,
)
from ploi.planning.scenegraph_planner import SceneGraphPlanner
from ploi.traineval import (
    test_planner,
    train_model_graphnetwork,
    train_model_hierarchical,
)


def _create_planner(planner_name):
    if planner_name == "fd-lama-first":
        return FD(alias_flag="--alias lama-first")
    if planner_name == "fd-opt-lmcut":
        return FD(alias_flag="--alias seq-opt-lmcut")
    raise ValueError(f"Uncrecognized planner name {planner_name}")


if __name__ == "__main__":

    parser = get_ploi_argument_parser()

    parser.add_argument(
        "--all-problems",
        action="store_true",
        help="Run testing on all problems in domain",
    )

    args = parser.parse_args()

    # Seed RNG
    torch.manual_seed(args.seed)

    # Create dir to log files to
    args.expdir = os.path.join(args.logdir, args.expid)
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir, exist_ok=True)

    # Capitalize the first letter of the domain name
    args.domain = args.domain.capitalize()

    # This datafile is the same for ploi and hierarchical variants
    args.datafile = os.path.join(args.logdir, f"ploi_{args.domain}.pkl")
    if args.domain.endswith("scrub"):
        args.datafile = os.path.join(args.logdir, f"ploi_{args.domain[:-5]}.pkl")

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

    graphs_inp, graphs_tgt, graph_metadata = None, None, None
    if args.method in ["hierarchical"]:
        graphs_inp, graphs_tgt, graph_metadata = create_graph_dataset_hierarchical(
            training_data
        )
    else:
        graphs_inp, graphs_tgt, graph_metadata = create_graph_dataset(training_data)

    # Use 10% for validation
    num_validation = max(1, int(len(graphs_inp) * 0.1))
    train_graphs_input = graphs_inp[num_validation:]
    train_graphs_target = graphs_tgt[num_validation:]
    valid_graphs_input = graphs_inp[:num_validation]
    valid_graphs_target = graphs_tgt[:num_validation]
    # Set up dataloaders
    graph_dataset = GraphDictDataset(train_graphs_input, train_graphs_target)
    graph_dataset_val = GraphDictDataset(valid_graphs_input, valid_graphs_target)
    datasets = {"train": graph_dataset, "val": graph_dataset_val}

    args.num_node_features_object = datasets["train"][0]["graph_input"]["nodes"].shape[
        -1
    ]
    args.num_edge_features_object = datasets["train"][0]["graph_input"]["edges"].shape[
        -1
    ]

    object_level_model = GraphNetwork(
        n_features=args.num_node_features_object,
        n_edge_features=args.num_edge_features_object,
        n_hidden=16,
    )

    if args.method == "scenegraph":

        if args.mode == "train":
            import sys

            warnings.warn("No training mode for scenegraph planner.")
            sys.exit(0)

        scenegraph_guidance = SceneGraphGuidance(graph_metadata)
        planner_to_eval = SceneGraphPlanner(
            is_strips_domain=is_strips_domain,
            base_planner=eval_planner,
            guidance=scenegraph_guidance,
        )
        test_stats, global_stats = test_planner(
            planner_to_eval,
            args.domain,
            args.num_test_problems,
            args.timeout,
            all_problems=args.all_problems,
        )

        statsfile = os.path.join(args.expdir, "scenegraph_test_stats.py")
        json_string = json.dumps(test_stats, indent=4)
        json_string = "STATS = " + json_string
        with open(statsfile, "w") as f:
            f.write(json_string)

        globalstatsfile = os.path.join(
            args.expdir, f"{args.domain.lower()}_{args.method}_test.json"
        )
        with open(globalstatsfile, "w") as fp:
            json.dump(global_stats, fp, indent=4, sort_keys=True)

    elif args.method == "hierarchical":

        args.num_node_features_room = datasets["train"][0]["graph_input"]["room_graph"][
            "nodes"
        ].shape[-1]
        args.num_edge_features_room = datasets["train"][0]["graph_input"]["room_graph"][
            "edges"
        ].shape[-1]

        room_level_model = GraphNetwork(
            n_features=args.num_node_features_room,
            n_edge_features=args.num_edge_features_room,
            n_hidden=32,
            # dropout=0.2,
        )

        if args.mode == "train":

            optimizer_room = torch.optim.Adam(room_level_model.parameters(), lr=1e-4)
            optimizer_object = torch.optim.Adam(
                object_level_model.parameters(), lr=1e-3
            )
            pos_weight = args.pos_weight * torch.ones([1])
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            room_level_model_dict = train_model_hierarchical(
                room_level_model,
                datasets,
                criterion=torch.nn.BCEWithLogitsLoss(pos_weight=2 * torch.ones([1])),
                optimizer=optimizer_room,
                use_gpu=False,
                epochs=args.epochs,
                save_folder=args.expdir,
                model_type="room",
                eval_every=10,
            )
            object_level_model_dict = train_model_hierarchical(
                object_level_model,
                datasets,
                criterion=criterion,
                optimizer=optimizer_object,
                use_gpu=False,
                epochs=args.epochs,
                save_folder=args.expdir,
                model_type="object",
            )
            room_level_model.load_state_dict(room_level_model_dict)
            object_level_model.load_state_dict(object_level_model_dict)

        elif args.mode == "test":

            with torch.no_grad():

                room_model_outfile = os.path.join(args.expdir, "room_best.pt")
                object_model_outfile = os.path.join(args.expdir, "object_best.pt")
                room_level_model.load_state_dict(torch.load(room_model_outfile))
                object_level_model.load_state_dict(torch.load(object_model_outfile))
                print(
                    f"Loaded saved models from {room_model_outfile}, {object_model_outfile}"
                )

                hierarchical_guider = HierarchicalGuidance(
                    room_level_model, object_level_model, graph_metadata
                )
                planner_to_eval = IncrementalHierarchicalPlanner(
                    is_strips_domain=is_strips_domain,
                    base_planner=eval_planner,
                    search_guider=hierarchical_guider,
                    seed=args.seed,
                    gamma=args.gamma,
                    threshold_mode="geometric",
                    # force_include_goal_objects=False,
                )

                test_stats, global_stats = test_planner(
                    planner_to_eval,
                    args.domain,
                    args.num_test_problems,
                    args.timeout,
                    all_problems=args.all_problems,
                )

                statsfile = os.path.join(args.expdir, "hierarchical_test_stats.py")
                json_string = json.dumps(test_stats, indent=4)
                json_string = "STATS = " + json_string
                with open(statsfile, "w") as f:
                    f.write(json_string)
                    # json.dump(test_stats, f, indent=4)

                globalstatsfile = os.path.join(
                    args.expdir, f"{args.domain.lower()}_{args.method}_test.json"
                )
                with open(globalstatsfile, "w") as fp:
                    json.dump(global_stats, fp, indent=4, sort_keys=True)

    elif args.method == "ploi":

        # PLOI training / testing

        args.num_node_features = datasets["train"][0]["graph_input"]["nodes"].shape[-1]
        args.num_edge_features = datasets["train"][0]["graph_input"]["edges"].shape[-1]

        model = GraphNetwork(
            n_features=args.num_node_features,
            n_edge_features=args.num_edge_features,
            n_hidden=16,
        )

        print("====================================")
        print(f"==== Expid: {args.expid} ==========")
        print("====================================")

        if args.mode == "train":
            """
            Train PLOI on pre-cached dataset of states and targets
            """
            if not args.load_model:
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                pos_weight = args.pos_weight * torch.ones([1])
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                model_dict = train_model_graphnetwork(
                    model,
                    datasets,
                    criterion=criterion,
                    optimizer=optimizer,
                    use_gpu=False,
                    epochs=args.epochs,
                    save_folder=args.expdir,
                )
                model.load_state_dict(model_dict)

        if args.mode == "test":
            """
            Test phase
            """

            model_outfile = os.path.join(args.expdir, "object_best.pt")
            try:
                object_level_model.load_state_dict(torch.load(model_outfile))
                print(f"Loaded saved model from {model_outfile}")
            except Exception as e1:
                try:
                    object_level_model.load_state_dict(
                        torch.load(os.path.join(args.expdir, "best.pt"))
                    )
                except Exception as e2:
                    raise IOError(f"No model file {model_outfile} or best.pt")

            ploiguider = PLOIGuidance(object_level_model, graph_metadata)
            planner_to_eval = IncrementalPlanner(
                is_strips_domain=is_strips_domain,
                base_planner=eval_planner,
                search_guider=ploiguider,
                seed=args.seed,
                gamma=args.gamma,
                # force_include_goal_objects=False,
            )

            test_stats, global_stats = test_planner(
                planner_to_eval,
                args.domain,
                args.num_test_problems,
                args.timeout,
                all_problems=args.all_problems,
            )
            statsfile = os.path.join(args.expdir, "ploi_test_stats.py")
            json_string = json.dumps(test_stats, indent=4)
            json_string = "STATS = " + json_string
            with open(statsfile, "w") as f:
                f.write(json_string)

            globalstatsfile = os.path.join(
                args.expdir, f"{args.domain.lower()}_{args.method}_test.json"
            )
            with open(globalstatsfile, "w") as fp:
                json.dump(global_stats, fp, indent=4, sort_keys=True)
