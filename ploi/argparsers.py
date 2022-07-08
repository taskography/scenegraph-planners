import argparse


def get_ploi_argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--method",
        type=str,
        choices=["scenegraph", "hierarchical", "ploi"],
        default="scenegraph",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "visualize"],
        default="train",
        help="Mode to run the script in",
    )

    parser.add_argument(
        "--domain",
        type=str,
        default="Taskographyv2tiny1",
        help="Name of the pddlgym domain to use.",
    )
    parser.add_argument(
        "--train-planner-name",
        type=str,
        choices=["fd-lama-first", "fd-opt-lmcut"],
        default="fd-lama-first",
        help="Train planner to use",
    )
    parser.add_argument(
        "--eval-planner-name",
        type=str,
        choices=["fd-lama-first", "fd-opt-lmcut"],
        default="fd-lama-first",
        help="Eval planner to use",
    )
    parser.add_argument(
        "--num-train-problems", type=int, default=25, help="Number of train problems"
    )
    parser.add_argument(
        "--num-test-problems", type=int, default=5, help="Number of test problems"
    )
    parser.add_argument(
        "--do-incremental-planning",
        action="store_true",
        help="Whether or not to do incremental planning",
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout for test-time planner"
    )

    parser.add_argument(
        "--expid", type=str, default="debug", help="Unique exp id to log data to"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="cache",
        help="Directory to store all expt logs in",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda:0"],
        default="cpu",
        help="torch.device argument",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        choices=["bce"],
        default="bce",
        help="Loss function to use",
    )

    parser.add_argument(
        "--pos-weight",
        type=float,
        default=10.0,
        help="Weight for the positive class in binary cross-entropy computation",
    )
    parser.add_argument(
        "--epochs", type=int, default=1001, help="Number of epochs to run training for"
    )
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
    parser.add_argument(
        "--load-model", action="store_true", help="Path to load model from"
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Number of iterations after which to print training progress.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Value of importance threshold (gamma) for PLOI.",
    )

    parser.add_argument(
        "--force-collect-data",
        action="store_true",
        help="Force data collection (ignore pre-cached datasets).",
    )

    return parser
