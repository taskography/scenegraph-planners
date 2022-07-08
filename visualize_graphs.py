import argparse
import json
import os
import pickle

import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from ploi.argparsers import get_ploi_argument_parser
from ploi.datautils import (
    collect_training_data,
    create_graph_dataset,
    create_graph_dataset_hierarchical,
    GraphDictDataset,
)
from ploi.guiders import HierarchicalGuidance, PLOIGuidance
from ploi.modelutils import GraphNetwork
from ploi.planning import FD, IncrementalPlanner
from ploi.planning.incremental_hierarchical_planner import (
    IncrementalHierarchicalPlanner,
)
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


def graph_attrs_from_torch_to_numpy(graph):
    if "nodes" in graph.keys():
        graph["nodes"] = graph["nodes"].numpy()
    if "edges" in graph.keys():
        graph["edges"] = graph["edges"].numpy()
    if "senders" in graph.keys():
        graph["senders"] = graph["senders"].numpy()
    if "receivers" in graph.keys():
        graph["receivers"] = graph["receivers"].numpy()
    return graph


def visualize_graph(
    input_graph,
    output_graph,
    outfile=None,
    node_color_fn=None,
    edge_color_fn=None,
    domain_name=None,
    problem_idx=None,
    **kwargs,
):

    """Draw input and output graphs side by side with networkx"""
    fig, axes = plt.subplots(1, 1)
    plot_title = "Scene graph"
    if domain_name is not None:
        plot_title = f"{plot_title} {domain_name}"
        if problem_idx is not None:
            plot_title = f"{plot_title} Problem {problem_idx}"
    axes.set_title(plot_title)

    if node_color_fn is None:
        node_color_fn = lambda *args: "white"
    if edge_color_fn is None:
        edge_color_fn = lambda *args: "black"

    g = input_graph

    G = nx.DiGraph()
    # Add nodes with colors
    for node in range(g["n_node"]):
        # color = node_color_fn(g, node, g["nodes"][node])
        color = "white"
        if output_graph["nodes"][node] == 1:
            color = "black"
        G.add_node(node, color=color)
    node_color_map = [G.nodes[u]["color"] for u in G.nodes()]
    # Add edges with colors
    for u, v, attrs in zip(g["senders"], g["receivers"], g["edges"]):
        color = edge_color_fn(g, u, v, attrs)
        G.add_edge(u, v, color=color)
    edge_color_map = [G[u][v]["color"] for u, v in G.edges()]

    pos = nx.spring_layout(G, iterations=100, seed=0)
    nx.draw(
        G, pos, axes, node_color=node_color_map
    )  # , node_color=node_color_map, edge_color=edge_color_map, **kwargs)
    axes.collections[0].set_edgecolor("#FF0000")

    plt.show()

    # plt.savefig(outfile)
    # print("Wrote out to {}".format(outfile))


if __name__ == "__main__":

    parser = get_ploi_argument_parser()
    parser.add_argument(
        "--hierarchical", action="store_true", help="Use hierarchical ploi instead."
    )
    parser.add_argument(
        "--idx", type=int, default=0, help="Index of the (train) problem to visualize."
    )
    args = parser.parse_args()

    # Seed RNG
    torch.manual_seed(args.seed)

    # Create dir to log files to
    args.expdir = os.path.join(args.logdir, args.expid)
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir, exist_ok=True)

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
    if not os.path.exists(args.datafile):
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
    if args.hierarchical:
        graphs_inp, graphs_tgt, graph_metadata = create_graph_dataset_hierarchical(
            training_data
        )
    else:
        graphs_inp, graphs_tgt, graph_metadata = create_graph_dataset(training_data)

    if args.hierarchical:
        # For the hierarchical dataset, the attributes are torch tensors. Cast them to numpy first
        graph_inp = graph_attrs_from_torch_to_numpy(graphs_inp[args.idx]["room_graph"])
        graph_tgt = graph_attrs_from_torch_to_numpy(graphs_tgt[args.idx]["room_graph"])
        visualize_graph(
            graph_inp, graph_tgt, domain_name=args.domain, problem_idx=args.idx
        )
    else:
        visualize_graph(
            graphs_inp[args.idx],
            graphs_tgt[args.idx],
            domain_name=args.domain,
            problem_idx=args.idx,
        )
