import copy
import os
import time
import warnings

import numpy as np
from torch._C import device
import pddlgym
import torch
import torch.nn as nn

from .planning import PlanningFailure, PlanningTimeout, validate_strips_plan


def train_model_graphnetwork(
    model,
    datasets,
    criterion,
    optimizer,
    use_gpu=False,
    print_every=10,
    save_every=100,
    save_folder="/tmp",
    epochs=1000,
    global_criterion=None,
    return_last_model_weights=True,
):
    since = time.time()
    best_seen_model_weights = None  # as measured over the validation set
    best_seen_running_validation_loss = np.inf

    trainset, validset = datasets["train"], datasets["val"]

    if use_gpu:
        device = "cuda:0"
    else:
        device = "cpu"

    for e in range(epochs):

        running_loss = 0.0
        running_num_samples = 0

        model.train()

        for idx in range(len(trainset)):
            g_inp = trainset[idx]["graph_input"]
            g_tgt = trainset[idx]["graph_target"]
            nfeat = torch.from_numpy(g_inp["nodes"]).float().to(device)
            efeat = torch.from_numpy(g_inp["edges"]).float().to(device)
            senders = torch.from_numpy(g_inp["senders"]).long().to(device)
            receivers = torch.from_numpy(g_inp["receivers"]).long().to(device)
            tgt = torch.from_numpy(g_tgt["nodes"]).float().to(device)
            edge_indices = torch.stack((senders, receivers))
            preds = model(nfeat, edge_indices, efeat)
            loss = criterion(preds, tgt)

            running_loss += loss.item()
            running_num_samples += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(
            f"== [EPOCH {e:03d} / {epochs}] Train loss: {(running_loss / running_num_samples):03.5f}"
        )

        if e % 100 == 0:

            model.eval()

            if e % save_every == 0:
                savefile = os.path.join(save_folder, f"model_{e:04d}.pt")
                torch.save(model.state_dict(), savefile)
                print(f"Saved model checkpoint {savefile}")

            running_loss = 0.0
            running_num_samples = 0

            for idx in range(len(validset)):
                g_inp = validset[idx]["graph_input"]
                g_tgt = validset[idx]["graph_target"]
                nfeat = torch.from_numpy(g_inp["nodes"]).float().to(device)
                efeat = torch.from_numpy(g_inp["edges"]).float().to(device)
                senders = torch.from_numpy(g_inp["senders"]).long().to(device)
                receivers = torch.from_numpy(g_inp["receivers"]).long().to(device)
                tgt = torch.from_numpy(g_tgt["nodes"]).float().to(device)
                edge_indices = torch.stack((senders, receivers))
                preds = model(nfeat, edge_indices, efeat)
                loss = criterion(preds, tgt)

                running_loss += loss.item()
                running_num_samples += 1

            print(
                f"===== [EPOCH {e:03d} / {epochs}] Val loss: {(running_loss / running_num_samples):03.5f}"
            )

            val_loss = running_loss / running_num_samples
            if val_loss < best_seen_running_validation_loss:
                best_seen_running_validation_loss = copy.deepcopy(val_loss)
                best_seen_model_weights = model.state_dict()
                savefile = os.path.join(save_folder, "best.pt")
                torch.save(best_seen_model_weights, savefile)
                print(
                    f"Found new best model with val loss {best_seen_running_validation_loss} at epoch {e}. Saved!"
                )

    time_elapsed = time.time() - since
    print(
        f"Training complete in {(time_elapsed // 60):.0f} m {(time_elapsed % 60):.0f} sec"
    )

    return best_seen_model_weights


def predict_graph_with_graphnetwork(model, input_graph):
    """Predict the target graph given the input graph"""
    model.eval()
    nfeat = torch.from_numpy(input_graph["nodes"]).float()
    efeat = torch.from_numpy(input_graph["edges"]).float()
    senders = torch.from_numpy(input_graph["senders"]).long()
    receivers = torch.from_numpy(input_graph["receivers"]).long()
    edge_indices = torch.stack((senders, receivers))
    scores = model(nfeat, edge_indices, efeat)
    scores = torch.sigmoid(scores)
    input_graph["nodes"] = scores.detach().cpu().numpy()
    return input_graph


def test_planner(
    planner, domain_name, num_problems, timeout, debug_mode=False, all_problems=False
):
    print("Running testing...")
    # In debug mode, use train problems for testing too (False by default)
    env = pddlgym.make("PDDLEnv{}Test-v0".format(domain_name))
    if debug_mode:
        warnings.warn(
            "WARNING: Running in debug mode (i.e., testing on train problems)"
        )
        env = pddlgym.make("PDDLEnv{}-v0".format(domain_name))
    num_problems = min(num_problems, len(env.problems))
    # If `all_problems` is set to True, override num_problems
    if all_problems:
        num_problems = len(env.problems)
    stats_to_log = ["num_node_expansions", "plan_length", "search_time", "total_time"]
    num_timeouts = 0
    num_failures = 0
    num_invalidated_plans = 0
    run_stats = []
    for problem_idx in range(num_problems):
        print(
            "\tTesting problem {} of {}".format(problem_idx + 1, num_problems),
            flush=True,
        )
        env.fix_problem_index(problem_idx)
        state, _ = env.reset()
        start = time.time()
        try:
            plan, planner_stats = planner(
                env.domain, state, timeout=timeout, domain_file_global=env._domain_file
            )
        except PlanningFailure as e:
            num_failures += 1
            print("\t\tPlanning failed with error: {}".format(e), flush=True)
            continue
        except PlanningTimeout as e:
            num_timeouts += 1
            print("\t\tPlanning failed with error: {}".format(e), flush=True)
            continue
        # Validate plan on the full test problem.
        if plan is None:
            num_failures += 1
            continue
        if not validate_strips_plan(
            domain_file=env.domain.domain_fname,
            problem_file=env.problems[problem_idx].problem_fname,
            plan=plan,
        ):
            print("\t\tPlanning returned an invalid plan")
            num_invalidated_plans += 1
            continue
        wall_time = time.time() - start
        print(
            "\t\tSuccess, got plan of length {} in {:.5f} seconds".format(
                len(plan), wall_time
            ),
            flush=True,
        )
        planner_stats["wall_time"] = wall_time
        run_stats.append(planner_stats)

    global_stats = dict()
    stats_to_track = {
        "num_node_expansions",
        "plan_length",
        "search_time",
        "total_time",
        "objects_used",
        "objects_total",
        "neural_net_time",
        "wall_time",
    }
    num_stats = len(run_stats)
    for stat in stats_to_track:
        if stat not in global_stats:
            global_stats[stat] = np.zeros(num_stats)
        for i, run in enumerate(run_stats):
            global_stats[stat][i] = run[stat]
    for stat in stats_to_track:
        stat_mean = float(global_stats[stat].mean().item())
        stat_std = float(global_stats[stat].std().item())
        global_stats[stat] = stat_mean
        global_stats[f"{stat}_std"] = stat_std
    global_stats["success_rate"] = float(num_stats / num_problems)
    global_stats["timeout_rate"] = float(num_timeouts / num_problems)
    global_stats["failure_rate"] = float(num_failures / num_problems)
    global_stats["invalid_rate"] = float(num_invalidated_plans / num_problems)

    global_stats["num_timeouts"] = num_timeouts
    global_stats["num_failures"] = num_failures
    global_stats["num_invalidated_plans"] = num_invalidated_plans
    global_stats["num_timeouts"] = num_timeouts
    return run_stats, global_stats


def train_model_hierarchical(
    model,
    datasets,
    criterion,
    optimizer,
    use_gpu=False,
    print_every=10,
    save_every=100,
    save_folder="/tmp",
    epochs=1000,
    global_criterion=None,
    return_last_model_weights=True,
    model_type="room",
    eval_every=100,
):
    if model_type not in ["room", "object"]:
        raise ValueError(
            f"Unknown model type {model_type}. Valid model types are 'room', 'object'."
        )

    if use_gpu:
        device = "cuda:0"
    else:
        device = "cpu"

    def unpack_item(item):
        if model_type == "object":
            _input_graph = item["graph_input"]
            _target_graph = item["graph_target"]
            _nfeat = _input_graph["nodes"].float().to(device)
            _efeat = _input_graph["edges"].float().to(device)
            _senders = _input_graph["senders"].long().to(device)
            _receivers = _input_graph["receivers"].long().to(device)
            _tgt = _target_graph["nodes"].float().to(device)
            _edge_indices = torch.stack((_senders, _receivers))
            return _nfeat, _edge_indices, _efeat, _tgt
        elif model_type == "room":
            _input_graph = item["graph_input"]
            _target_graph = item["graph_target"]
            _nfeat = _input_graph["room_graph"]["nodes"].float().to(device)
            _efeat = _input_graph["room_graph"]["edges"].float().to(device)
            _senders = _input_graph["room_graph"]["senders"].long().to(device)
            _receivers = _input_graph["room_graph"]["receivers"].long().to(device)
            _tgt = _target_graph["room_graph"]["nodes"].float().to(device)
            _edge_indices = torch.stack((_senders, _receivers))
            return _nfeat, _edge_indices, _efeat, _tgt

    since = time.time()
    best_seen_model_weights = None  # as measured over the validation set
    best_seen_running_validation_loss = np.inf

    trainset, validset = datasets["train"], datasets["val"]

    for e in range(epochs):

        running_loss = 0.0
        running_num_samples = 0

        model.train()

        permuted_indx = torch.randperm(len(trainset))

        for idx in range(len(trainset)):
            i = permuted_indx[idx]
            nfeat, edge_indices, efeat, tgt = unpack_item(trainset[i])
            preds = model(nfeat, edge_indices, efeat)
            loss = criterion(preds, tgt)

            running_loss += loss.item()
            running_num_samples += 1

            loss.backward()

            if idx % 20 == 0 or idx == len(trainset) - 1:
                optimizer.step()
                optimizer.zero_grad()

        print(
            f"== [Model: {model_type}] [EPOCH {e:03d} / {epochs}] Train loss: {(running_loss / running_num_samples):03.5f}"
        )

        if e % eval_every == 0:

            model.eval()

            if e % save_every == 0:
                savefile = os.path.join(save_folder, f"{model_type}_model_{e:04d}.pt")
                torch.save(model.state_dict(), savefile)
                print(f"Saved model checkpoint {savefile}")

            running_loss = 0.0
            running_num_samples = 0

            for idx in range(len(validset)):
                nfeat, edge_indices, efeat, tgt = unpack_item(validset[idx])
                preds = model(nfeat, edge_indices, efeat)
                loss = criterion(preds, tgt)

                running_loss += loss.item()
                running_num_samples += 1

            print(
                f"===== [Model: {model_type}] [EPOCH {e:03d} / {epochs}] Val loss: {(running_loss / running_num_samples):03.5f}"
            )

            val_loss = running_loss / running_num_samples
            if val_loss < best_seen_running_validation_loss:
                best_seen_running_validation_loss = copy.deepcopy(val_loss)
                best_seen_model_weights = model.state_dict()
                savefile = os.path.join(save_folder, f"{model_type}_best.pt")
                torch.save(best_seen_model_weights, savefile)
                print(
                    f"Found new best model with val loss {best_seen_running_validation_loss} at epoch {e}. Saved!"
                )

    time_elapsed = time.time() - since
    print(
        f"Training complete in {(time_elapsed // 60):.0f} m {(time_elapsed % 60):.0f} sec"
    )

    return best_seen_model_weights


def predict_graph_with_graphnetwork_hierarchical(room_model, object_model, input_graph):
    """Predict scores across both levels of the hierarchy. """
    room_model.eval()
    object_model.eval()

    # Get room scores
    nfeat = torch.from_numpy(input_graph["room_graph"]["nodes"]).float()
    efeat = torch.from_numpy(input_graph["room_graph"]["edges"]).float()
    senders = torch.from_numpy(input_graph["room_graph"]["senders"]).long()
    receivers = torch.from_numpy(input_graph["room_graph"]["receivers"]).long()
    edge_indices = torch.stack((senders, receivers))
    room_scores = room_model(nfeat, edge_indices, efeat)
    room_scores = torch.sigmoid(room_scores)

    nfeat = torch.from_numpy(input_graph["nodes"]).float()
    efeat = torch.from_numpy(input_graph["edges"]).float()
    senders = torch.from_numpy(input_graph["senders"]).long()
    receivers = torch.from_numpy(input_graph["receivers"]).long()
    edge_indices = torch.stack((senders, receivers))
    object_scores = object_model(nfeat, edge_indices, efeat)
    object_scores = torch.sigmoid(object_scores)

    return room_scores, object_scores
