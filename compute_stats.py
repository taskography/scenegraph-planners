import argparse
import json
import math
import os
import shutil
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="Input directory containing py files with planner stats",
    )
    args = parser.parse_args()

    statsfiles = []
    if os.path.exists(os.path.join(args.i, "scenegraph_test_stats.py")):
        statsfiles.append(os.path.join(args.i, "scenegraph_test_stats.py"))
    if os.path.exists(os.path.join(args.i, "hierarchical_test_stats.py")):
        statsfiles.append(os.path.join(args.i, "hierarchical_test_stats.py"))
    if os.path.exists(os.path.join(args.i, "ploi_test_stats.py")):
        statsfiles.append(os.path.join(args.i, "ploi_test_stats.py"))

    for statsfile in statsfiles:
        print(f"======== {statsfile} ===========")
        stats = None

        sys.path.append(args.i)
        if statsfile == os.path.join(args.i, "scenegraph_test_stats.py"):
            import scenegraph_test_stats

            stats = scenegraph_test_stats.STATS
        elif statsfile == os.path.join(args.i, "hierarchical_test_stats.py"):
            # if os.path.exists(os.path.join(args.i, "hierarchical_test_stats.py")):
            import hierarchical_test_stats

            stats = hierarchical_test_stats.STATS
        # elif os.path.exits(os.path.join(args.i, "ploi_test_stats.py")):
        elif statsfile == os.path.join(args.i, "ploi_test_stats.py"):
            import ploi_test_stats

            stats = ploi_test_stats.STATS

        num_stats = len(stats)
        avg_total_time = 0.0
        avg_plan_length = 0.0
        avg_objects_used = 0.0
        avg_objects_total = 0.0
        avg_fraction_of_objects_used = 0.0
        avg_neural_net_time = 0.0
        total_num_replanning_steps = 0.0
        for i in range(num_stats):
            avg_total_time += stats[i]["total_time"]
            avg_plan_length += stats[i]["plan_length"]
            avg_objects_used += stats[i]["objects_used"]
            avg_objects_total += stats[i]["objects_total"]
            fraction_of_objects_used = float(avg_objects_used) / float(
                avg_objects_total
            )
            avg_fraction_of_objects_used += fraction_of_objects_used
            if "neural_net_time" in stats[i].keys():
                avg_neural_net_time += stats[i]["neural_net_time"]
            if "num_replanning_steps" in stats[i].keys():
                total_num_replanning_steps += stats[i]["num_replanning_steps"]
        avg_total_time /= num_stats
        avg_plan_length /= num_stats
        avg_objects_used /= num_stats
        avg_objects_total /= num_stats
        avg_fraction_of_objects_used /= num_stats
        avg_neural_net_time /= num_stats
        print("Avg. total time:", avg_total_time)
        print("Avg. plan length:", avg_plan_length)
        print("Avg. objects used:", avg_objects_used)
        print("Avg. objects total:", avg_objects_total)
        print("Avg. fraction of objects used:", avg_fraction_of_objects_used)
        print("Avg. neural net time:", avg_neural_net_time)
        print("Total number of replanning steps:", total_num_replanning_steps)
