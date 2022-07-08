#!/bin/bash

declare -a methods=("scenegraph" "hierarchical" "ploi")
# declare -a modes=("train" "test")
# declare -a tiny_domains=("v2tiny1" "v2tiny2" "v2tiny10" "v3tiny10bagslots3" "v3tiny10bagslots5" "v3tiny10bagslots7" "v3tiny10bagslots10" "v4tiny5" "v5tiny5bagslots7")
# declare -a medium_domains=("v2medium1" "v2medium2" "v2medium10" "v3medium10bagslots3" "v3medium10bagslots5" "v3medium10bagslots7" "v3medium10bagslots10")

declare -a official_grounded_optimal_domains=("v2tiny1" "v2tiny2" "v2medium1" "v2medium2")
declare -a official_grounded_domains=("v2tiny10" "v2medium10" "v3tiny10bagslots10" "v3medium10bagslots10")
declare -a official_lifted_domains=("v4tiny5" "v4medium5" "v5tiny5bagslots5" "v5medium5bagslots5")
declare -a ablation_domains=("v3tiny10bagslots3" "v3tiny10bagslots5" "v3tiny10bagslots7" "v3medium10bagslots3" "v3medium10bagslots5" "v3medium10bagslots7")

declare -a modes=("train" "test")
declare -a methods=("ploi" "hierarchical" "scenegraph")
# declare -a tiny_domains=("v2tiny10" "v3tiny10bagslots3" "v3tiny10bagslots5" "v3tiny10bagslots7" "v3tiny10bagslots10")
# declare -a tiny_domains=("v4tiny5" "v5tiny5bagslots7")
# declare -a medium_domains=("v4medium5" "v5medium5bagslots7")


# for i in "${official_grounded_optimal_domains[@]}"
# do
#     for j in "${methods[@]}"
#     do
#         for k in "${modes[@]}"
#         do
#             echo "# $i"
#             cmd="python main.py --domain taskography$i --method $j --num-train-problems 40 --epochs 401 --mode $k  --timeout 30 --expid taskography${i}_${j} --logdir cache/results-official --all-problems"
#             echo "$cmd"
#         done
#     done
# done

# for i in "${official_grounded_domains[@]}"
# do
#     for j in "${methods[@]}"
#     do
#         for k in "${modes[@]}"
#         do
#             echo "# $i"
#             cmd="python main.py --domain taskography$i --method $j --num-train-problems 40 --epochs 401 --mode $k  --timeout 30 --expid taskography${i}_${j} --logdir cache/results-official --all-problems"
#             echo "$cmd"
#         done
#     done
# done

# for i in "${official_lifted_domains[@]}"
# do
#     for j in "${methods[@]}"
#     do
#         for k in "${modes[@]}"
#         do
#             echo "# $i"
#             cmd="python main.py --domain taskography$i --method $j --num-train-problems 40 --epochs 401 --mode $k  --timeout 30 --expid taskography${i}_${j} --logdir cache/results-official --all-problems"
#             echo "$cmd"
#         done
#     done
# done

# for i in "${ablation_domains[@]}"
# do
#     for j in "${methods[@]}"
#     do
#         for k in "${modes[@]}"
#         do
#             echo "# $i"
#             cmd="python main.py --domain taskography$i --method $j --num-train-problems 40 --epochs 401 --mode $k  --timeout 30 --expid taskography${i}_${j} --logdir cache/results-official --all-problems"
#             echo "$cmd"
#         done
#     done
# done







# The following code ended up not being used in the official results

# for i in "${medium_domains[@]}"
# do
#     for j in "${methods[@]}"
#     do
#         for k in "${modes[@]}"
#         do
#             echo "$i"
#             cmd="python main.py --domain $i --method $j --num-train-problems 40 --num-test-problems 172 --epochs 401 --mode $k  --timeout 30 --expid main-$i"
#             eval $cmd
#         done
#     done
# done
