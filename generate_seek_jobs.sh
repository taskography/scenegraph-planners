declare -a domains=("v2medium2") # "v3tiny10bagslots3" "v4tiny5" "v5tiny5bagslots5")
declare -a methods=("random" "ploi" "hierarchical")
declare -a timeout=30

for i in "${domains[@]}"
do
    echo "echo \"# Training in domain $i\""
    cmd="python main_seek.py --domain taskography$i --num-train-problems 40 --epochs 101 --mode train  --timeout 30 --expid taskography${i} --logdir cache/results-official-seek --all-problems"
    echo "$cmd"
    for j in "${methods[@]}"
    do
        echo "echo \"# Planning in domain $i with planner $j (NO SEEK)\""
        cmd="python main_seek.py --mode test --domain taskography$i --scoring-mode $j --timeout 30 --expid taskography${i} --logdir cache/results-official-seek --all-problems"
        echo "$cmd"
        echo "echo \"# Planning in domain $i with planner $j (WITH SEEK)\""
        cmd="python main_seek.py --mode test --domain taskography$i --scoring-mode $j --timeout 30 --expid taskography${i} --logdir cache/results-official-seek --seek --all-problems"
        echo "$cmd"
    done
done
