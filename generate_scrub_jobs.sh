declare -a domains=("v2medium10scrub" "v3medium10bagslots10scrub" "v4medium5" "v5medium5bagslots5")
declare -a methods=("ploi")
declare -a modes=("train" "test")
declare -a timeout=30

for i in "${domains[@]}"
do
    for j in "${methods[@]}"
    do
        for k in "${modes[@]}"
        do
            echo "echo \"# Planning in domain $i with planner $j\""
            cmd="python main.py --domain taskography$i --method $j --num-train-problems 40 --epochs 401 --mode $k  --timeout 30 --expid taskography${i}_${j} --logdir cache/results-official --all-problems"
            echo "$cmd"
        done
    done
done
