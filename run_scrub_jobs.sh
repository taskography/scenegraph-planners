echo "# Planning in domain v2medium10scrub with planner ploi"
python main.py --domain taskographyv2medium10scrub --method ploi --num-train-problems 40 --epochs 401 --mode train  --timeout 30 --expid taskographyv2medium10scrub_ploi --logdir cache/results-official --all-problems
echo "# Planning in domain v2medium10scrub with planner ploi"
python main.py --domain taskographyv2medium10scrub --method ploi --num-train-problems 40 --epochs 401 --mode test  --timeout 30 --expid taskographyv2medium10scrub_ploi --logdir cache/results-official --all-problems
echo "# Planning in domain v3medium10bagslots10scrub with planner ploi"
python main.py --domain taskographyv3medium10bagslots10scrub --method ploi --num-train-problems 40 --epochs 401 --mode train  --timeout 30 --expid taskographyv3medium10bagslots10scrub_ploi --logdir cache/results-official --all-problems
echo "# Planning in domain v3medium10bagslots10scrub with planner ploi"
python main.py --domain taskographyv3medium10bagslots10scrub --method ploi --num-train-problems 40 --epochs 401 --mode test  --timeout 30 --expid taskographyv3medium10bagslots10scrub_ploi --logdir cache/results-official --all-problems
echo "# Planning in domain v4medium5 with planner ploi"
python main.py --domain taskographyv4medium5 --method ploi --num-train-problems 40 --epochs 401 --mode train  --timeout 30 --expid taskographyv4medium5_ploi --logdir cache/results-official --all-problems
echo "# Planning in domain v4medium5 with planner ploi"
python main.py --domain taskographyv4medium5 --method ploi --num-train-problems 40 --epochs 401 --mode test  --timeout 30 --expid taskographyv4medium5_ploi --logdir cache/results-official --all-problems
echo "# Planning in domain v5medium5bagslots5 with planner ploi"
python main.py --domain taskographyv5medium5bagslots5 --method ploi --num-train-problems 40 --epochs 401 --mode train  --timeout 30 --expid taskographyv5medium5bagslots5_ploi --logdir cache/results-official --all-problems
echo "# Planning in domain v5medium5bagslots5 with planner ploi"
python main.py --domain taskographyv5medium5bagslots5 --method ploi --num-train-problems 40 --epochs 401 --mode test  --timeout 30 --expid taskographyv5medium5bagslots5_ploi --logdir cache/results-official --all-problems
