echo "# Training in domain v2medium2"
python main_seek.py --domain taskographyv2medium2 --num-train-problems 40 --epochs 101 --mode train  --timeout 30 --expid taskographyv2medium2 --logdir cache/results-official-seek --all-problems
echo "# Planning in domain v2medium2 with planner random (NO SEEK)"
python main_seek.py --mode test --domain taskographyv2medium2 --scoring-mode random --timeout 30 --expid taskographyv2medium2 --logdir cache/results-official-seek --all-problems
echo "# Planning in domain v2medium2 with planner random (WITH SEEK)"
python main_seek.py --mode test --domain taskographyv2medium2 --scoring-mode random --timeout 30 --expid taskographyv2medium2 --logdir cache/results-official-seek --seek --all-problems
echo "# Planning in domain v2medium2 with planner ploi (NO SEEK)"
python main_seek.py --mode test --domain taskographyv2medium2 --scoring-mode ploi --timeout 30 --expid taskographyv2medium2 --logdir cache/results-official-seek --all-problems
echo "# Planning in domain v2medium2 with planner ploi (WITH SEEK)"
python main_seek.py --mode test --domain taskographyv2medium2 --scoring-mode ploi --timeout 30 --expid taskographyv2medium2 --logdir cache/results-official-seek --seek --all-problems
echo "# Planning in domain v2medium2 with planner hierarchical (NO SEEK)"
python main_seek.py --mode test --domain taskographyv2medium2 --scoring-mode hierarchical --timeout 30 --expid taskographyv2medium2 --logdir cache/results-official-seek --all-problems
echo "# Planning in domain v2medium2 with planner hierarchical (WITH SEEK)"
python main_seek.py --mode test --domain taskographyv2medium2 --scoring-mode hierarchical --timeout 30 --expid taskographyv2medium2 --logdir cache/results-official-seek --seek --all-problems
