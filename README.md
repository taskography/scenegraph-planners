# Modern scene-graph task planners

This repo houses code for the modern planners used in Taskography, in particular the PLOI, SCRUB, and SEEK planners.

This codebase originally was built out of the [code release for PLOI](https://github.com/tomsilver/ploi) by Tom Silver, Rohan Chitnis, and Aidan Curtis. If you find this codebase useful, we urge you to strongly consider citing PLOI, in addition to Taskography.

**Note**: We reimplement the graph network in PLOI using pytorch-geometric, which improves speed while retaining performance.

## Installation (requirements)

We recommend running this in a `conda` or `virtualenv` environment. Requirements include
```sh
torch>1.6.0
pandas
networkx
matplotlib
```

For use with taskography, we require our fork of [pddlgym](https://github.com/taskography/pddlgym), which houses our custom domains and problems.

Another essential requirement is `torch_geometric`, which is best installed by following [these instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

Download and build the plan validation tool available at https://github.com/KCL-Planning/VAL, then make a symlink called validate on your path that points to the build/Validate binary, e.g. `ln -s <path to VAL>/build/Validate /usr/local/bin/validate`. If done successfully, running validate on your command line should give an output that starts with the line: `VAL: The PDDL+ plan validation tool`.


## Running PLOI/SCRUB/SEEK on a registered pddlgym environment

To train a planner on an environment already registered with `pddlgym`, simply run `main.py` passing the appropriate commandline arguments.

For example, PLOI may be run on a domain `Taskographyv2tiny10` by executing the following command. It trains on 40 problem instances for 401 epochs, and tests on all validation problem instances.
```
python main.py --domain taskographyv2tiny10 --method ploi --num-train-problems 40 --epochs 401 --mode train  --timeout 30 --expid taskographyv2tiny10_ploi --logdir cache/results --all-problems
```

To run evaluation using a pretrained model a PLOI baseline on the domain, set the `--mode` argument to `test` instead. The code will then pick up the best model from the directory pointed to by `--expid`.

Here's the list of supported commandline arguments across all planners.
```sh
usage: main.py [-h] [--seed SEED] [--method {scenegraph,hierarchical,ploi}]
               [--mode {train,test,visualize}] [--domain DOMAIN]
               [--train-planner-name {fd-lama-first,fd-opt-lmcut}]
               [--eval-planner-name {fd-lama-first,fd-opt-lmcut}]
               [--num-train-problems NUM_TRAIN_PROBLEMS]
               [--num-test-problems NUM_TEST_PROBLEMS]
               [--do-incremental-planning] [--timeout TIMEOUT] [--expid EXPID]
               [--logdir LOGDIR] [--device {cpu,cuda:0}] [--criterion {bce}]
               [--pos-weight POS_WEIGHT] [--epochs EPOCHS] [--lr LR]
               [--load-model] [--print-every PRINT_EVERY] [--gamma GAMMA]
               [--force-collect-data]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Random seed
  --method {scenegraph,hierarchical,ploi}
  --mode {train,test,visualize}
                        Mode to run the script in
  --domain DOMAIN       Name of the pddlgym domain to use.
  --train-planner-name {fd-lama-first,fd-opt-lmcut}
                        Train planner to use
  --eval-planner-name {fd-lama-first,fd-opt-lmcut}
                        Eval planner to use
  --num-train-problems NUM_TRAIN_PROBLEMS
                        Number of train problems
  --num-test-problems NUM_TEST_PROBLEMS
                        Number of test problems
  --do-incremental-planning
                        Whether or not to do incremental planning
  --timeout TIMEOUT     Timeout for test-time planner
  --expid EXPID         Unique exp id to log data to
  --logdir LOGDIR       Directory to store all expt logs in
  --device {cpu,cuda:0}
                        torch.device argument
  --criterion {bce}     Loss function to use
  --pos-weight POS_WEIGHT
                        Weight for the positive class in binary cross-entropy
                        computation
  --epochs EPOCHS       Number of epochs to run training for
  --lr LR               Learning rate
  --load-model          Path to load model from
  --print-every PRINT_EVERY
                        Number of iterations after which to print training
                        progress.
  --gamma GAMMA         Value of importance threshold (gamma) for PLOI.
  --force-collect-data  Force data collection (ignore pre-cached datasets).
```

## Reproducing experiments from our CoRL 2021 paper

We also provide scripts that generate configs to reproduce experiments from our CoRL 2021 submission.

* `runexp.sh` is the primary file that is capable of generating all other shell scripts to run various planners on all of the benchmark domains. The script is simple to follow, and is currently heavily commented out to generate only a specific subset of scenarios. Users are welcome to uncomment these lines and accordingly generate other valid scenarios (by uncommenting the code block of interest, setting parameters as desired, and running `./runexp.sh > output_file.sh`). If one wishes to directly use the generated shell scripts to launch experiments used in our paper, please read on.
* `run_official_grounded_domains.sh` runs ploi, and two variants---scenegraph, hierarchical, on all of our grounded domains.
* `run_official_lifted_domains.sh` runs the above variants on all of our lifted domains.
* `run_scrub_jobs.sh` runs the above planner variants on the SCRUBbed versions of all domains (both grounded and lifted). SCRUBbed versions of these domains are generated by following `generate_scrubed_problems.py` and `generate_scrub_jobs.sh`.
* `run_seek_jobs.sh` runs SEEK on all domains (both grounded and lifted). These jobs are generated by running `generate_seek_jobs.sh`. (SEEK can also be run on SCRUBbed versions of both grounded and lifted domains).
* `run_ablation_domains.sh` runs the planners on domains used in our ablation studies.


## Citing

If you use this repo in your work, we ask that you cite both the Taskography paper and the PLOI paper.

```
@inproceedings{agia2022taskography,
  title={Taskography: Evaluating robot task planning over large 3D scene graphs},
  author={Agia, Christopher and Jatavallabhula, {Krishna Murthy} and Khodeir, Mohamed and Miksik, Ondrej and Vineet, Vibhav and Mukadam, Mustafa and Paull, Liam and Shkurti, Florian},
  booktitle={Conference on Robot Learning},
  pages={46--58},
  year={2022},
  organization={PMLR}
}
```

```
@article{ploi,
  title={Planning with learned object importance in large problem instances using graph neural networks},
  author={Silver, Tom and Chitnis, Rohan and Curtis, Aidan and Tenenbaum, Joshua and Lozano-Perez, Tomas and Kaelbling, {Leslie Pack}},
  journal=aaai,
  year={2020}
}
```
