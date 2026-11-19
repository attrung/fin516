#!/bin/bash
# Step 3: the models. Adroit runs at most 2 of our jobs at once, so each job runs several
# models side by side. Group A: LightGBM + MLP (6 runs). Group B: GRU (3 runs, the slowest).
#   sbatch --export=GROUP=A slurm/models.sbatch
#   sbatch --export=GROUP=B slurm/models.sbatch
#SBATCH --job-name=fin516-models
#SBATCH --partition=class
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/home/an8303/fin516/logs/%x-%j.out

cd /home/an8303/fin516/src
PY=/scratch/network/an8303/fin516/venv/bin/python

if [ "$GROUP" = "A" ]; then
    RUNS=("trees.py 5" "trees.py 30" "trees.py 60" "nn.py mlp 5" "nn.py mlp 30" "nn.py mlp 60"); CPUS=5
else
    RUNS=("nn.py gru 5" "nn.py gru 30" "nn.py gru 60"); CPUS=10
fi
for run in "${RUNS[@]}"; do
    SLURM_CPUS_PER_TASK=$CPUS $PY $run > ../logs/${run// /_}.log 2>&1 &
done
wait
grep -l Traceback ../logs/*.py_*.log && exit 1 || true
