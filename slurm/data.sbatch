#!/bin/bash
# Steps 1-3a: features from the raw book (one process per coin), the train/val/test tables,
# and the linear models (a few minutes).
#   sbatch slurm/data.sbatch
#SBATCH --job-name=fin516-data
#SBATCH --partition=class
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/home/an8303/fin516/logs/%x-%j.out

set -e
cd /home/an8303/fin516/src
PY=/scratch/network/an8303/fin516/venv/bin/python

$PY features.py
$PY dataset.py
$PY linear.py
