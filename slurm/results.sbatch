#!/bin/bash
# Steps 4-5: score the forecasts and run the trading simulation.
#   sbatch slurm/results.sbatch
#SBATCH --job-name=fin516-results
#SBATCH --partition=class
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/an8303/fin516/logs/%x-%j.out

set -e
cd /home/an8303/fin516/src
PY=/scratch/network/an8303/fin516/venv/bin/python

$PY evaluate.py
$PY trading.py
