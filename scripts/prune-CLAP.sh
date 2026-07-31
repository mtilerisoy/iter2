#!/bin/bash
#SBATCH --job-name=prune-eval-CLAP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=%j-%x.out

# Structured-pruning sweep of the CLAP audio tower.
#
#   sbatch scripts/prune-CLAP.sh --prune_type none          # intact reference, ~10 min
#   sbatch scripts/prune-CLAP.sh --prune_type block         # 12 configurations
#   sbatch scripts/prune-CLAP.sh --prune_type head          # 184 configurations
#   sbatch scripts/prune-CLAP.sh --prune_type head --pooling pooled --datasets KAUH
#
# Every argument is forwarded to eval_prune_clap.py. The head sweep is the long one:
# each dataset is decoded and feature-extracted once (that cache is what the 64 GB is
# for), after which the 184 configurations are pure GPU forward passes. Re-submit the
# same command with --resume if the walltime runs out; finished (dataset, index) rows
# are read back from the CSV and skipped.

set -euo pipefail

echo "Activating virtual environment..."
source /home/milerisoy/miniconda3/bin/activate audio

which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u eval_prune_clap.py "$@"
