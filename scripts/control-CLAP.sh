#!/bin/bash
#SBATCH --job-name=control-eval-CLAP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%j-%x.out

# Shuffle / noise control for the head-pruning gains.
#
#   sbatch scripts/control-CLAP.sh                      # every dataset, 3+3 heads, 10 trials
#   sbatch scripts/control-CLAP.sh --datasets KAUH
#   sbatch scripts/control-CLAP.sh --trials 20
#   sbatch scripts/control-CLAP.sh --resume             # continue a timed-out job
#
# Arguments are forwarded to eval_control_clap.py. With the defaults this is
# 7 datasets x (6 heads x 21 runs + 1 intact) = 889 k-NN evaluations, ~30 min on one
# A100. Features are cached per dataset exactly as in the sweep, and rows are flushed as
# they land, so --resume picks up wherever the walltime cut it off.
#
# Requires the sweep it controls: results/prune_clap_head_<pooling>_seed<seed>.csv and
# the matching --prune_type none reference.

set -euo pipefail

echo "Activating virtual environment..."
source /home/milerisoy/miniconda3/bin/activate audio

which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u eval_control_clap.py "$@"
