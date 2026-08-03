#!/bin/bash
#SBATCH --job-name=family-transfer-CLAP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%j-%x.out

# Held-out family-transfer test: select head sets on other datasets, remove them jointly,
# measure on a dataset never used for the selection.
#
#   sbatch scripts/family-transfer-CLAP.sh
#   sbatch scripts/family-transfer-CLAP.sh --k 5
#   sbatch scripts/family-transfer-CLAP.sh --resume
#
# 7 targets x (1 intact + 3 selected sets + 10 random) = 98 k-NN evaluations. Features are
# cached per dataset, so this is dominated by audio decoding, not the GPU.

set -euo pipefail

echo "Activating virtual environment..."
source /home/milerisoy/miniconda3/bin/activate audio

which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u eval_family_transfer_clap.py "$@"
