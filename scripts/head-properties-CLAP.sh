#!/bin/bash
#SBATCH --job-name=head-props-CLAP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=%j-%x.out

# Characterises all 184 CLAP audio heads: static weight properties + activation energy on
# each medical corpus and on general audio (FSD50k).
#
#   sbatch scripts/head-properties-CLAP.sh
#   sbatch scripts/head-properties-CLAP.sh --general-limit 2000
#
# Every head is instrumented at once, so this is ONE forward pass per corpus (8 passes
# total), not one per head. Arguments are forwarded to eval_head_properties.py.

set -euo pipefail

echo "Activating virtual environment..."
source /home/milerisoy/miniconda3/bin/activate audio

which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u eval_head_properties.py "$@"
