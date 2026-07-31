#!/bin/bash
#SBATCH --job-name=kNN-eval-CLAP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=%j-%x.out


set -euo pipefail

echo "Activating virtual environment..."
source /home/milerisoy/miniconda3/bin/activate audio

which python
python --version

python -u eval_knn_clap.py