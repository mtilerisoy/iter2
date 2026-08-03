#!/bin/bash
#SBATCH --job-name=analyze-transfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rome
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=%j-%x.out

# Cross-task transfer map from the head sweep (pure re-analysis). CPU only, no GPU:
#
#   sbatch scripts/analyze-transfer.sh
#   sbatch scripts/analyze-transfer.sh --formats png pdf
#
# Arguments are forwarded to analyze_transfer.py.

set -euo pipefail

echo "Activating virtual environment..."
source /home/milerisoy/miniconda3/bin/activate audio

which python
python --version

python -u analyze_transfer.py "$@"
