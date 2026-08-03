#!/bin/bash
#SBATCH --job-name=analyze-famtransfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rome
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=%j-%x.out

# Renders every results/prune_clap_*.csv into figures/<sweep>/. CPU only, no GPU:
#
#   sbatch scripts/analyze-famtransfer.sh
#   sbatch scripts/analyze-famtransfer.sh --formats png pdf
#
# Arguments are forwarded to analyze_family_transfer.py.

set -euo pipefail

echo "Activating virtual environment..."
source /home/milerisoy/miniconda3/bin/activate audio

which python
python --version

python -u analyze_family_transfer.py "$@"
