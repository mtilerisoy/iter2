#!/bin/bash
#SBATCH --job-name=analyze-headprops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rome
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=%j-%x.out

# Renders the head-property analysis into figures/head_properties_*/. CPU only, no GPU:
#
#   sbatch scripts/analyze-headprops.sh
#   sbatch scripts/analyze-headprops.sh --formats png pdf
#
# Arguments are forwarded to analyze_head_properties.py.

set -euo pipefail

echo "Activating virtual environment..."
source /home/milerisoy/miniconda3/bin/activate audio

which python
python --version

python -u analyze_head_properties.py "$@"
