#!/bin/bash
#SBATCH --job-name=data_analysis
#SBATCH --output=data_analysis_%j.txt
#SBATCH --time=08:00:00
#SBATCH --nodes=1                  # Use 1 node for now (scale up later if needed)
#SBATCH --ntasks=1                 # 1 task total
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32GB
#SBATCH --partition=72hour      # Or whatever default CPU queue your HPC uses
#SBATCH --mail-user=peter.t.smith@northumbria.ac.uk
#SBATCH --mail-type=ALL

# Remove CUDA (not needed for CPU-only PyTorch)
module purge

# Activate Conda
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Run your script
python main.py
