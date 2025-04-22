#!/bin/bash
#SBATCH --job-name=Test_gan_training
#SBATCH --output=test_gan_%j.txt
#SBATCH --time=08:00:00
#SBATCH --nodes=1                  # Use 1 node for now (scale up later if needed)
#SBATCH --ntasks=1                 # 1 task total
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32GB
#SBATCH --partition=standard      # Or whatever default CPU queue your HPC uses
#SBATCH --mail-user=peter.t.smith@northumbria.ac.uk
#SBATCH --mail-type=ALL

# Remove CUDA (not needed for CPU-only PyTorch)
module purge

# Activate Conda
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate main

# Info
echo "Using Python: $(which python)"
echo "Using Conda Env: $(conda info --envs)"

# Run your script
python /home/osw_w16018262/PhD-experiments/testgan.py
