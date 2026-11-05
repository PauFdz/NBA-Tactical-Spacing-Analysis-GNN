#!/bin/bash
#SBATCH --job-name=gnn_spatial_optimizer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=spatial_optimizer_%j.out
#SBATCH --error=spatial_optimizer_%j.err

source /home/usuaris/imatge/pau.fernandez.cester/venv/myenv/bin/activate
python team_spatial_optimizer.py