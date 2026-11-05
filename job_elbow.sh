#!/bin/bash
#SBATCH --job-name=find_optimal_k
#SBATCH --output=elbow_%j.out
#SBATCH --error=elbow_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00


source /home/usuaris/imatge/pau.fernandez.cester/venv/myenv/bin/activate
python find_optimal_k.py