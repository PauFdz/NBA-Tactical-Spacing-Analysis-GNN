#!/bin/bash
#SBATCH --job-name=tactical_gnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=combined_%j.out
#SBATCH --error=combined_%j.err

source /home/usuaris/imatge/pau.fernandez.cester/venv/myenv/bin/activate
python combined_tactical_analysis.py