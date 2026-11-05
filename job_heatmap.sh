#!/bin/bash
#SBATCH --job-name=gnn_heatmap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=heatmap_%j.out
#SBATCH --error=heatmap_%j.err

source /home/usuaris/imatge/pau.fernandez.cester/venv/myenv/bin/activate
python heatmap_analysis.py