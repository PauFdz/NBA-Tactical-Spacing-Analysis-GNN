#!/bin/bash
#SBATCH --job-name=inspect_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=150G              # Massive RAM needed just to load the .pt file!
#SBATCH --time=01:00:00         # Should only take a few minutes
#SBATCH --output=inspect_%j.out
#SBATCH --error=inspect_%j.err

source /home/usuaris/imatge/pau.fernandez.cester/venv/myenv/bin/activate

# The -u flag forces Python to print instantly, bypassing the buffer delay!
python -u inspect_structure.py