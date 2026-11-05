#!/bin/bash
#SBATCH --job-name=prep_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      
#SBATCH --mem=200G             # Massive RAM allocation to prevent OOM
#SBATCH --time=12:00:00        # Allow up to 12 hours
#SBATCH --output=data_%j.out
#SBATCH --error=data_%j.err

source /home/usuaris/imatge/pau.fernandez.cester/venv/myenv/bin/activate
python data_prep.py