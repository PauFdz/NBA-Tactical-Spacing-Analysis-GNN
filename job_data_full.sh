#!/bin/bash
#SBATCH --job-name=prep_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # Doubled to 16 cores to parallelize the parsing
#SBATCH --mem=300G                 # Extreme RAM allocation for 600 games
#SBATCH --time=24:00:00            # Allow up to 24 hours just for data prep
#SBATCH --output=data_full_%j.out
#SBATCH --error=data_full_%j.err

source /home/usuaris/imatge/pau.fernandez.cester/venv/myenv/bin/activate
python -u data_prep_full.py