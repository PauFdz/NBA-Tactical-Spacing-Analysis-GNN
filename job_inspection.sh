#!/bin/bash
#SBATCH --job-name=prep_safe_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      
#SBATCH --mem=120G             # Massively increased RAM 
#SBATCH --time=06:00:00        
#SBATCH --output=data_%j.out
#SBATCH --error=data_%j.err

source /home/usuaris/imatge/pau.fernandez.cester/venv/myenv/bin/activate
python inspect_data.py