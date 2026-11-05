#!/bin/bash
#SBATCH --job-name=train_full_exp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # 8 CPUs to feed the GPU fast enough
#SBATCH --mem=250G                  # Massive RAM just to load the 20 million graphs
#SBATCH --gres=gpu:1           
#SBATCH --exclude=gpic09       
#SBATCH --time=24:00:00
#SBATCH --output=train_full_%j.out
#SBATCH --error=train_full_%j.err        

source /home/usuaris/imatge/pau.fernandez.cester/venv/myenv/bin/activate
python -u train_full_exp2.py