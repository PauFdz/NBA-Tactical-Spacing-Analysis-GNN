#!/bin/bash
#SBATCH --job-name=exp3             
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --gres=gpu:1           
#SBATCH --exclude=gpic09       
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err        

source /home/usuaris/imatge/pau.fernandez.cester/venv/myenv/bin/activate
python train_exp3.py