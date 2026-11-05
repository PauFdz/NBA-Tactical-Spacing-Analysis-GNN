#!/bin/bash
#SBATCH --job-name=nba_clustering
#SBATCH --output=clustering_%j.out
#SBATCH --error=clustering_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# --- LOAD MODULES AND ENVIRONMENT ---
# module load anaconda
# conda activate your_env_name

echo "=================================================="
echo "Starting Latent Tactical Clustering Analysis..."
echo "Date: $(date)"
echo "=================================================="

# Run the python script and redirect the console output into a clean text file
python tactical_clustering.py > top_tactical_clusters_report.txt

echo "=================================================="
echo "Analysis Complete! Check top_tactical_clusters_report.txt"
echo "Date: $(date)"
echo "=================================================="