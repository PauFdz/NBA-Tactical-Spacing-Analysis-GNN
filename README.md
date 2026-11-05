# 🏀 NBA Tactical Spacing Analysis via Graph Neural Networks (GNNs)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-green)
![Data Engineering](https://img.shields.io/badge/Data%20Engineering-HPC%20%7C%20SLURM-orange)

> **Grade 10/10 (Honors)** | Bachelor's Thesis in Data Science and Engineering (Universitat Politècnica de Catalunya - UPC)

This repository contains the code and methodology for evaluating and quantifying **offensive spacing in basketball** using high-resolution optical tracking data and Deep Learning. By representing basketball possessions as multi-agent dynamic systems, this project utilizes **Graph Attention Networks (GATs)** to extract tactical principles, overriding traditional static geometric metrics.

---

## 🧠 Project Overview

Basketball is inherently a relational, multi-agent system where the value of a player's location depends strictly on the spatial distribution of the other 9 players on the court. This project tackles the translation of abstract coaching concepts (like *spacing* and *gravity*) into a differentiable mathematical framework.

### Key Contributions:
1. **The Climax Finder Algorithm:** A spatio-temporal synchronization pipeline that parses millions of nested JSON frames and rewinds exactly 2.5 seconds from the shot apex to isolate the definitive "tactical trigger" of the possession.
2. **Identity-Free Representation:** Player names are replaced with continuous statistical profiles (FG%, 3P% gravity). The model learns functional tactical concepts rather than memorizing a historical roster.
3. **Graph Attention Network (GAT):** Outperformed isotropic Graph Convolutional Networks (GCNs) by dynamically learning attention coefficients ($\alpha_{ij}$) to prioritize crucial tactical interactions (e.g., ball-handler vs. primary defender).
4. **Spatial Optimization & Heatmaps:** By freezing the trained GNN weights and applying gradient descent on the spatial inputs, the model acts as a tactical simulator that can restructure offensive formations to maximize expected scoring probability.

---

## 📂 Repository Structure

The core files needed to replicate the pipeline and run inference:

*   **`data_prep_full.py`**: The data engineering pipeline (parsing, temporal downsampling to 5Hz, kinematic feature extraction, and Climax Finder synchronization).
*   **`model.py` / `model_exp*.py`**: PyTorch Geometric definitions of the GAT and GCN architectures.
*   **`train.py` / `train_full_exp1.py`**: Training loops, BCE loss optimization, and evaluation metrics (ROC-AUC).
*   **`team_spatial_optimizer.py`**: The reversed gradient-descent script that optimizes player (X,Y) coordinates to maximize scoring probability with soft-collision penalties.
*   **`heatmap_analysis.py` & `shooter_threat_analysis.py`**: Visual diagnostic pipelines mapping latent GNN evaluations to 2D court projections.
*   **`tactical_clustering.py`**: K-Means clustering on the 64-dimensional latent space to extract basketball spacing archetypes.
*   **`trained_exp1_gat.pth`**: The final trained weights for the baseline GAT model.
*   **`Thesis_report_PauFernándezCester.pdf`**: The full academic report detailing the mathematical formulation, HPC deployment on the UPC CALCULA cluster, and results.

---

## 📊 Data Source

Due to GitHub's file size limitations, the raw 90+ GB tracking dataset and the generated PyTorch tensor files (`.pt`) are not hosted in this repository. 

The raw data comes from the **2015-2016 NBA SportVU optical tracking release**. 
To acquire the data:
1. Visit the Hugging Face dataset repository: [`dcayton/nba_tracking_data_15_16`](https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16)
2. Download the partitioned Apache Arrow shards.

---

## 🚀 How to Reproduce

### 1. Environment Setup
Clone the repository and install the exact dependencies to avoid PyTorch Geometric version conflicts:
```bash
git clone [https://github.com/PauFdz/NBA-Tactical-Spacing-Analysis-GNN.git](https://github.com/PauFdz/NBA-Tactical-Spacing-Analysis-GNN.git)
cd NBA-Tactical-Spacing-Analysis-GNN
pip install -r requirements.txt
```

### 2. Generate the Graph Dataset
Once the Hugging Face dataset is downloaded locally, configure the input path inside `data_prep_full.py` and run the pipeline to generate the structured PyTorch `.pt` datasets:
```bash
python data_prep_full.py
```

### 3. Train the Model
To train the Graph Attention Network (Experiment 1) using the isolated tactical frames:
```bash
python train_full_exp1.py
```
*(Note: Original trainings were executed on the UPC CALCULA HPC cluster using SLURM workload managers via `job_train.sh` on NVIDIA RTX 2080 Ti GPUs).*

### 4. Run Visual Diagnostics
To generate the optimization and heatmap court visuals using the pre-trained weights (`trained_exp1_gat.pth`):
```bash
python team_spatial_optimizer.py
python heatmap_analysis.py
```

---

## 📈 Results & Tactical Archetypes

While the model operates under the intrinsic ceiling of human stochastic variance (where a perfectly spaced shot can still miss due to physical execution), it successfully learned robust spatial principles without hardcoded rules. 

Applying **K-Means (k=4)** to the GNN's 64-dimensional latent space revealed 4 distinct tactical archetypes:
1. **Cluster 2:** Perimeter isolation vs. extended defense *(Highest predicted efficiency: ~53%)*
2. **Cluster 0:** Perimeter isolation vs. collapsed defense *(Standard half-court execution)*
3. **Cluster 1:** Motion spacing vs. extended defense
4. **Cluster 3:** Failed transitions / broken plays *(Outliers: <1%)*

For deep technical insights on the mathematical collision penalties, the *over-smoothing* ablation studies, and the *Silhouette Score* cluster optimizations, please refer to the attached PDF thesis report.
