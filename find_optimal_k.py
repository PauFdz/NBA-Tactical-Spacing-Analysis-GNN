import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MAX_K = 15
MODEL_PATH = 'trained_exp1_gat.pth'
DATA_PATH = 'nba_spacing_graphs_full.pt'

# --- MODIFIED EMBEDDING EXTRACTION MODEL ---
class EmbeddingGAT(torch.nn.Module):
    def __init__(self, num_node_features=12, hidden_channels=64):
        super(EmbeddingGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

def evaluate_clusters():
    print("Loading data and model...")
    device = torch.device('cpu') # Forcing CPU to avoid SLURM GPU partition issues
    
    model = EmbeddingGAT(num_node_features=12, hidden_channels=64).to(device)
    # weights_only=False needed here if the saved model dict contains complex objects, 
    # but usually weights_only=True is fine for state_dicts. Using True for safety.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    model.eval()

    graphs = torch.load(DATA_PATH, weights_only=False)

    print("Extracting Latent Embeddings via GNN...")
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    all_embeddings = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model(batch.x, batch.edge_index, batch.batch)
            all_embeddings.append(emb.numpy())
            
    embeddings_matrix = np.vstack(all_embeddings)

    print(f"Testing k from 2 to {MAX_K}. This will take a few minutes...")
    inertias = []
    silhouette_scores = []
    K_range = range(2, MAX_K + 1)

    for k in K_range:
        print(f"  -> Fitting K-Means for k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_matrix)
        
        inertias.append(kmeans.inertia_)
        
        # Silhouette score can be extremely slow on 100k samples. We use a random sample of 20k to speed it up.
        score = silhouette_score(embeddings_matrix, labels, sample_size=20000, random_state=42)
        silhouette_scores.append(score)

    # --- PLOTTING THE RESULTS ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters ($k$)')
    ax1.set_ylabel('Inertia (WCSS)', color=color)
    ax1.plot(K_range, inertias, marker='o', color=color, linewidth=2, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)  
    ax2.plot(K_range, silhouette_scores, marker='s', color=color, linewidth=2, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Optimal Number of Clusters ($k$)\nElbow Method & Silhouette Analysis', fontsize=14)
    fig.tight_layout()  
    plt.savefig('elbow_method_analysis.png', dpi=300, bbox_inches='tight')
    print("\nDone! Saved 'elbow_method_analysis.png'.")

if __name__ == "__main__":
    evaluate_clusters()