import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
NUM_CLUSTERS = 4
MODEL_PATH = 'trained_exp1_gat.pth'
DATA_PATH = 'nba_spacing_graphs_full.pt'

# --- MODIFIED EMBEDDING EXTRACTION MODEL ---
class EmbeddingGAT(torch.nn.Module):
    def __init__(self, num_node_features=12, hidden_channels=64):
        super(EmbeddingGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # Extract the Latent Embedding
        embedding = global_mean_pool(x, batch) 
        prob = torch.sigmoid(self.lin(embedding))
        return embedding, prob

def calculate_geometric_spread(x_tensor):
    """Calculates the average distances to identify the tactical setup"""
    off_mask = x_tensor[:, 6] == 1.0
    def_mask = ~off_mask
    handler_mask = x_tensor[:, 8] == 1.0
    
    # Un-normalize coordinates back to real basketball feet (94x50)
    real_coords = x_tensor[:, :2] * torch.tensor([94.0, 50.0])
    
    off_coords = real_coords[off_mask]
    def_coords = real_coords[def_mask]
    
    # 1. Offensive and Defensive Spread (How stretched is the floor?)
    off_spread = torch.pdist(off_coords).mean().item() if len(off_coords) > 1 else 0
    def_spread = torch.pdist(def_coords).mean().item() if len(def_coords) > 1 else 0
    
    # 2. Isolation Index (How far is the ball handler from their teammates?)
    try:
        handler_pos = real_coords[handler_mask]
        teammates_pos = real_coords[off_mask & ~handler_mask]
        iso_index = torch.cdist(handler_pos, teammates_pos).mean().item()
    except:
        iso_index = 0.0 # Fallback if handler flag is missing
        
    return off_spread, def_spread, iso_index

def generate_cluster_name(off_spread, def_spread, iso_index):
    """Translates raw geometric spreads into NBA tactical terminology"""
    
    # Analyze Offense
    if iso_index > 22.0:
        off_name = "Perimeter Isolation"
    elif off_spread > 25.0:
        off_name = "5-Out Spread Offense"
    elif off_spread > 21.0:
        off_name = "Standard Motion Spacing"
    else:
        off_name = "Clustered / Pick & Roll Set"
        
    # Analyze Defense
    if def_spread > 22.0:
        def_name = "Extended Perimeter Defense"
    elif def_spread > 18.0:
        def_name = "Standard Half-Court Defense"
    else:
        def_name = "Collapsed / Paint-Packed Defense"
        
    return f"{off_name} vs {def_name}"

def run_clustering_analysis():
    print("Loading data and model...")
    device = torch.device('cpu') # Enforce CPU to prevent SLURM partition errors
    
    model = EmbeddingGAT(num_node_features=12, hidden_channels=64).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    graphs = torch.load(DATA_PATH, weights_only=False)
    
    event_ids, off_spreads, def_spreads, iso_indices = [], [], [], []
    for g in graphs:
        os, ds, iso = calculate_geometric_spread(g.x)
        off_spreads.append(os)
        def_spreads.append(ds)
        iso_indices.append(iso)
        event_ids.append(g.event_id)

    print("Extracting Latent Embeddings via GNN...")
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    all_embeddings, all_probs = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb, prob = model(batch.x, batch.edge_index, batch.batch)
            all_embeddings.append(emb.cpu().numpy())
            all_probs.append(prob.cpu().numpy())
            
    embeddings_matrix = np.vstack(all_embeddings)
    probs_array = np.vstack(all_probs).flatten()

    print(f"Clustering {len(graphs)} plays into {NUM_CLUSTERS} tactical concepts...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_matrix)

    closest_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings_matrix)

    df = pd.DataFrame({
        'EventID': event_ids,
        'Cluster': cluster_labels,
        'Probability': probs_array,
        'Off_Spread': off_spreads,
        'Def_Spread': def_spreads,
        'Iso_Index': iso_indices
    })

    cluster_stats = df.groupby('Cluster').agg(
        Mean_Prob=('Probability', 'mean'),
        Avg_Off_Spread=('Off_Spread', 'mean'),
        Avg_Def_Spread=('Def_Spread', 'mean'),
        Avg_Iso_Index=('Iso_Index', 'mean'),
        Count=('EventID', 'count')
    ).reset_index()

    # Sort ALL clusters by Highest Probability
    ranked_clusters = cluster_stats.sort_values(by='Mean_Prob', ascending=False)

    print("\n" + "="*75)
    print(f"FULL TACTICAL SPACING CLUSTER RANKING (All {NUM_CLUSTERS} Clusters)")
    print("="*75)

    for i, (_, row) in enumerate(ranked_clusters.iterrows(), 1):
        c_id = int(row['Cluster'])
        archetype_event = event_ids[closest_idx[c_id]]
        
        # Automatically generate the tactical name
        tactical_name = generate_cluster_name(row['Avg_Off_Spread'], row['Avg_Def_Spread'], row['Avg_Iso_Index'])
        
        if i == 1:
            print("\n" + "TOP 3 MOST EFFECTIVE")
        elif i == 4:
            print("\n" + "-"*75)
            print(" "*25 + "OTHER IDENTIFIED CLUSTERS")
            print("-"*75)
            
        print(f"\nRANK #{i} - Cluster {c_id} ({int(row['Count'])} plays)")
        print(f"➜ Setup Name: {tactical_name}")
        print(f"➜ Avg. Scoring Probability: {row['Mean_Prob']*100:.2f}%")
        print(f"   • Offense Spread: {row['Avg_Off_Spread']:.1f} ft | Defense Spread: {row['Avg_Def_Spread']:.1f} ft | Iso Index: {row['Avg_Iso_Index']:.1f} ft")
        print(f"➜ ARCHETYPE PLAY TO VISUALIZE: '{archetype_event}'")

if __name__ == "__main__":
    run_clustering_analysis()