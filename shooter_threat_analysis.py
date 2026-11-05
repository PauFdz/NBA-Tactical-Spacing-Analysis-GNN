import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import random
import pandas as pd

# --- CONFIGURATION ---
EXPERIMENT_ID = 1
TARGET_EVENT_ID = "event_190010"

# --- PLAYER NAME & TEAM LOOKUP ---
ID_TO_NAME, ID_TO_TEAM = {}, {}
try:
    stats_df = pd.read_csv('nba_2014_15_player_stats.csv')
    stats_df.columns = stats_df.columns.str.strip()
    ID_TO_NAME = dict(zip(stats_df['PLAYER_ID'], stats_df['PLAYER_NAME']))
    t_col = 'TEAM_ABBREVIATION' if 'TEAM_ABBREVIATION' in stats_df.columns else 'TEAM'
    if t_col in stats_df.columns:
        ID_TO_TEAM = dict(zip(stats_df['PLAYER_ID'], stats_df[t_col]))
except FileNotFoundError:
    print("WARNING: Stats CSV not found. Using fallback names.")

# --- EMBEDDED MODELS ---
class BaselineGAT(torch.nn.Module):
    def __init__(self, num_node_features=12, hidden_channels=64):
        super(BaselineGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.lin(global_mean_pool(x, batch)))

class DeepWideGAT(torch.nn.Module):
    def __init__(self, num_node_features=12, hidden_channels=128):
        super(DeepWideGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return torch.sigmoid(self.lin(global_mean_pool(x, batch)))

class BaselineGCN(torch.nn.Module):
    def __init__(self, num_node_features=12, hidden_channels=64):
        super(BaselineGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.lin(global_mean_pool(x, batch)))

def draw_half_court(ax=None, color='black', lw=2):
    if ax is None: ax = plt.gca()
    ax.add_patch(Circle((5.25, 25), radius=0.75, linewidth=lw, color=color, fill=False))
    ax.add_patch(Rectangle((0, 17), 19, 16, linewidth=lw, color=color, fill=False))
    ax.add_patch(Arc((19, 25), 12, 12, theta1=-90, theta2=90, linewidth=lw, color=color))
    ax.add_patch(Arc((5.25, 25), 47.5, 47.5, theta1=-68.3, theta2=68.3, linewidth=lw, color=color))
    ax.plot([0, 14], [3, 3], color=color, lw=lw)
    ax.plot([0, 14], [47, 47], color=color, lw=lw)
    ax.plot([47, 47], [0, 50], color=color, lw=lw)
    ax.set_xlim(0, 47); ax.set_ylim(0, 50)
    ax.set_aspect('equal') 
    return ax

def analyze_best_shooter():
    device = torch.device('cpu')
    
    if EXPERIMENT_ID == 1:
        model = BaselineGAT(num_node_features=12, hidden_channels=64).to(device)
        model.load_state_dict(torch.load('trained_exp1_gat.pth', map_location=device, weights_only=True))
    elif EXPERIMENT_ID == 2:
        model = DeepWideGAT(num_node_features=12, hidden_channels=128).to(device)
        model.load_state_dict(torch.load('trained_exp2_deepwide.pth', map_location=device, weights_only=True))
    else:
        model = BaselineGCN(num_node_features=12, hidden_channels=64).to(device)
        model.load_state_dict(torch.load('trained_exp3_gcn.pth', map_location=device, weights_only=True))
    model.eval()
    
    graphs = torch.load('nba_spacing_graphs_full.pt', weights_only=False)
    
    if TARGET_EVENT_ID is not None:
        matched = [g for g in graphs if str(g.event_id) == str(TARGET_EVENT_ID)]
        if not matched:
            print(f"ID {TARGET_EVENT_ID} not found. Defaulting to random.")
            graph = random.choice([g for g in graphs if g.y.item() == 0.0])
        else:
            graph = matched[0]
    else:
        graph = random.choice([g for g in graphs if g.y.item() == 0.0])
        
    graph = graph.to(device)
    x_features = graph.x.clone().detach()

    offense_mask = (x_features[:, 6] == 1.0)
    offense_indices = offense_mask.nonzero(as_tuple=True)[0]
    
    threat_scores = []
    batch = torch.zeros(10, dtype=torch.long)
    
    with torch.no_grad():
        for p_idx in offense_indices:
            threat_x = x_features.clone()
            p_pos = x_features[p_idx, :2] * torch.tensor([94.0, 50.0])
            all_real_xy = x_features[:, :2] * torch.tensor([94.0, 50.0])
            
            threat_x[:, 7] = torch.sqrt(torch.sum((all_real_xy - p_pos)**2, dim=1) + 1e-6) / 94.0
            threat_x[:, 8] = 0.0
            threat_x[p_idx, 8] = 1.0 
            
            prob = model(threat_x, graph.edge_index, batch).item()
            threat_scores.append((p_idx.item(), prob))

    best_node, best_prob = max(threat_scores, key=lambda x: x[1])

    offense_teams, defense_teams = [], []
    offense_players_info, defense_players_info = [], []
    off_counter, def_counter = 1, 1
    labels_dict = {}

    for idx in range(10):
        pid = graph.player_ids[idx].item()
        name = ID_TO_NAME.get(pid, f"Player {pid}")
        p_team = ID_TO_TEAM.get(pid, None)
        fg_pct = x_features[idx, 9].item() * 100
        p3_pct = x_features[idx, 10].item() * 100
        
        if offense_mask[idx]:
            lbl = f"O{off_counter}"
            off_counter += 1
            if p_team: offense_teams.append(p_team)
            prob_val = [p for n, p in threat_scores if n == idx][0] * 100
            offense_players_info.append(f"[{lbl}] {name}\n      FG: {fg_pct:.1f}% | 3P: {p3_pct:.1f}% | Threat: {prob_val:.1f}%\n")
        else:
            lbl = f"D{def_counter}"
            def_counter += 1
            if p_team: defense_teams.append(p_team)
            defense_players_info.append(f"[{lbl}] {name}\n      FG: {fg_pct:.1f}% | 3P: {p3_pct:.1f}%\n")
        labels_dict[idx] = lbl

    off_team_name = max(set(offense_teams), key=offense_teams.count) if offense_teams else None
    def_team_name = max(set(defense_teams), key=defense_teams.count) if defense_teams else None
    
    if off_team_name and def_team_name:
        title_prefix = f"{def_team_name} @ {off_team_name} | "
    else:
        title_prefix = ""

    stats_text = "Player Statistics & Threat\n" + "-"*35 + "\n"
    for p_str in offense_players_info: stats_text += p_str
    stats_text += "-"*35 + "\n"
    for p_str in defense_players_info: stats_text += p_str

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(right=0.7) 
    draw_half_court(ax)

    real_xy = x_features[:, :2].numpy() * np.array([94.0, 50.0])

    for idx in range(10):
        x, y = real_xy[idx]
        lbl = labels_dict[idx]

        if not offense_mask[idx]:
            ax.scatter(x, y, color='red', s=150, edgecolors='black', zorder=5, label='Defense' if idx == 0 else "")
            ax.text(x, y + 1.2, lbl, fontsize=9, fontweight='bold', ha='center', color='darkred', zorder=11)
        else:
            if idx == best_node:
                ax.scatter(x, y, color='gold', marker='*', s=600, edgecolors='black', zorder=10, label='Highest Threat')
                ax.text(x, y + 1.5, lbl, color='green', fontweight='bold', fontsize=11, ha='center', zorder=11)
            else:
                ax.scatter(x, y, color='blue', marker='o', s=200, edgecolors='white', zorder=6, label='Offense' if idx==offense_indices[0] else "")
                ax.text(x, y + 1.2, lbl, color='black', fontweight='bold', fontsize=9, ha='center', zorder=11)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i] and l != ""]
    ax.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(1.35, 1))

    ax.text(1.35, 0.8, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', 
            family='monospace', bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

    plt.title(f"{title_prefix}Play ID: {graph.event_id} (Exp {EXPERIMENT_ID})\nIndividual Spacing Threat Analysis", fontsize=11)
    plt.savefig(f'shooter_threat_exp{EXPERIMENT_ID}_event_190010.png', dpi=300, bbox_inches='tight')
    print(f"Saved as shooter_threat_exp{EXPERIMENT_ID}.png")

if __name__ == "__main__":
    analyze_best_shooter()