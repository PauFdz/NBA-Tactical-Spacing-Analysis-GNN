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

def optimize_team_spacing():
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
    batch = torch.zeros(10, dtype=torch.long)
    orig_prob = model(x_features, graph.edge_index, batch).item()

    offense_mask = (x_features[:, 6] == 1.0)
    offense_indices = offense_mask.nonzero(as_tuple=True)[0]
    
    handler_local_idx = torch.argmin(x_features[offense_mask, 7]).item()
    ball_handler_global_idx = offense_indices[handler_local_idx].item()

    offense_coords = x_features[offense_mask, :2].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([offense_coords], lr=0.002)

    absolute_best_prob = orig_prob
    absolute_best_coords = offense_coords.clone().detach()

    for _ in range(80):
        optimizer.zero_grad()
        current_x = x_features.clone()
        current_x[offense_mask, :2] = offense_coords
        
        real_xy = offense_coords * torch.tensor([94.0, 50.0])
        ball_pos = real_xy[handler_local_idx]
        
        new_dists = torch.sqrt(torch.sum((real_xy - ball_pos)**2, dim=1) + 1e-6)
        current_x[offense_mask, 7] = new_dists / 94.0
        
        prob = model(current_x, graph.edge_index, batch)
        if prob.item() > absolute_best_prob:
            absolute_best_prob = prob.item()
            absolute_best_coords = offense_coords.clone().detach()

        dist_matrix = torch.cdist(real_xy, real_xy)
        spacing_penalty = torch.sum(torch.relu(4.0 - dist_matrix)) * 0.0001 
        
        loss = -prob + spacing_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_([offense_coords], max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            offense_coords.clamp_(0.02, 0.48) 

    final_x = x_features.clone()
    final_x[offense_mask, :2] = absolute_best_coords
    final_prob = model(final_x, graph.edge_index, batch).item()

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
            offense_players_info.append(f"[{lbl}] {name}\n      FG: {fg_pct:.1f}% | 3P: {p3_pct:.1f}%\n")
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

    stats_text = "Player Statistics\n" + "-"*35 + "\n"
    for p_str in offense_players_info: stats_text += p_str
    stats_text += "-"*35 + "\n"
    for p_str in defense_players_info: stats_text += p_str

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(right=0.7) 
    draw_half_court(ax)

    final_xy = absolute_best_coords.numpy() * np.array([94.0, 50.0])
    orig_xy = x_features[offense_mask, :2].numpy() * np.array([94.0, 50.0])
    defenders_xy = x_features[~offense_mask, :2].numpy() * np.array([94.0, 50.0])

    for i, idx in enumerate((~offense_mask).nonzero(as_tuple=True)[0]):
        x, y = defenders_xy[i]
        ax.scatter(x, y, color='red', s=150, edgecolors='black', label='Defenders' if i==0 else "", zorder=5)
        ax.text(x, y + 1.2, labels_dict[idx.item()], fontsize=9, fontweight='bold', color='darkred', ha='center', zorder=11)
    
    for i in range(5):
        node_id = offense_indices[i].item()
        lbl = labels_dict[node_id]

        if node_id == ball_handler_global_idx:
            ax.scatter(orig_xy[i,0], orig_xy[i,1], color='gold', alpha=0.5, s=250, label='Orig. Ball Handler', edgecolors='black', zorder=4)
        else:
            leg_lbl = 'Original Position' if i == 0 else ""
            ax.scatter(orig_xy[i,0], orig_xy[i,1], color='blue', alpha=0.2, s=150, label=leg_lbl, zorder=4)
        
        ax.arrow(orig_xy[i,0], orig_xy[i,1], final_xy[i,0]-orig_xy[i,0], final_xy[i,1]-orig_xy[i,1], 
                 head_width=0.8, color='gray', linestyle='--', length_includes_head=True, alpha=0.7, zorder=3)
        
        leg_lbl_new = 'Optimized Position' if i == 0 else ""
        ax.scatter(final_xy[i,0], final_xy[i,1], color='blue', s=200, edgecolors='white', label=leg_lbl_new, zorder=6)
        ax.text(final_xy[i,0], final_xy[i,1] + 1.2, lbl, fontsize=10, ha='center', weight='bold', zorder=11)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for j, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:j] and l != ""]
    ax.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(1.35, 1))

    ax.text(1.35, 0.8, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', 
            family='monospace', bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

    plt.title(f"{title_prefix}Play ID: {graph.event_id} (Exp {EXPERIMENT_ID})\nGlobal Team Optimization\nOriginal Prob: {orig_prob*100:.1f}%  -->  Optimized Prob: {final_prob*100:.1f}%", fontsize=12)
    plt.savefig(f'team_spacing_optimization_exp{EXPERIMENT_ID}_event_190010.png', dpi=300, bbox_inches='tight')
    print(f"Saved as team_spacing_optimization_exp{EXPERIMENT_ID}.png")

if __name__ == "__main__":
    optimize_team_spacing()