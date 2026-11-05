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
EXPERIMENT_ID = 1       # 1, 2, OR 3
TARGET_EVENT_ID = "event_271409"   # Specify an event ID string (e.g., "event_236669") or None for random

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
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
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

def generate_spacing_heatmap():
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
    original_x = graph.x.clone()

    offense_mask = (original_x[:, 6] == 1.0)
    offense_idx = offense_mask.nonzero(as_tuple=True)[0]
    
    handler_local_idx = torch.argmin(original_x[offense_mask, 7]).item()
    ball_handler_idx = offense_idx[handler_local_idx].item()
    
    furthest_local_idx = torch.argmax(original_x[offense_mask, 7]).item()
    target_player_idx = offense_idx[furthest_local_idx].item()

    ball_x_real = original_x[ball_handler_idx, 0].item() * 94.0
    ball_y_real = original_x[ball_handler_idx, 1].item() * 50.0

    grid_x = np.linspace(0, 47, 40) 
    grid_y = np.linspace(0, 50, 40)
    heatmap = np.zeros((len(grid_y), len(grid_x)))
    batch = torch.zeros(10, dtype=torch.long)

    with torch.no_grad():
        for i, y_val in enumerate(grid_y):
            for j, x_val in enumerate(grid_x):
                test_x = original_x.clone()
                test_x[target_player_idx, 0] = x_val / 94.0
                test_x[target_player_idx, 1] = y_val / 50.0
                new_dist = np.sqrt((x_val - ball_x_real)**2 + (y_val - ball_y_real)**2)
                test_x[target_player_idx, 7] = new_dist / 94.0
                heatmap[i, j] = model(test_x, graph.edge_index, batch).item()

    max_idx = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    optimal_y = grid_y[max_idx[0]]
    optimal_x = grid_x[max_idx[1]]
    max_prob = heatmap[max_idx[0], max_idx[1]]
    orig_prob = model(original_x, graph.edge_index, batch).item()

    offense_teams, defense_teams = [], []
    off_lines, def_lines = [], []
    off_counter, def_counter = 1, 1
    labels_dict = {}

    for idx in range(10):
        pid = graph.player_ids[idx].item()
        name = ID_TO_NAME.get(pid, f"Player {pid}")
        p_team = ID_TO_TEAM.get(pid, None)
        fg_pct = original_x[idx, 9].item() * 100
        p3_pct = original_x[idx, 10].item() * 100
        
        if offense_mask[idx]:
            lbl = f"O{off_counter}"
            off_counter += 1
            if p_team: offense_teams.append(p_team)
            off_lines.append(f"[{lbl}] {name} ({pid})\n      FG: {fg_pct:.1f}% | 3P: {p3_pct:.1f}%")
        else:
            lbl = f"D{def_counter}"
            def_counter += 1
            if p_team: defense_teams.append(p_team)
            def_lines.append(f"[{lbl}] {name} ({pid})\n      FG: {fg_pct:.1f}% | 3P: {p3_pct:.1f}%")
        labels_dict[idx] = lbl

    off_team_name = max(set(offense_teams), key=offense_teams.count) if offense_teams else None
    def_team_name = max(set(defense_teams), key=defense_teams.count) if defense_teams else None

    # Conditional formatting based on whether teams were found
    if off_team_name and def_team_name:
        title_prefix = f"{def_team_name} @ {off_team_name} | "
        off_lbl = f"OFFENSE ({off_team_name})"
        def_lbl = f"DEFENSE ({def_team_name})"
    else:
        title_prefix = ""
        off_lbl = "OFFENSE"
        def_lbl = "DEFENSE"

    # Stack the text vertically for the side panel
    stats_text = f"{off_lbl}:\n" + "-"*35 + "\n"
    for p_str in off_lines: stats_text += p_str + "\n"
    stats_text += f"\n{def_lbl}:\n" + "-"*35 + "\n"
    for p_str in def_lines: stats_text += p_str + "\n"

    # Widen the figure to 15x10 to give the right panel plenty of room
    fig, ax = plt.subplots(figsize=(15, 10)) 
    # Squash the plot to the left 65% of the screen, leaving 35% for the text and legend
    fig.subplots_adjust(right=0.65)
    
    draw_half_court(ax)
    
    im = ax.imshow(heatmap, extent=[0, 47, 0, 50], origin='lower', cmap='RdYlGn', alpha=0.6, zorder=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Predicted Scoring Probability', rotation=270, labelpad=15)
    
    real_x = original_x[:, 0].numpy() * 94.0
    real_y = original_x[:, 1].numpy() * 50.0

    for idx in range(10):
        lbl = labels_dict[idx]
        if idx == target_player_idx:
            ax.scatter(real_x[idx], real_y[idx], color='blue', marker='P', s=250, zorder=5, edgecolors='white', label='Target Original')
            ax.text(real_x[idx], real_y[idx] + 1.2, lbl, fontsize=10, ha='center', weight='bold')
            continue
            
        color = 'blue' if original_x[idx, 6] == 1.0 else 'red'
        ax.scatter(real_x[idx], real_y[idx], color=color, marker='o', s=150, zorder=4, edgecolors='white')
        ax.text(real_x[idx], real_y[idx] + 1.2, lbl, fontsize=9, ha='center', weight='bold')
        
        if idx == ball_handler_idx:
            ax.scatter(real_x[idx], real_y[idx], color='gold', marker='*', s=300, zorder=6, edgecolors='black', label='Ball Handler')

    ax.scatter(optimal_x, optimal_y, color='lime', marker='*', s=400, zorder=7, edgecolors='black', label='Optimal Spacing')
    ax.plot([real_x[target_player_idx], optimal_x], [real_y[target_player_idx], optimal_y], color='black', linestyle='--', linewidth=2, zorder=3)

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for j, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:j]]
    
    # Place legend safely to the right of the colorbar (x=1.22)
    ax.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(1.22, 1))
    
    # Place text box safely to the right of the colorbar and below the legend
    ax.text(1.22, 0.85, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', 
            family='monospace', bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))

    plt.title(f"{title_prefix}Play ID: {graph.event_id} (Exp {EXPERIMENT_ID})\nOriginal Prob: {orig_prob*100:.1f}%  -->  Optimal Prob: {max_prob*100:.1f}%", fontsize=13)
    plt.savefig(f'spacing_heatmap_exp{EXPERIMENT_ID}_event_271409.png', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as spacing_heatmap_exp{EXPERIMENT_ID}.png!")

if __name__ == "__main__":
    generate_spacing_heatmap()