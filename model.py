import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear

class SpacingGNN(torch.nn.Module):
    def __init__(self, num_node_features=8, hidden_channels=64):
        super(SpacingGNN, self).__init__()
        
        # 1. Message Passing Layers (GAT)
        # heads=4 means 4 independent attention mechanisms analyzing the spacing
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        
        # 2. Readout / Classifier Layers
        self.lin1 = Linear(hidden_channels, 32)
        self.lin2 = Linear(32, 1) # 1 output neuron for binary classification (Score/No Score)

    def forward(self, x, edge_index, batch):
        # --- PHASE 1: Node level (Players analyzing the court) ---
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training) # Prevent overfitting
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # --- PHASE 2: Graph level (Team Spacing) ---
        # global_mean_pool averages the 10 player vectors into 1 master vector for the frame
        x = global_mean_pool(x, batch)
        
        # --- PHASE 3: Prediction (Will this possession be effective?) ---
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        # Sigmoid squashes the output into a probability between 0.0 (0%) and 1.0 (100%)
        return torch.sigmoid(x)