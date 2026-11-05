import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

class SpacingGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(SpacingGNN, self).__init__()
        # Using GCNConv instead of GATConv
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.lin(x))