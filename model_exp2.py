import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear

class SpacingGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128): # Changed to 128
        super(SpacingGNN, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels) # Added 3rd layer
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index).relu() # Apply 3rd layer
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.lin(x))