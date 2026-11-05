import torch
import torch.nn.functional as F
import wandb
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import random
from torch_geometric.nn import GATConv, global_mean_pool

# --- 1. CONFIGURATION ---
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

# --- 2. ARCHITECTURE DEFINITION ---
class BaselineGAT(torch.nn.Module):
    def __init__(self, num_node_features=12, hidden_channels=64):
        super(BaselineGAT, self).__init__()
        # Baseline GAT with 2 layers and 64 channels
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling to get a single vector per graph
        x = global_mean_pool(x, batch)
        
        # Linear layer and Sigmoid for binary classification
        x = self.lin(x)
        return torch.sigmoid(x)

def train():
    # --- 3. INITIALIZE W&B ---
    wandb.init(
        project="nba-spacing-gnn",
        name="Exp1_Baseline_GAT_FULL_new",
        config={
            "architecture": "GAT_Baseline",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dataset": "full_600_games"
        }
    )

    # --- 4. DATA LOADING & GROUP SPLITTING ---
    print("Loading pre-filtered dataset (nba_spacing_graphs_full.pt)...")
    graphs = torch.load('nba_spacing_graphs_full.pt', weights_only=False)
    print(f"Loaded {len(graphs)} pure tactical setups.")
    
    # Extract game_ids to group the split
    game_ids = [g.game_id for g in graphs]
    
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(graphs, groups=game_ids))
    
    train_dataset = [graphs[i] for i in train_idx]
    val_dataset = [graphs[i] for i in val_idx]
    
    print(f"Split by Game ID: {len(train_dataset)} train graphs, {len(val_dataset)} val graphs.")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 5. SETUP MODEL & GPU ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = BaselineGAT(num_node_features=12, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss() # Binary Cross Entropy

    # --- 6. TRAINING LOOP ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct_train, total_train = 0, 0, 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            preds = (out > 0.5).float()
            correct_train += (preds == batch.y).sum().item()
            total_train += batch.num_graphs
            
        train_loss = total_loss / len(train_dataset)
        train_acc = correct_train / total_train

        # --- 7. VALIDATION LOOP ---
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
                
                preds = (out > 0.5).float()
                correct_val += (preds == batch.y).sum().item()
                total_val += batch.num_graphs
                
                # Store predictions and labels for ROC-AUC
                all_val_preds.extend(out.cpu().numpy())
                all_val_labels.extend(batch.y.cpu().numpy())
                
        val_loss = val_loss / len(val_dataset)
        val_acc = correct_val / total_val
        
        # Calculate ROC-AUC
        try:
            val_auc = roc_auc_score(all_val_labels, all_val_preds)
        except ValueError:
            val_auc = 0.5 # Fallback if there's only one class in a tiny batch

        print(f'Epoch: {epoch:03d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}')
        
        wandb.log({
            "epoch": epoch, 
            "train_loss": train_loss, 
            "train_acc": train_acc,
            "val_loss": val_loss, 
            "val_acc": val_acc,
            "val_auc": val_auc
        })
        
    wandb.finish()
    torch.save(model.state_dict(), 'trained_exp1_gat.pth')
    print("Training complete! Model saved.")

if __name__ == "__main__":
    train()