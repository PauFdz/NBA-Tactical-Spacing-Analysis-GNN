import torch
import random
import wandb
from torch_geometric.loader import DataLoader
from model import SpacingGNN

# --- 1. CONFIGURATION ---
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

def assign_binary_labels(graphs):
    """
    Converts the raw text outcome into a binary label (1 = Score, 0 = No Score/Turnover/Rebound)
    """
    processed_graphs = []
    for graph in graphs:
        outcome_text = str(graph.outcome).lower()
        # Simple heuristic: if 'pts' (points) or 'makes' is in the play description, it's a successful play
        if 'pts' in outcome_text or 'makes' in outcome_text:
            graph.y = torch.tensor([[1.0]], dtype=torch.float)
        else:
            graph.y = torch.tensor([[0.0]], dtype=torch.float)
        processed_graphs.append(graph)
    return processed_graphs

def train():
    # --- 2. INITIALIZE W&B ---
    wandb.init(
        project="nba-spacing-gnn",
        config={
            "architecture": "GAT",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE
        }
    )

    # --- 3. DATA LOADING ---
    print("Loading dataset...")
    graphs = torch.load('nba_spacing_graphs.pt', weights_only=False)
    graphs = assign_binary_labels(graphs)
    
    # Shuffle and split: 80% Training, 20% Validation
    random.shuffle(graphs)
    split_idx = int(len(graphs) * 0.8)
    train_dataset = graphs[:split_idx]
    val_dataset = graphs[split_idx:]
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 4. SETUP MODEL & GPU ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = SpacingGNN(num_node_features=8, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss() # Binary Cross Entropy for 1/0 classification

    # --- 5. TRAINING LOOP ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            # Calculate training accuracy
            preds = (out > 0.5).float()
            correct_train += (preds == batch.y).sum().item()
            total_train += batch.num_graphs
            
        train_loss = total_loss / len(train_dataset)
        train_acc = correct_train / total_train

        # --- 6. VALIDATION LOOP ---
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
                
                preds = (out > 0.5).float()
                correct_val += (preds == batch.y).sum().item()
                total_val += batch.num_graphs
                
        val_loss = val_loss / len(val_dataset)
        val_acc = correct_val / total_val

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # --- 7. LOG TO W&B ---
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
    wandb.finish()
    
    # Save the trained model weights
    torch.save(model.state_dict(), 'trained_spacing_gnn.pth')
    print("Training complete! Model saved.")

if __name__ == "__main__":
    train()