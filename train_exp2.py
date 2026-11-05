import torch
import random
import wandb
from collections import defaultdict
import numpy as np
from torch_geometric.loader import DataLoader
from model_exp2 import SpacingGNN

# --- 1. CONFIGURATION ---
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

def filter_and_label_graphs(graphs):
    print("Isolating pure Field Goal setups (Rewinding 2.5s)...")
    
    # Group frames by event_id
    events = defaultdict(list)
    for g in graphs:
        events[g.event_id].append(g)
    
    valid_graphs = []
    makes = 0
    misses = 0
    
    for event_id, frames in events.items():
        try:
            # 1. THE FREE THROW PURGE: Use exact NBA Event Types
            # Type 1 = Made Field Goal, Type 2 = Missed Field Goal
            e_type = int(frames[0].event_type)
            if e_type not in [1, 2]:
                continue
            
            is_make = (e_type == 1)
            
            # 2. Find the Climax (Ball hits the rim/net)
            min_dist = float('inf')
            climax_clock = frames[0].game_clock
            
            for g in frames:
                bx, by = g.ball_pos[0].item(), g.ball_pos[1].item()
                dist_to_hoop = np.sqrt((bx - 5.25)**2 + (by - 25.0)**2)
                if dist_to_hoop < min_dist:
                    min_dist = dist_to_hoop
                    climax_clock = g.game_clock
            
            # 3. THE 3-POINTER FIX: Rewind 2.5 seconds to catch the true tactical setup
            target_clock = climax_clock + 2.5
            
            setup_graph = None
            min_time_diff = float('inf')
            for g in frames:
                diff = abs(g.game_clock - target_clock)
                if diff < min_time_diff:
                    min_time_diff = diff
                    setup_graph = g
                    
            if setup_graph is None:
                continue
                
            # 4. Assign the mathematical label
            if is_make:
                setup_graph.y = torch.tensor([[1.0]], dtype=torch.float)
                makes += 1
            else:
                setup_graph.y = torch.tensor([[0.0]], dtype=torch.float)
                misses += 1
                
            valid_graphs.append(setup_graph)
            
        except Exception:
            continue
        
    print(f"\n--- DATASET FILTERING COMPLETE ---")
    print(f"Total Frames Processed: {len(graphs)}")
    print(f"Pure Tactical Setups Captured: {len(valid_graphs)}")
    print(f"Field Goal Makes: {makes} | Field Goal Misses: {misses}\n")
    
    return valid_graphs

def train():
    # --- 2. INITIALIZE W&B ---
    wandb.init(
        project="nba-spacing-gnn",
        name="Exp2_Deep_Wide_correction",
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
    graphs = filter_and_label_graphs(graphs)
    
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