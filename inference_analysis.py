import torch
from model import SpacingGNN
from torch_geometric.loader import DataLoader

def analyze_spacing_sensitivity():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the model
    model = SpacingGNN(num_node_features=7, hidden_channels=64).to(device)
    model.load_state_dict(torch.load('trained_spacing_gnn.pth', map_location=device))
    model.eval()

    # 2. Load your graphs
    graphs = torch.load('nba_spacing_graphs.pt', weights_only=False)
    sample_graph = graphs[100].to(device) # Pick any frame from a live play

    # 3. Get Baseline Prediction
    # We add a fake batch dimension because the model expects it [10, 7] -> [1, 10, 7]
    batch = torch.zeros(10, dtype=torch.long).to(device)
    with torch.no_grad():
        baseline_prob = model(sample_graph.x, sample_graph.edge_index, batch).item()

    print(f"Original Description: {sample_graph.outcome}")
    print(f"Baseline Scoring Probability: {baseline_prob:.4f}")

    # 4. HYPOTHETICAL: Move an offensive player 5 feet (0.05 scaled) further out
    # Let's find the first offensive player (team_binary is the 7th column, index 6)
    modified_x = sample_graph.x.clone()
    
    # Heuristic: Find player furthest from the basket and move them even further
    # Assuming offensive team is 1.0
    offense_indices = (modified_x[:, 6] == 1.0).nonzero(as_tuple=True)[0]
    player_to_move = offense_indices[0] # Just pick the first offensive player for this test
    
    # Move them +5 feet in X direction (approx 0.05 in our 0-1 scale)
    modified_x[player_to_move, 0] += 0.05 

    with torch.no_grad():
        new_prob = model(modified_x, sample_graph.edge_index, batch).item()

    print(f"Modified Scoring Probability (after moving player): {new_prob:.4f}")
    print(f"Impact of Spacing Change: {((new_prob - baseline_prob) / baseline_prob) * 100:.2f}%")

if __name__ == "__main__":
    analyze_spacing_sensitivity()