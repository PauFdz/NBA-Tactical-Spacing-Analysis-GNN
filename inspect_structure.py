import torch

# --- CONFIGURATION ---
# Change this number to inspect more or fewer graphs!
NUM_GRAPHS_TO_INSPECT = 500000  

def inspect_graph_structure():
    print("Loading nba_spacing_graphs.pt...")
    graphs = torch.load('nba_spacing_graphs.pt', weights_only=False)
    
    print(f"\nTotal Graphs in Dataset: {len(graphs)}")
    print(f"Inspecting the first {NUM_GRAPHS_TO_INSPECT} graphs...\n")
    
    # Loop through the requested number of graphs
    for i in range(min(NUM_GRAPHS_TO_INSPECT, len(graphs))):
        g = graphs[i]
        
        print("=" * 60)
        print(f"                  GRAPH {i} METADATA ")
        print("=" * 60)
        print(f"Event ID:           {g.event_id}")
        print(f"Game Clock:         {g.game_clock}")
        print(f"Raw Text Outcome:   '{g.outcome}'")
        print(f"NBA Event Type:     {g.event_type} (1=Make, 2=Miss, etc.)")
        print(f"Ball Position:      X: {g.ball_pos[0]:.2f}, Y: {g.ball_pos[1]:.2f}")
        
        print("\n--- NODE MATRIX (10 Players x 8 Features) ---")
        print(f"Shape of 'x' tensor: {g.x.shape}")
        
        # We'll just print Player 1 to save terminal space, but you know there are 10
        p1 = g.x[0].tolist()
        print("  [Player 1 Sample]:")
        print(f"  X, Y:           {p1[0]:.4f}, {p1[1]:.4f}")
        print(f"  vX, vY:         {p1[2]:.4f}, {p1[3]:.4f}")
        print(f"  aX, aY:         {p1[4]:.4f}, {p1[5]:.4f}")
        print(f"  Team Identity:  {p1[6]:.1f} (1.0=Offense, 0.0=Defense)")
        print(f"  Dist to Ball:   {p1[7]:.4f}")
        
        print("\n--- EDGE MATRIX (The Connections) ---")
        print(f"Shape of 'edge_index': {g.edge_index.shape}\n")

if __name__ == "__main__":
    inspect_graph_structure()