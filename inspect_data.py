import torch

def inspect_dataset():
    print("Loading nba_spacing_graphs_full.pt...")
    try:
        graphs = torch.load('nba_spacing_graphs_full.pt', weights_only=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    makes = 0
    misses = 0
    other = 0

    for g in graphs:
        outcome = str(g.outcome).lower()
        if 'pts' in outcome:     # Corrected Make identifier
            makes += 1
        elif 'miss' in outcome:
            misses += 1
        else:
            other += 1

    print("\n--- RAW OUTCOME DISTRIBUTION ---")
    print(f"Makes:  {makes}")
    print(f"Misses: {misses}")
    print(f"Other:  {other}")

if __name__ == "__main__":
    inspect_dataset()