import os
import json
from datasets import Dataset

def inspect_raw_structure():
    path_to_dataset = "/mnt/work/datasets/sports_analytics/basketball"
    
    # Agafem només el primer fitxer per fer la inspecció ràpida
    arrow_files = [os.path.join(path_to_dataset, f) for f in os.listdir(path_to_dataset) if f.endswith('.arrow')]
    arrow_files.sort()
    
    if not arrow_files:
        print("No arrow files found.")
        return

    print(f"Loading first shard: {arrow_files[0]}")
    shard_ds = Dataset.from_file(arrow_files[0])
    
    # Extraiem l'esdeveniment 0 (la primera jugada del primer partit)
    event_dict = shard_ds[0]
    
    print("\n" + "="*50)
    print("                 RAW EVENT KEYS")
    print("="*50)
    print(list(event_dict.keys()))
    
    print("\n" + "="*50)
    print("              HOME TEAM METADATA")
    print("="*50)
    # Imprimim l'estructura de l'equip local (això hauria de contenir els jugadors)
    print(json.dumps(event_dict.get('home', {}), indent=4))
    
    print("\n" + "="*50)
    print("             VISITOR TEAM METADATA")
    print("="*50)
    print(json.dumps(event_dict.get('visitor', {}), indent=4))
    
    print("\n" + "="*50)
    print("            SINGLE FRAME (MOMENT) SAMPLE")
    print("="*50)
    moments = event_dict.get('moments', [])
    if moments:
        # Imprimim només el primer frame de la jugada per no saturar la pantalla
        print(json.dumps(moments[0], indent=4))
    else:
        print("No moments found in this event.")

if __name__ == "__main__":
    inspect_raw_structure()