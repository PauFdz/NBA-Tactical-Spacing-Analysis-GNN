import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from torch_geometric.data import Data
import warnings

# Ignore pandas FutureWarnings to keep the terminal output clean
warnings.simplefilter(action='ignore', category=FutureWarning)

def parse_event_to_dataframe(event_dict, loop_index):
    game_id = event_dict.get('gameid', 'unknown')
    event_info = event_dict.get('event_info', {})
    
    event_id = (event_dict.get('eventId') or event_dict.get('eventid') or 
                event_info.get('eventId') or event_info.get('eventid') or f"event_{loop_index}")
    
    desc_home = event_info.get('desc_home')
    desc_away = event_info.get('desc_away')
    desc_home_str = "" if pd.isna(desc_home) else str(desc_home)
    desc_away_str = "" if pd.isna(desc_away) else str(desc_away)
    play_outcome = f"{desc_home_str} {desc_away_str}".replace("  ", " ").strip()
    
    event_type = event_info.get('type')
    possession_team_id = event_info.get('possession_team_id')
    
    moments = event_dict.get('moments', [])
    rows = []
    
    for moment in moments:
        quarter = moment.get('quarter')
        game_clock = moment.get('game_clock')
        shot_clock = moment.get('shot_clock')
        if shot_clock is None: shot_clock = game_clock
            
        ball = moment.get('ball_coordinates')
        if ball: 
            rows.append({'game_id': game_id, 'event_id': event_id, 'event_type': event_type,                 
                         'possession_team_id': possession_team_id, 'play_outcome': play_outcome, 
                         'quarter': quarter, 'game_clock': game_clock, 'shot_clock': shot_clock,
                         'team_id': -1, 'player_id': -1, 'is_ball': 1,
                         'x': ball.get('x'), 'y': ball.get('y'), 'z': ball.get('z')})
            
        players = moment.get('player_coordinates', [])
        for player in players:
            rows.append({'game_id': game_id, 'event_id': event_id, 'event_type': event_type,                 
                         'possession_team_id': possession_team_id, 'play_outcome': play_outcome, 
                         'quarter': quarter, 'game_clock': game_clock, 'shot_clock': shot_clock,
                         'team_id': player.get('teamid'), 'player_id': player.get('playerid'),
                         'is_ball': 0, 'x': player.get('x'), 'y': player.get('y'), 'z': player.get('z')})
            
    return pd.DataFrame(rows)

def downsample_data(df, target_fps=5, original_fps=25):
    keep_every_n = original_fps // target_fps
    unique_frames = df[['event_id', 'game_clock']].drop_duplicates()
    unique_frames = unique_frames.sort_values(by=['event_id', 'game_clock'], ascending=[True, False])
    unique_frames['frame_idx'] = unique_frames.groupby('event_id').cumcount()
    frames_to_keep = unique_frames[unique_frames['frame_idx'] % keep_every_n == 0]
    return pd.merge(df, frames_to_keep[['event_id', 'game_clock']], on=['event_id', 'game_clock'], how='inner')

def calculate_kinematics(df, dt=0.2):
    df = df.sort_values(by=['event_id', 'player_id', 'game_clock'], ascending=[True, True, False])
    grouped = df.groupby(['event_id', 'player_id'])
    
    dx = grouped['x'].shift(-1) - grouped['x'].shift(1)
    dy = grouped['y'].shift(-1) - grouped['y'].shift(1)
    
    df['v_x'] = dx / (2 * dt)
    df['v_y'] = dy / (2 * dt)
    
    dv_x = grouped['v_x'].shift(-1) - grouped['v_x'].shift(1)
    dv_y = grouped['v_y'].shift(-1) - grouped['v_y'].shift(1)
    
    df['a_x'] = dv_x / (2 * dt)
    df['a_y'] = dv_y / (2 * dt)
    
    df = df.fillna({'v_x': 0, 'v_y': 0, 'a_x': 0, 'a_y': 0})
    return df

def build_graphs_from_dataframe(df):
    print("Building PyTorch Geometric Graphs (with Pure Spatial Possession)...")
    graphs = []
    
    # Process play by play
    grouped_events = df.groupby('event_id')
    
    for event_id, event_data in grouped_events:
        # 1. Determine True Offense from the start of the play
        first_frame_clock = event_data['game_clock'].max() # Game clock counts down, so max is start
        first_frame = event_data[event_data['game_clock'] == first_frame_clock]
        
        b_df = first_frame[first_frame['is_ball'] == 1]
        p_df = first_frame[first_frame['is_ball'] == 0]
        if len(b_df) == 0 or len(p_df) == 0: 
            continue
            
        bx_start = b_df.iloc[0]['x']
        by_start = b_df.iloc[0]['y']
        
        # Player closest to the ball at the start IS the offense
        dists = np.sqrt((p_df['x'] - bx_start)**2 + (p_df['y'] - by_start)**2)
        closest_idx = dists.idxmin()
        true_offense_team = p_df.loc[closest_idx, 'team_id']
        
        # 2. Process all frames with the corrected team logic
        grouped_frames = event_data.groupby('game_clock')
        for game_clock, frame_data in grouped_frames:
            players_data = frame_data[frame_data['is_ball'] == 0].copy()
            ball_data = frame_data[frame_data['is_ball'] == 1]
            
            if len(players_data) != 10 or len(ball_data) != 1: 
                continue
                
            # Team 1.0 is now GUARANTEED to be the attacking team
            players_data['team_binary'] = (players_data['team_id'] == true_offense_team).astype(float)
            
            ball_x = ball_data['x'].iloc[0]
            ball_y = ball_data['y'].iloc[0]

            # 3. Court Unification (Always attack left)
            offense_avg_x = players_data[players_data['team_binary'] == 1.0]['x'].mean()
            if offense_avg_x > 47.0:
                players_data['x'] = 94.0 - players_data['x']
                ball_x = 94.0 - ball_x
                players_data['v_x'] = -players_data['v_x']
                players_data['a_x'] = -players_data['a_x']

            dist_to_ball = np.sqrt((players_data['x'].values - ball_x)**2 + (players_data['y'].values - ball_y)**2)
            
            node_features = np.column_stack((
                players_data['x'].values / 94.0,
                players_data['y'].values / 50.0,
                players_data['v_x'].values / 30.0,
                players_data['v_y'].values / 30.0,
                players_data['a_x'].values / 10.0,
                players_data['a_y'].values / 10.0,
                players_data['team_binary'].values, 
                dist_to_ball / 94.0  
            ))
            
            x_tensor = torch.tensor(node_features, dtype=torch.float)
            edge_indices = [[i, j] for i in range(10) for j in range(10) if i != j]
            edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            play_outcome = players_data['play_outcome'].iloc[0]
            event_type = players_data['event_type'].iloc[0]
            
            # 4. Save ball_pos inside the graph for the Time Machine logic!
            graph = Data(x=x_tensor, edge_index=edge_index_tensor, event_id=event_id, 
                         game_clock=game_clock, outcome=play_outcome, event_type=event_type,
                         ball_pos=torch.tensor([ball_x, ball_y], dtype=torch.float))
            graphs.append(graph)
            
    return graphs

if __name__ == "__main__":
    print("1. Loading MEDIUM dataset from Hugging Face cache...")
    dataset = load_dataset("dcayton/nba_tracking_data_15_16", "medium", split="train", trust_remote_code=True)
    
    print("2. Parsing nested data...")
    parsed_dfs = []
    
    # We are processing the entire medium dataset!
    events_to_process = len(dataset) 
    
    for i in range(events_to_process):
        df_event = parse_event_to_dataframe(dataset[i], loop_index=i)
        if not df_event.empty:
            parsed_dfs.append(df_event)
            
        # Keep track of progress in the .out file
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} / {events_to_process} events...")
            
    df_tracking = pd.concat(parsed_dfs, ignore_index=True)
    df_tracking['possession_team_id'] = df_tracking['possession_team_id'].astype('Int64')
    
    print("3. Downsampling to 5 FPS...")
    df_downsampled = downsample_data(df_tracking)
    
    print("4. Calculating kinematics...")
    df_features = calculate_kinematics(df_downsampled)
    
    print("5. Building PyTorch Geometric Graphs (Unified & Pure Possession)...")
    dataset_graphs = build_graphs_from_dataframe(df_features)
    
    print(f"6. Saving {len(dataset_graphs)} graphs to disk...")
    torch.save(dataset_graphs, 'nba_spacing_graphs.pt')
    print("✅ DATA PIPELINE COMPLETE! Full medium dataset saved.")