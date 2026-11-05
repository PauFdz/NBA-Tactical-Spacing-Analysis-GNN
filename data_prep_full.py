import pandas as pd
import numpy as np
import torch
import os
import time
from datasets import Dataset
from torch_geometric.data import Data
from nba_api.stats.endpoints import leaguedashplayerstats
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# PHASE 1: API INTEGRATION & STATS CACHING
# ==========================================
def fetch_and_cache_player_stats():
    stats_file = 'nba_2014_15_player_stats.csv'
    if os.path.exists(stats_file):
        print(f"Loading player stats from local cache ({stats_file})...")
        return pd.read_csv(stats_file)
    
    print("Fetching 2014-15 player stats from NBA API (This takes a few seconds)...")
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2014-15')
        df = stats.get_data_frames()[0]
        df_clean = df[['PLAYER_ID', 'PLAYER_NAME', 'FG_PCT', 'FG3_PCT', 'FT_PCT']]
        df_clean.to_csv(stats_file, index=False)
        time.sleep(1) # Polite delay
        return df_clean
    except Exception as e:
        print(f"Failed to fetch from NBA API: {e}")
        # Fallback empty dataframe if API is down
        return pd.DataFrame(columns=['PLAYER_ID', 'PLAYER_NAME', 'FG_PCT', 'FG3_PCT', 'FT_PCT'])

# ==========================================
# CORE PIPELINE FUNCTIONS
# ==========================================
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

def process_and_isolate_setup(event_dict, loop_index):
    df = parse_event_to_dataframe(event_dict, loop_index)
    if df.empty: return None

    e_type = df['event_type'].iloc[0]
    if pd.isna(e_type) or int(e_type) not in [1, 2]:
        return None

    df = downsample_data(df)
    if df.empty: return None
    df = calculate_kinematics(df)

    first_clock = df['game_clock'].max()
    first_frame = df[df['game_clock'] == first_clock]
    b_df = first_frame[first_frame['is_ball'] == 1]
    p_df = first_frame[first_frame['is_ball'] == 0]
    if len(b_df) == 0 or len(p_df) == 0: return None

    bx_start, by_start = b_df.iloc[0]['x'], b_df.iloc[0]['y']
    dists = np.sqrt((p_df['x'] - bx_start)**2 + (p_df['y'] - by_start)**2)
    true_offense_team = p_df.loc[dists.idxmin(), 'team_id']

    ball_tracking = df[df['is_ball'] == 1]
    if ball_tracking.empty: return None
    
    dist_left = np.sqrt((ball_tracking['x'] - 5.25)**2 + (ball_tracking['y'] - 25.0)**2)
    dist_right = np.sqrt((ball_tracking['x'] - 88.75)**2 + (ball_tracking['y'] - 25.0)**2)
    min_dists = np.minimum(dist_left, dist_right)
    
    climax_idx = min_dists.idxmin()
    climax_clock = ball_tracking.loc[climax_idx, 'game_clock']

    target_clock = climax_clock + 2.5
    df['clock_diff'] = (df['game_clock'] - target_clock).abs()
    best_clock = df.loc[df['clock_diff'].idxmin(), 'game_clock']
    
    setup_frame = df[df['game_clock'] == best_clock].copy()
    if len(setup_frame[setup_frame['is_ball'] == 0]) != 10 or len(setup_frame[setup_frame['is_ball'] == 1]) != 1:
        return None
        
    setup_frame['true_offense_team'] = true_offense_team
    return setup_frame

def build_graphs_from_setups(df, stats_df):
    graphs = []
    grouped = df.groupby('event_id')
    
    # Create O(1) lookup dictionary for API stats
    # Defaults: 45% FG, 35% 3PT, 75% FT if player is unknown/rookie
    stats_dict = {}
    for _, row in stats_df.iterrows():
        stats_dict[row['PLAYER_ID']] = {
            'fg': row['FG_PCT'], 'fg3': row['FG3_PCT'], 'ft': row['FT_PCT']
        }
    DEFAULT_STATS = {'fg': 0.45, 'fg3': 0.35, 'ft': 0.75}

    for event_id, frame_data in grouped:
        players_data = frame_data[frame_data['is_ball'] == 0].copy()
        ball_data = frame_data[frame_data['is_ball'] == 1]
        if len(players_data) != 10 or len(ball_data) != 1: continue
            
        true_offense_team = players_data['true_offense_team'].iloc[0]
        players_data['team_binary'] = (players_data['team_id'] == true_offense_team).astype(float)
        
        ball_x, ball_y = ball_data['x'].iloc[0], ball_data['y'].iloc[0]

        offense_avg_x = players_data[players_data['team_binary'] == 1.0]['x'].mean()
        if offense_avg_x > 47.0:
            players_data['x'] = 94.0 - players_data['x']
            ball_x = 94.0 - ball_x
            players_data['v_x'] = -players_data['v_x']
            players_data['a_x'] = -players_data['a_x']

        dist_to_ball = np.sqrt((players_data['x'].values - ball_x)**2 + (players_data['y'].values - ball_y)**2)
        
        # --- PHASE 2: ISOLATE BALL HANDLER ---
        offense_mask = players_data['team_binary'].values == 1.0
        dist_offense = np.where(offense_mask, dist_to_ball, np.inf) 
        bh_index = np.argmin(dist_offense)
        
        # --- PHASE 1: INJECT SHOOTING STATS, HANDLER FLAG & PLAYER IDS ---
        fg_pcts, fg3_pcts, ft_pcts, is_ball_handler, player_ids = [], [], [], [], []
        
        for idx, pid in enumerate(players_data['player_id'].values):
            p_stats = stats_dict.get(pid, DEFAULT_STATS)
            fg_pcts.append(p_stats['fg'])
            fg3_pcts.append(p_stats['fg3'])
            ft_pcts.append(p_stats['ft'])
            is_ball_handler.append(1.0 if idx == bh_index else 0.0)
            player_ids.append(pid)
            
        node_features = np.column_stack((
            players_data['x'].values / 94.0, players_data['y'].values / 50.0,
            players_data['v_x'].values / 30.0, players_data['v_y'].values / 30.0,
            players_data['a_x'].values / 10.0, players_data['a_y'].values / 10.0,
            players_data['team_binary'].values, 
            dist_to_ball / 94.0,
            np.array(is_ball_handler),  
            np.array(fg_pcts),          
            np.array(fg3_pcts),         
            np.array(ft_pcts)           
        ))
        
        x_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_indices = [[i, j] for i in range(10) for j in range(10) if i != j]
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        event_type = int(players_data['event_type'].iloc[0])
        y_label = 1.0 if event_type == 1 else 0.0 
        
        graph = Data(x=x_tensor, edge_index=edge_index_tensor, event_id=event_id, 
                     game_id=players_data['game_id'].iloc[0], 
                     game_clock=players_data['game_clock'].iloc[0], 
                     outcome=players_data['play_outcome'].iloc[0], event_type=event_type,
                     y=torch.tensor([[y_label]], dtype=torch.float),
                     ball_pos=torch.tensor([ball_x, ball_y], dtype=torch.float),
                     player_ids=torch.tensor(player_ids, dtype=torch.long))
        graphs.append(graph)
        
    return graphs

if __name__ == "__main__":
    path_to_dataset = "/mnt/work/datasets/sports_analytics/basketball"
    arrow_files = [os.path.join(path_to_dataset, f) for f in os.listdir(path_to_dataset) if f.endswith('.arrow')]
    arrow_files.sort()
    
    # 1. Fetch the NBA API Stats
    player_stats_df = fetch_and_cache_player_stats()
    
    print(f"2. Reading {len(arrow_files)} Arrow shards applying Early Filtering...")
    
    setup_frames_list = []
    loop_index = 0
    
    for file_path in arrow_files:
        print(f"Parsing shard: {file_path}")
        try:
            shard_ds = Dataset.from_file(file_path)
            for event_dict in shard_ds:
                setup_df = process_and_isolate_setup(event_dict, loop_index)
                if setup_df is not None:
                    setup_frames_list.append(setup_df)
                
                loop_index += 1
                if loop_index % 5000 == 0:
                    print(f"Processed {loop_index} events... Saved {len(setup_frames_list)} setups so far.")
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            continue

    print(f"\n3. Concatenating {len(setup_frames_list)} pure setup frames...")
    df_tracking = pd.concat(setup_frames_list, ignore_index=True)
    
    print("4. Building PyTorch Geometric Graphs with 12 Node Features...")
    dataset_graphs = build_graphs_from_setups(df_tracking, player_stats_df)
    
    output_filename = 'nba_spacing_graphs_full.pt'
    print(f"5. Saving {len(dataset_graphs)} graphs to {output_filename}...")
    torch.save(dataset_graphs, output_filename)
    print("DATA PIPELINE COMPLETE! 12-Feature GNN dataset saved.")