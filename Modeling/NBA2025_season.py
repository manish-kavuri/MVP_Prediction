import pandas as pd
import numpy as np
import requests
import joblib
from Scraping.mvp_calculations import get_mvps
from Scraping.team_stats import get_team_stats_by_year

def clean_name(name):
    return name.replace('*', '').replace('\\', '').strip()

def fetch_player_stats(url):
    df = pd.read_html(url)[0]
    df = df[df['Rk'] != 'Rk']
    df = df[df['Player'] != 'League Average']
    df = df[df['Player'].notna()]

    if 'Tm' not in df.columns and 'Team' in df.columns:
        df.rename(columns={'Team': 'Tm'}, inplace=True)

    df = df[df['Tm'] != 'TOT']
    df['Player'] = df['Player'].apply(clean_name)
    return df

def fetch_advanced_stats(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    df = pd.read_html(url)[0]
    df = df[df['Rk'] != 'Rk']
    df = df[df['Player'].notna()]
    df['Player'] = df['Player'].apply(clean_name)
    return df

def normalize_column(df, col):
    col_vals = df[col].astype(float)
    return (col_vals - col_vals.min()) / (col_vals.max() - col_vals.min())

def compute_advanced_scores(df):
    df = df.copy()
    df['PPG'] = df['PTS'].astype(float) / df['G'].astype(float)
    df['APG'] = df['AST'].astype(float) / df['G'].astype(float)
    df['RPG'] = df['TRB'].astype(float) / df['G'].astype(float)
    df['SPG'] = df['STL'].astype(float) / df['G'].astype(float)
    df['BPG'] = df['BLK'].astype(float) / df['G'].astype(float)

    df['PPG_Score'] = normalize_column(df, 'PPG')
    df['APG_Score'] = normalize_column(df, 'APG')
    df['RPG_Score'] = normalize_column(df, 'RPG')
    df['SPG_Score'] = normalize_column(df, 'SPG')
    df['BPG_Score'] = normalize_column(df, 'BPG')
    df['eFG%_Score'] = normalize_column(df, 'eFG%')
    return df

def add_team_context(df, year):
    team_df = pd.DataFrame(get_team_stats_by_year(year))
    team_df.rename(columns={'Team Abbreviation': 'Tm'}, inplace=True)
    df = pd.merge(df, team_df[['Tm', 'win_percentage', 'ranking']], on='Tm', how='left')
    return df

def filter_candidates(df):
    df_filtered = df[
        (df['per'] > 15) &
        (df['winShares'] > 4) &
        (df['ranking'] < 20)
    ]
    return df_filtered

def main():
    year = 2025

    # Step 1: Load per-game stats
    per_game_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    df = fetch_player_stats(per_game_url)

    # Step 2: Compute scoring features
    df = compute_advanced_scores(df)

    # Step 3: Load advanced stats
    adv_df = fetch_advanced_stats(year)
    adv_df.rename(columns={
        'PER': 'per',
        'USG%': 'usagePercent',
        'OWS': 'offensiveWs',
        'WS': 'winShares',
        'OBPM': 'offensiveBox',
        'VORP': 'vorp',
        'DBPM': 'defensiveBox',
        'DWS': 'defensiveWs',
        'G': 'games_x'
    }, inplace=True)

    # Step 4: Merge with advanced stats
    df = pd.merge(df, adv_df[['Player', 'per', 'usagePercent', 'offensiveWs', 'winShares',
                              'offensiveBox', 'vorp', 'defensiveBox', 'defensiveWs', 'games_x']],
                  on='Player', how='left')

    # Step 5: Add team context
    df = add_team_context(df, year)

    # Step 6: Filter players
    df_filtered = filter_candidates(df)

    # Step 7: Drop unused columns
    X_2025 = df_filtered.drop(columns=[
        'Player', 'Tm', 'Pos', 'Age', 'Rk', 'MP', 'GS', 'Awards'
    ], errors='ignore')

    # Step 8: Match model features
    # Load the model and threshold
    model = joblib.load('best_mvp_model.pkl')
    threshold = np.load('best_threshold.npy')

    # Align prediction features
    X_2025 = df_filtered.copy()

    # Remove any columns not used during training
    expected_features = model.named_steps['standardscaler'].feature_names_in_
    missing_features = set(expected_features) - set(X_2025.columns)
    extra_features = set(X_2025.columns) - set(expected_features)

    # Fill missing features with zeros
    for feat in missing_features:
        X_2025[feat] = 0

    # Drop extra features not seen during training
    X_2025 = X_2025[expected_features]

    # Predict MVP probabilities
    mvp_probs = model.predict_proba(X_2025)[:, 1]
    df_filtered['MVP_Prob'] = mvp_probs
    df_top_5 = df_filtered.sort_values(by='MVP_Prob', ascending=False).head(5)

    print("Top 5 MVP Candidates for 2025")
    print(df_top_5[['Player', 'MVP_Prob']])


if __name__ == '__main__':
    main()
