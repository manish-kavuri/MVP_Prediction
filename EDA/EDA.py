# -*- coding: utf-8 -*-
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


# read dataset

df = pd.read_csv('MVP_dataset.csv')

# MVPs vs nonMVPs EDA
stat_metrics = ['per', 'usagePercent', 'offensiveWs', 'defensiveWs', 'winShares', 'offensiveBox', 'defensiveWs', 'winShares', 'offensiveBox', 'defensiveBox', 'vorp', 'PPG_Score', 'APG_Score', 'RPG_Score', 'SPG_Score', 'BPG_Score', 'ranking', 'eFG%_Score']

df_mvp = df[df['MVP'] == 1]
df_non_mvp = df[df['MVP'] == 0]

for metric in stat_metrics:
    plt.figure(figsize=(8, 4))
    
    sns.histplot(df_non_mvp[metric], color='red', label='Non-MVPs', kde=True, alpha=0.5)
    sns.histplot(df_mvp[metric], color='blue', label='MVPs', kde=True, alpha=0.5)
    
    plt.title(f'Distribution of {metric} for MVPs vs. Non-MVPs')
    plt.xlabel(metric)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

''' Findings about MVP for each of these statistics:
    PER > 16
    usagePercent > 17
    offensiveWs > 2.5
    defensiveWs was too widespread, no point
    winShares > 5
    offensiveBox > 2
    defensiveBox same thing too widespread
    winShares > 5
    offensiveBox > 2
    defensiveBox again doesnt matter
    vorp > 2.25
    per > 17.5
    PPG_Score > 0.5
    APG_Score > 0.3
    RPG_Score > 0.2
    SPG_Score doesnt matter
    BPG_Score doesnt matter
    ranking > 17
'''
# filter data by these values
df_filtered = df[
    (df['per'] > 17.5) &
    (df['usagePercent'] > 17) &
    (df['offensiveWs'] > 2.5) &
    (df['winShares'] > 5) &
    (df['offensiveBox'] > 2) &
    (df['vorp'] > 2.25) &
    (df['PPG_Score'] > 0.5) &
    (df['APG_Score'] > 0.3) &
    (df['RPG_Score'] > 0.2) &
    (df['ranking'] < 17)
].copy()

# ensure that we have 25 MVPs
mvp_count = df_filtered[df_filtered['MVP'] == 1]

# drop all irrelevant data
df_filtered = df_filtered.drop(columns = ['playerName', 'position_x', 'team', 'points', 'assists', 'totalRb', 'steals', 'blocks', 'season', 'position_y', 'games_y', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'full_team_name', 'wins', 'effectFgPercent'])

df_filtered.to_csv('filtered_MVP_dataset.csv', index=False)

    
    
