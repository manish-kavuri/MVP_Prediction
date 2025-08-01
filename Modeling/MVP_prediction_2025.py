import pandas as pd
import numpy as np
import joblib

# Load trained model and threshold
model = joblib.load('best_mvp_model.pkl')
threshold = np.load('best_threshold.npy').item()

# Load 2025 candidate dataset
df_2025 = pd.read_csv('2025_mvp_candidates.csv')

# Rename necessary columns to match the trained model
column_renames = {
    'DBPM': 'defensiveBox',
    'DWS': 'defensiveWs',
    'G': 'games_x',
    'TOV': 'turnovers'
}
df_2025.rename(columns=column_renames, inplace=True)

# Backup player names before dropping
player_names = df_2025['Player']

# Drop non-numeric or unnecessary columns not used in training
drop_cols = ['Player', 'Tm', 'Awards']
X_2025 = df_2025.drop(columns=[col for col in drop_cols if col in df_2025.columns])

# Get expected features from the trained model's scaler step
expected_features = model.named_steps['standardscaler'].get_feature_names_out()

# Fill any missing features with 0
missing = [col for col in expected_features if col not in X_2025.columns]
for col in missing:
    X_2025[col] = 0.0

# Ensure correct column order
X_2025 = X_2025[expected_features]

# Predict MVP probabilities
mvp_probs = model.predict_proba(X_2025)[:, 1]
df_2025['MVP_Prob'] = mvp_probs

# Get Top 5 MVP predictions
top_5 = df_2025[['Player', 'MVP_Prob']].sort_values(by='MVP_Prob', ascending=False).head(5)

# Print results
print("=== Top 5 MVP Candidates for 2025 ===")
print(top_5.to_string(index=False))
