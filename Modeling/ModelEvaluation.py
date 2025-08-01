from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
import pandas as pd
warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category= DeprecationWarning)


df = pd.read_csv('filtered_MVP_dataset.csv')
model_data = pd.read_csv('MVP_datatset.csv')

# ensure that we have 25 MVPs
mvp_count = model_data[model_data['MVP'] == 1]

# before filtering we want to keep names of players
df['original_index'] = df.index  # Store original index
# drop all irrelevant data
model_data = model_data.drop(columns = ['playerName', 'position_x', 'team', 'points', 'assists', 'totalRb', 'steals', 'blocks', 'season', 'position_y', 'games_y', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'full_team_name', 'wins', 'effectFgPercent', 'turnovers'])

features = model_data.drop('MVP', axis = 1)

labels = model_data['MVP']

# split into training, validation, and testing data (splitting it twice to do this)

# first split: train vs. temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    features, labels, test_size=0.4, stratify=labels, random_state=42)

# second split: val vs. test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

for dataset in [y_train, y_val, y_test]:
    print(round(len(dataset) / len(labels), 2))
    
# function for printing out the best parameters

def print_results(results):
    print('Best parameters: {}\n'.format(results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std*2, 3), params))
        
gb = GradientBoostingClassifier()
parameters = {
    'n_estimators': [5, 50, 250, 500], 
    'max_depth': [1, 3, 4, 7, 9], 
    'learning_rate': [0.01, 0.1, 1, 10, 100]
    }

cv = GridSearchCV(gb, parameters, cv = 5)
cv.fit(X_train, y_train)

print_results(cv)

