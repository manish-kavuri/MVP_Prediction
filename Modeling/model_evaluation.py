import pandas as pd
import numpy as np
import warnings
import joblib
from imblearn.over_sampling import BorderlineSMOTE

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, precision_recall_curve
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# Suppress warnings
warnings.filterwarnings('ignore')

# Load and filter dataset
raw_df = pd.read_csv('MVP_dataset.csv')
df_filtered = raw_df[
    (raw_df['per'] > 17.5) &
    (raw_df['usagePercent'] > 17) &
    (raw_df['offensiveWs'] > 2.5) &
    (raw_df['winShares'] > 5) &
    (raw_df['offensiveBox'] > 2) &
    (raw_df['vorp'] > 2.25) &
    (raw_df['PPG_Score'] > 0.5) &
    (raw_df['APG_Score'] > 0.3) &
    (raw_df['RPG_Score'] > 0.2) &
    (raw_df['ranking'] < 17)
].copy()

drop_cols = ['playerName', 'position_x', 'team', 'points', 'assists', 'totalRb', 'steals', 'blocks',
             'season', 'position_y', 'games_y', 'PPG', 'APG', 'RPG', 'SPG', 'BPG',
             'full_team_name', 'wins', 'effectFgPercent']
df_filtered.drop(columns=[col for col in drop_cols if col in df_filtered.columns], inplace=True)
df_filtered.to_csv('filtered_MVP_dataset.csv', index=False)

# Split features/labels
X = df_filtered.drop('MVP', axis=1)
y = df_filtered['MVP']

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Apply SMOTE
X_train, y_train = BorderlineSMOTE(random_state=42).fit_resample(X_train, y_train)

# Define models and randomized param grids
models = {
    'LogisticRegression': (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')),
        {
            'logisticregression__C': np.logspace(-3, 3, 10),
            'logisticregression__penalty': ['l2'],
            'logisticregression__solver': ['lbfgs']
        }
    ),
    'RandomForest': (
        make_pipeline(StandardScaler(), RandomForestClassifier(class_weight='balanced')),
        {
            'randomforestclassifier__n_estimators': [50, 100, 200, 300],
            'randomforestclassifier__max_depth': [3, 5, 7, 10, None],
            'randomforestclassifier__min_samples_split': [2, 5, 10]
        }
    ),
    'XGBoost': (
        make_pipeline(StandardScaler(), XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0)),
        {
            'xgbclassifier__n_estimators': [50, 100, 200],
            'xgbclassifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'xgbclassifier__max_depth': [3, 5, 7, 10],
            'xgbclassifier__subsample': [0.6, 0.8, 1.0]
        }
    ),
    'SVC': (
        make_pipeline(StandardScaler(), SVC(probability=True)),
        {
            'svc__C': [0.1, 1, 10, 100],
            'svc__kernel': ['linear', 'rbf', 'poly'],
            'svc__gamma': ['scale', 'auto']
        }
    ),
    'MLPClassifier': (
        make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000)),
        {
            'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'mlpclassifier__alpha': [0.0001, 0.001, 0.01],
            'mlpclassifier__learning_rate': ['constant', 'adaptive']
        }
    ),
    'GradientBoosting': (
        make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        {
            'gradientboostingclassifier__n_estimators': [50, 100, 200],
            'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.2],
            'gradientboostingclassifier__max_depth': [3, 5, 7]
        }
    )
}

# Train and evaluate models
results = []
best_model = None
best_f1_score = 0

for name, (model, param_grid) in models.items():
    print(f"Training {name}...")
    search = RandomizedSearchCV(model, param_grid, n_iter=25, scoring='f1', cv=5, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    model = search.best_estimator_

    if hasattr(model, 'predict_proba'):
        val_probs = model.predict_proba(X_val)[:, 1]
    else:
        val_probs = model.decision_function(X_val)
        val_probs = (val_probs - val_probs.min()) / (val_probs.max() - val_probs.min())

    thresholds = np.linspace(0.1, 0.9, 9)
    best_f1, best_thresh = 0, 0.5
    for thresh in thresholds:
        preds = (val_probs >= thresh).astype(int)
        score = f1_score(y_val, preds)
        if score > best_f1:
            best_f1, best_thresh = score, thresh

    final_preds = (val_probs >= best_thresh).astype(int)
    roc = roc_auc_score(y_val, val_probs)
    prec, rec, _ = precision_recall_curve(y_val, val_probs)
    pr_auc = np.trapz(rec, prec)
    conf_matrix = confusion_matrix(y_val, final_preds)

    results.append({
        'Model': name,
        'F1 Score': round(best_f1, 3),
        'ROC AUC': round(roc, 3),
        'PR AUC': round(pr_auc, 3),
        'Best Threshold': round(best_thresh, 2),
        'Best Params': search.best_params_,
        'Confusion Matrix': conf_matrix
    })

    if best_f1 > best_f1_score:
        best_f1_score = best_f1
        best_model = model
        best_threshold = best_thresh

# Save the best model and threshold
joblib.dump(best_model, 'best_mvp_model.pkl')
np.save('best_threshold.npy', best_threshold)

# Display summary
summary_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
print("\n Model Summary:")
print(summary_df[['Model', 'F1 Score', 'ROC AUC', 'PR AUC', 'Best Threshold']])

print("\n Details:")
for row in results:
    print(f"\nModel: {row['Model']}")
    print(f"Confusion Matrix:\n{row['Confusion Matrix']}")
    print(f"Best Params: {row['Best Params']}")
