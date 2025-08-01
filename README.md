# 🏀 NBA MVP Prediction (2025 Season)

This project predicts the **Most Valuable Player (MVP)** of the NBA 2025 season using a machine learning pipeline trained on historical player and team statistics.

## 📌 Project Overview

The goal of this project is to identify likely MVP candidates based on advanced and per-game statistics, along with team performance. It includes:

- Web scraping of player stats from [Basketball Reference](https://www.basketball-reference.com/)
- Feature engineering using per-game and advanced metrics
- Integration of team rankings and win percentage
- MVP labeling using historical data
- Model training using multiple classifiers
- Final prediction of MVP probabilities for 2025

## 📂 Project Structure

MVP_PREDICTION/
├── Data/ # Raw and processed datasets
├── EDA/ # Notebooks and scripts for exploratory analysis
├── Modeling/ # Model training, tuning, and evaluation
├── Scraping/ # Scripts for scraping per-game and advanced stats
├── cache/ # Intermediate data (ignored in Git)
├── .gitignore # Ignored files (e.g., virtual env, cache, model files)
├── LICENSE
└── README.md
## 🔍 Data Sources

- **Player stats:** Scraped from Basketball Reference:
  - Per-game stats: `/leagues/NBA_2025_per_game.html`
  - Advanced stats: `/leagues/NBA_2025_advanced.html`
- **Team standings:** Derived from team win percentage and rankings

## 🧠 Model Training

The model was trained using the following:
- Feature filtering based on performance thresholds
- Data balancing with `BorderlineSMOTE`
- Hyperparameter tuning using `RandomizedSearchCV`
- Evaluated on F1-score, ROC-AUC, PR-AUC
- Classifiers used:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - SVM
  - MLP
  - Gradient Boosting

The best-performing model is saved as `best_mvp_model.pkl`.

## 🔮 2025 Prediction


### 🏆 Top 5 MVP Candidates for 2025 
(Note: These values are probabilities from a highly imbalanced dataset, hence the small scale.)

| Rank | Player                  | MVP Probability |
|------|--------------------------|-----------------|
| 1    | Shai Gilgeous-Alexander | 0.000119        |
| 2    | Nikola Jokić             | 0.000116        |
| 3    | Giannis Antetokounmpo    | 0.000106        |
| 4    | James Harden              | 0.000103        |
| 5    | Jayson Tatum            | 0.000102        |
