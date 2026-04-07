# MLB Prediction

MLB game outcome prediction using historical stats from pybaseball (Baseball Reference, FanGraphs, Statcast).

## Setup

```bash
pip3 install -r requirements.txt
```

## Fetch Data

```bash
python3 utils/data_fetcher.py
```

Downloads game results, team batting/pitching stats, and park factors for 2022-2025 to `data/raw/`.

## Project Structure

```
mlb-prediction/
  data/raw/             # Downloaded data (git-ignored)
  data/processed/       # Cleaned feature matrices
  models/               # Prediction models (Elo, LR, XGBoost, LightGBM)
  strategies/           # Strategy comparison and evaluation
  utils/                # Data fetching and feature engineering
  notebooks/            # Exploratory analysis
```

## Models

- **Elo** (`models/elo.py`) -- Baseline rating system
- **Logistic Regression** (`models/logistic.py`) -- Feature-based baseline
- **XGBoost** (`models/xgboost_model.py`) -- Gradient boosting
- **LightGBM** (`models/lgbm_model.py`) -- Gradient boosting (fast)
- **Ensemble** (`models/ensemble.py`) -- Weighted combination of best models

## Key Features

- Starting pitcher quality (FIP, WHIP, K/9, BB/9)
- Team offense splits (OPS, wOBA vs LHP/RHP)
- Bullpen strength (rolling bullpen ERA/FIP)
- Elo ratings with margin-of-victory adjustment
- Park factors
- Rest days and travel
- Home advantage (~54% baseline in MLB)
