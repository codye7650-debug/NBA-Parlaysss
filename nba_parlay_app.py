import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression  # Simple model example
from sklearn.model_selection import train_test_split
import requests  # For real API

# --------------------
# Step 1: Load or train simple model (demo with synthetic/historical)
# In real: Load Kaggle CSV (https://www.kaggle.com/datasets/ehallmar/nba-historical-stats-and-betting-data)
# Features: e.g., home team rating diff, rest days, etc. Target: home_win (1/0)
# Here: Synthetic for demo
@st.cache_data
def get_demo_data():
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'home_rating_diff': np.random.normal(5, 10, n),
        'rest_diff': np.random.normal(0, 1, n),
        'home_win': np.random.binomial(1, 0.6, n)  # ~60% home win bias
    })
    data['home_win_prob'] = 1 / (1 + np.exp(-(0.1 * data['home_rating_diff'] + 0.2 * data['rest_diff'])))
    return data

data = get_demo_data()

# Train simple logistic model
X = data[['home_rating_diff', 'rest_diff']]
y = data['home_win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# --------------------
# Step 2: Parlay generator
def generate_high_prob_parlays(games_df, leg_count=2, min_leg_prob=0.60):
    high_prob = games_df[games_df['pred_prob'] >= min_leg_prob]
    if len(high_prob) < leg_count:
        return pd.DataFrame()
    
    parlay_list = []
    for combo in combinations(high_prob.index, leg_count):
        sel = high_prob.loc[list(combo)]
        comb_prob = np.prod(sel['pred_prob'])
        parlay_list.append({
            'Legs': ', '.join(sel['game']),
            'Combined Win Prob': f"{comb_prob:.2%}",
            'Est. Implied Odds': f"+{int((1/comb_prob - 1) * 100)}" if comb_prob < 1 else "N/A",
            'Games List': sel['game'].tolist()
        })
    parlay_df = pd.DataFrame(parlay_list)
    return parlay_df.sort_values('Combined Win Prob', ascending=False)

# --------------------
# UI
st.title("NBA High-Probability Parlay Predictor MVP")
st.markdown("**Demo:** Uses simple ML + synthetic data. Replace with real historical training & The Odds API for live games.")

# Option: Use real API (uncomment & add your key)
API_KEY = st.secrets.get("THE_ODDS_API_KEY", None)
if not API_KEY:
    st.warning("No Odds API key in secrets → demo mode only.")
SPORT = "basketball_nba"
REGIONS = "us"
MARKETS = "h2h"  # moneylines

if st.button("Fetch Upcoming Games & Predict (Live Mode)"):
    if API_KEY:
        url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}"
        try:
            resp = requests.get(url).json()
            games = []
            for g in resp[:10]:  # Limit for demo
                home = g['home_team']
                away = g['away_team']
                # Dummy pred_prob (replace with real model inference)
                pred_prob = np.random.uniform(0.55, 0.85)  # Simulate high-prob favorites
                games.append({'game': f"{away} @ {home}", 'pred_prob': pred_prob})
            games_df = pd.DataFrame(games)
        except Exception as e:
            st.error(f"API error: {e}")
            games_df = pd.DataFrame()
    else:
        st.info("No API key → using synthetic upcoming games.")
        games = [
            {'game': 'LAL @ BOS', 'pred_prob': 0.78},
            {'game': 'GSW @ DEN', 'pred_prob': 0.68},
            {'game': 'MIA @ NYK', 'pred_prob': 0.82},
            {'game': 'PHX @ DAL', 'pred_prob': 0.62},
            {'game': 'MIL @ PHI', 'pred_prob': 0.71},
        ]
        games_df = pd.DataFrame(games)
    
    if not games_df.empty:
        st.subheader("Predicted High-Prob Legs")
        st.dataframe(games_df.style.format({'pred_prob': '{:.1%}'}))
        
        st.subheader("Top 2-Leg Parlays (Highest Combined Prob)")
        parlays = generate_high_prob_parlays(games_df, leg_count=2)
        if not parlays.empty:
            st.dataframe(parlays)
        else:
            st.warning("No combinations meet min prob threshold.")
else:
    st.info("Click button to generate predictions.")

st.markdown("**Next Steps:** Train on real Kaggle data, add XGBoost, include spreads/totals, player props, or value detection (pred_prob vs. implied odds).")
