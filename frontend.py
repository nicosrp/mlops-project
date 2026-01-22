"""Streamlit frontend for football match prediction."""

import json

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Load league mapping
try:
    with open("league_mapping.json", "r") as f:
        LEAGUE_MAPPING = json.load(f)
        # Convert string keys back to int
        LEAGUE_MAPPING = {int(k): v for k, v in LEAGUE_MAPPING.items()}
        # Create reverse mapping for lookup
        LEAGUE_NAME_TO_ID = {v: k for k, v in LEAGUE_MAPPING.items()}

    # Load encoding mapping (raw league_id → encoded value for model)
    with open("league_id_to_encoded.json", "r") as f:
        LEAGUE_ID_TO_ENCODED = json.load(f)
        LEAGUE_ID_TO_ENCODED = {int(k): int(v) for k, v in LEAGUE_ID_TO_ENCODED.items()}
except FileNotFoundError:
    LEAGUE_MAPPING = {636: "Superliga", 752: "Primera Division", 734: "Liga Nacional"}
    LEAGUE_NAME_TO_ID = {v: k for k, v in LEAGUE_MAPPING.items()}
    LEAGUE_ID_TO_ENCODED = {636: 766, 752: 801, 734: 795}

# API endpoint - toggle between local and cloud
# API_URL = "http://localhost:8000"  # Local
API_URL = "https://football-api-617960074163.europe-west1.run.app"  # Cloud Run

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Football Match Outcome Predictor")
st.markdown("Predict the probability of home win, draw, or away win using historical match data")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        """
    This application uses an LSTM model trained on 150,000+ football matches
    to predict match outcomes.

    **Model**: LSTM Neural Network
    **Dataset**: Football Match Probability Prediction
    **Classes**: Home Win / Draw / Away Win
    """
    )

    # Health check
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API is healthy")
        else:
            st.error("❌ API not responding")
    except Exception:
        st.error("❌ Cannot connect to API")

# Main content
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "API Info"])

with tab1:
    st.subheader("Make a Single Match Prediction")

    st.info("📊 Enter team statistics to predict match outcome. Model uses 22 features per match.")
    st.warning("""
    ⚠️ **Important: Understanding Football Prediction Uncertainty**

    The model achieves **64% home win accuracy** on validation data, which means:
    - When a match resulted in a home win, the model correctly predicted "home" 64% of the time
    - However, predictions often show **similar probabilities** (e.g., 34% home, 33% draw, 33% away)
    - The model picks the outcome with the **slightly higher probability**, not with high confidence
    - This is **normal for football** - outcomes depend on many unmeasured factors (tactics, injuries, momentum, referee decisions, luck)

    **Example**: A "home win prediction" might be 34% home vs 32% away vs 34% draw. Home wins by having the highest probability, even if only by 1-2%.
    """)

    # League selection (these are encoded integer IDs from training data)
    st.markdown("### 🏆 League & Coach Selection")

    # Get top 20 most common leagues for easier selection
    popular_leagues = [
        "Premier League",
        "La Liga",
        "Serie A",
        "Bundesliga",
        "Ligue 1",
        "Eredivisie",
        "Primeira Liga",
        "Scottish Premiership",
        "MLS",
        "Liga MX",
        "Superliga",
        "Primera Division",
        "Liga Nacional",
        "Premiership",
    ]
    popular_leagues = [league for league in popular_leagues if league in LEAGUE_NAME_TO_ID]
    all_league_names = sorted(LEAGUE_NAME_TO_ID.keys())

    col_league1, col_league2 = st.columns(2)
    with col_league1:
        home_league_name = st.selectbox(
            "🏠 Home team league",
            options=all_league_names,
            index=all_league_names.index("Superliga") if "Superliga" in all_league_names else 0,
        )
        home_league_raw = LEAGUE_NAME_TO_ID[home_league_name]
        home_league = LEAGUE_ID_TO_ENCODED.get(home_league_raw, 400)  # Get encoded value for model
        home_coach = st.number_input(
            "Home coach ID (0-10000)", 0, 10000, 5000, step=100, help="Numeric coach ID from training data"
        )
    with col_league2:
        away_league_name = st.selectbox(
            "✈️ Away team league",
            options=all_league_names,
            index=all_league_names.index("Primera Division") if "Primera Division" in all_league_names else 1,
        )
        away_league_raw = LEAGUE_NAME_TO_ID[away_league_name]
        away_league = LEAGUE_ID_TO_ENCODED.get(away_league_raw, 400)  # Get encoded value for model
        away_coach = st.number_input(
            "Away coach ID (0-10000)", 0, 10000, 5100, step=100, help="Numeric coach ID from training data"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏠 Home Team")
        home_goals_avg = st.number_input("Goals scored (avg last 5)", 0.0, 5.0, 1.8, step=0.1)
        home_conceded_avg = st.number_input("Goals conceded (avg last 5)", 0.0, 5.0, 0.9, step=0.1)
        home_rating = st.slider("Team rating", 0.0, 10.0, 7.5, step=0.1)
        home_win_rate = st.slider("Win rate %", 0, 100, 55)
        home_form = st.slider("Recent form (1-5)", 1.0, 5.0, 3.8, step=0.1)
        home_draws = st.slider("Draw rate %", 0, 100, 25, key="home_draws")
        home_shots_avg = st.number_input("Shots per game (avg)", 0.0, 30.0, 14.5, step=0.5, key="home_shots")
        home_possession = st.slider("Possession %", 30, 70, 52, key="home_poss")
        home_corners_avg = st.number_input("Corners per game", 0.0, 15.0, 6.2, step=0.5, key="home_corners")
        home_fouls_avg = st.number_input("Fouls per game", 0.0, 25.0, 11.3, step=0.5, key="home_fouls")

    with col2:
        st.markdown("### ✈️ Away Team")
        away_goals_avg = st.number_input("Goals scored (avg last 5)", 0.0, 5.0, 1.3, key="away_goals", step=0.1)
        away_conceded_avg = st.number_input("Goals conceded (avg last 5)", 0.0, 5.0, 1.4, key="away_conceded", step=0.1)
        away_rating = st.slider("Team rating", 0.0, 10.0, 6.2, step=0.1, key="away_rating")
        away_win_rate = st.slider("Win rate %", 0, 100, 40, key="away_win")
        away_form = st.slider("Recent form (1-5)", 1.0, 5.0, 2.9, step=0.1, key="away_form")
        away_draws = st.slider("Draw rate %", 0, 100, 30, key="away_draws")
        away_shots_avg = st.number_input("Shots per game (avg)", 0.0, 30.0, 12.8, step=0.5, key="away_shots")
        away_possession = st.slider("Possession %", 30, 70, 48, key="away_poss")
        away_corners_avg = st.number_input("Corners per game", 0.0, 15.0, 5.4, step=0.5, key="away_corners")
        away_fouls_avg = st.number_input("Fouls per game", 0.0, 25.0, 12.1, step=0.5, key="away_fouls")

    if st.button("⚽ Predict Match Outcome", type="primary", use_container_width=True):
        # StandardScaler parameters from ORIGINAL raw data (before scaling was applied)
        # These are typical football statistics ranges that were scaled during preprocessing
        GOAL_MEAN = 1.5  # Average goals per match
        GOAL_STD = 1.3  # Std dev of goals
        RATING_MEAN = 6.6  # Average team rating (1-10 scale)
        RATING_STD = 0.9  # Std dev of ratings
        DAYS_MEAN = 28.0  # Average days between matches
        DAYS_STD = 15.0  # Std dev of days

        # Use user-selected coach and league IDs (these are already encoded values from processed data)
        # No need for medians - users can pick specific values

        # Create feature vector matching training data format
        # The model expects 10 historical matches, each with 22 features (18 base + 4 aggregated)
        historical_matches = []

        for i in range(10):
            # Simulate historical match i (most recent = 0, oldest = 9)
            days_since = (i + 1) * 7  # 7, 14, 21, ... 70 days ago

            match_features = [
                # Home team features
                (home_goals_avg - GOAL_MEAN) / GOAL_STD,  # home_team_history_goal
                (away_conceded_avg - GOAL_MEAN) / GOAL_STD,  # home_team_history_opponent_goal
                (home_rating - RATING_MEAN) / RATING_STD,  # home_team_history_rating
                (away_rating - RATING_MEAN) / RATING_STD,  # home_team_history_opponent_rating
                1.0,  # home_team_history_is_play_home
                0.0,  # home_team_history_is_cup
                float(home_coach),  # home_team_history_coach (encoded ID)
                float(home_league),  # home_team_history_league_id (encoded ID)
                (days_since - DAYS_MEAN) / DAYS_STD,  # home_team_history_match_days_since
                # Away team features
                (away_goals_avg - GOAL_MEAN) / GOAL_STD,  # away_team_history_goal
                (home_conceded_avg - GOAL_MEAN) / GOAL_STD,  # away_team_history_opponent_goal
                (away_rating - RATING_MEAN) / RATING_STD,  # away_team_history_rating
                (home_rating - RATING_MEAN) / RATING_STD,  # away_team_history_opponent_rating
                0.0,  # away_team_history_is_play_home
                0.0,  # away_team_history_is_cup
                float(away_coach),  # away_team_history_coach (encoded ID)
                float(away_league),  # away_team_history_league_id (encoded ID)
                (days_since - DAYS_MEAN) / DAYS_STD,  # away_team_history_match_days_since
                # Aggregated features (calculated by MyDataset._add_aggregated_features)
                ((home_goals_avg + away_conceded_avg) - 3.0) / 2.2,  # total_goals
                ((home_goals_avg - away_goals_avg) - 0.0) / 1.8,  # goal_diff
                ((home_rating + away_rating) / 2 - RATING_MEAN) / RATING_STD,  # avg_rating
                ((home_rating - away_rating) - 0.0) / 1.3,  # rating_diff
            ]
            historical_matches.append(match_features)

        features = historical_matches

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=5,
            )

            if response.status_code == 200:
                result = response.json()
                probabilities = result["probabilities"]
                prediction = result["prediction"]

                st.success("✅ Prediction Complete!")

                # Calculate relative probabilities
                home_prob = probabilities["home"] * 100
                draw_prob = probabilities["draw"] * 100
                away_prob = probabilities["away"] * 100

                # Determine winner and confidence
                max_prob = max(probabilities.values())
                winner_emoji = {"home": "🏠", "draw": "🤝", "away": "✈️"}
                winner_text = {"home": "HOME WIN", "draw": "DRAW", "away": "AWAY WIN"}
                winner_color = {"home": "#1f77b4", "draw": "#ff7f0e", "away": "#d62728"}

                # Calculate relative likelihood message
                sorted_outcomes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                first_outcome, first_prob = sorted_outcomes[0]
                second_outcome, second_prob = sorted_outcomes[1]

                relative_diff = ((first_prob - second_prob) / second_prob) * 100

                first_name = {"home": "Home", "draw": "a Draw", "away": "Away"}[first_outcome]
                second_name = {"home": "Home", "draw": "a Draw", "away": "Away"}[second_outcome]

                if relative_diff > 5:
                    likelihood_msg = f"{first_name} is **{relative_diff:.1f}% more likely** than {second_name}"
                else:
                    likelihood_msg = f"Very close match - {first_name} slightly favored"

                # Big winner display with relative comparison
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, {winner_color[prediction]}22, {winner_color[prediction]}44); border-radius: 15px; margin: 20px 0;">
                        <h1 style="font-size: 4em; margin: 0;">{winner_emoji[prediction]}</h1>
                        <h2 style="color: {winner_color[prediction]}; margin: 10px 0; font-size: 2.5em;">{winner_text[prediction]}</h2>
                        <p style="font-size: 1.3em; color: #555; margin: 10px 0; font-weight: 500;">{likelihood_msg}</p>
                        <p style="font-size: 1.2em; color: #888; margin: 5px 0;">Absolute probability: {max_prob:.1%}</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                # Show all probabilities
                st.markdown("### 📊 Detailed Probabilities")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("🏠 Home Win", f"{probabilities['home']:.1%}")
                with col2:
                    st.metric("🤝 Draw", f"{probabilities['draw']:.1%}")
                with col3:
                    st.metric("✈️ Away Win", f"{probabilities['away']:.1%}")

                # Bar chart
                df = pd.DataFrame(
                    {
                        "Outcome": ["Home Win", "Draw", "Away Win"],
                        "Probability": [probabilities["home"], probabilities["draw"], probabilities["away"]],
                    }
                )
                st.bar_chart(df.set_index("Outcome"))

            else:
                st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to API: {str(e)}")

with tab2:
    st.subheader("Batch Predictions")
    st.markdown("Upload a CSV file with match features for bulk predictions")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            st.info("Batch prediction not yet implemented in API")

with tab3:
    st.subheader("API Documentation")

    st.markdown("### Available Endpoints")

    st.code(
        """
GET /
    Root endpoint - returns welcome message

GET /health
    Health check endpoint
    Returns: {"status": "healthy"}

POST /predict
    Make prediction for a single match
    Body: {"features": [float, float, ...]}
    Returns: {"predictions": [p_home, p_draw, p_away]}
""",
        language="text",
    )

    st.markdown("### Example Request")
    st.code(
        """
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [1.5, 1.0, 1.2, 1.3]}
)

print(response.json())
# {"predictions": [0.45, 0.30, 0.25]}
""",
        language="python",
    )

st.markdown("---")
st.caption("DTU MLOps Project - Football Match Prediction")
