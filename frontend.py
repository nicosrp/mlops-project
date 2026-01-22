"""
Streamlit Football Match Prediction Frontend - Version 2
Users input raw historical match data (10 matches per team)
Frontend converts to scaled features matching model training format
"""

import json

import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide",
)

# API URL toggle
API_URL = "https://football-api-617960074163.europe-west1.run.app"
# API_URL = "http://localhost:8000"  # For local testing

# Scaling parameters from training data
GOAL_MEAN, GOAL_STD = 1.5, 1.3
RATING_MEAN, RATING_STD = 6.6, 0.9
DAYS_MEAN, DAYS_STD = 28.0, 15.0

# ======================
# Header
# ======================
st.title("⚽ Football Match Outcome Predictor")
st.markdown(
    """
    Predict match outcomes using **LSTM with Attention** trained on 110K+ historical matches.
    Enter the last 10 matches for each team, and the model will predict: **Home Win**, **Draw**, or **Away Win**.
    """
)

# ======================
# Load example games
# ======================
try:
    with open("example_games_real.json") as f:
        example_games_data = json.load(f)
    example_options = ["➕ Custom Match (enter manually)"] + [f"{game['name']}" for game in example_games_data]
except FileNotFoundError:
    example_games_data = []
    example_options = ["➕ Custom Match (enter manually)"]

# ======================
# Load league mappings
# ======================
try:
    with open("league_mapping.json") as f:
        league_mapping = json.load(f)
    with open("league_id_to_encoded.json") as f:
        league_id_to_encoded = json.load(f)
except FileNotFoundError:
    st.error("❌ League mapping files not found. Run create_league_mapping.py first.")
    st.stop()

# ======================
# Example Selection
# ======================
st.markdown("---")
selected_example = st.selectbox(
    "📚 Load Example Match or Enter Custom Data",
    example_options,
    index=0,
    help="Select a real match from validation data or enter your own",
)

example_data = None
if selected_example != "➕ Custom Match (enter manually)":
    example_name = selected_example
    example_data = next((g for g in example_games_data if g["name"] == example_name), None)

# ======================
# Match Context
# ======================
st.markdown("---")
st.markdown("### ⚙️ Match Context")

col1, col2 = st.columns(2)

with col1:
    # League selection
    league_names = list(league_mapping.values())
    if example_data:
        default_league_idx = next(
            (i for i, (lid, name) in enumerate(league_mapping.items()) if int(lid) == example_data["league_id"]), 0
        )
        league_name = st.selectbox("Match League", league_names, index=default_league_idx, disabled=True)
    else:
        league_name = st.selectbox("Match League", league_names, index=0)

    league_id = next(lid for lid, name in league_mapping.items() if name == league_name)
    league_encoded = league_id_to_encoded.get(str(league_id), 0)

with col2:
    # Coach IDs
    col2a, col2b = st.columns(2)
    with col2a:
        if example_data:
            home_coach = st.number_input("Home Coach ID", 0, 100_000_000, example_data["home_coach"], disabled=True)
        else:
            home_coach = st.number_input("Home Coach ID", 0, 100_000_000, 12345, help="Encoded coach identifier")
    with col2b:
        if example_data:
            away_coach = st.number_input("Away Coach ID", 0, 100_000_000, example_data["away_coach"], disabled=True)
        else:
            away_coach = st.number_input("Away Coach ID", 0, 100_000_000, 67890, help="Encoded coach identifier")

st.caption("ℹ️ Coach IDs are encoded identifiers. We could not match them to actual coach names in our dataset.")

# ======================
# Historical Match Data Input
# ======================
st.markdown("---")
st.markdown("### 📊 Historical Match Data (Last 10 Matches)")

if example_data and "features" in example_data:
    st.info(
        f"📌 Loaded: **{example_data['home_team']} vs {example_data['away_team']}** ({example_data['date'][:10]}) | Actual result: **{['HOME WIN ✅', 'DRAW ➖', 'AWAY WIN ✅'][example_data['actual_outcome']]}**"
    )

    # Display historical matches from real data
    with st.expander("🔍 View Historical Match Details", expanded=False):
        st.markdown("#### 🏠 Home Team Historical Matches")
        for i in range(10):
            match_f = example_data["features"][i]
            scored = round(match_f[2] * GOAL_STD + GOAL_MEAN)
            conceded = round(match_f[3] * GOAL_STD + GOAL_MEAN)
            rating = match_f[4] * RATING_STD + RATING_MEAN
            opp_rating = match_f[5] * RATING_STD + RATING_MEAN
            was_home = bool(match_f[0])
            days = round(match_f[16] * DAYS_STD + DAYS_MEAN)
            st.caption(
                f"Match {i+1}: {'🏠' if was_home else '✈️'} | Scored {scored}, Conceded {conceded} | Rating {rating:.1f} vs {opp_rating:.1f} | {days} days ago"
            )

        st.markdown("#### ✈️ Away Team Historical Matches")
        for i in range(10):
            match_f = example_data["features"][i]
            scored = round(match_f[10] * GOAL_STD + GOAL_MEAN)
            conceded = round(match_f[11] * GOAL_STD + GOAL_MEAN)
            rating = match_f[12] * RATING_STD + RATING_MEAN
            opp_rating = match_f[13] * RATING_STD + RATING_MEAN
            was_home = bool(match_f[8])
            days = round(match_f[17] * DAYS_STD + DAYS_MEAN)
            st.caption(
                f"Match {i+1}: {'🏠' if was_home else '✈️'} | Scored {scored}, Conceded {conceded} | Rating {rating:.1f} vs {opp_rating:.1f} | {days} days ago"
            )

    # Convert real example features from 10x18 to 10x22 by adding aggregated features
    features_ready = []
    for match_features_18 in example_data["features"]:
        # Column order: [0-1] home flags, [2-3] home goals, [4-5] home ratings, [6-7] home meta,
        #              [8-9] away flags, [10-11] away goals, [12-13] away ratings, [14-15] away meta, [16-17] days
        home_goal = match_features_18[2]
        home_opp_goal = match_features_18[3]
        home_rating = match_features_18[4]
        home_opp_rating = match_features_18[5]
        away_goal = match_features_18[10]
        away_opp_goal = match_features_18[11]
        away_rating = match_features_18[12]
        away_opp_rating = match_features_18[13]

        # Calculate 4 aggregated features from SCALED values (matching MyDataset._add_aggregated_features)
        total_goals = home_goal + home_opp_goal + away_goal + away_opp_goal
        goal_diff = home_goal - home_opp_goal
        avg_rating = (home_rating + home_opp_rating + away_rating + away_opp_rating) / 4
        rating_diff = home_rating - home_opp_rating

        match_features_22 = match_features_18 + [total_goals, goal_diff, avg_rating, rating_diff]
        features_ready.append(match_features_22)

else:
    # Custom input mode
    st.markdown("Enter the last 10 matches for each team. **Match 1 = most recent match**")

    # Initialize session state with default match history
    if "home_history" not in st.session_state:
        st.session_state.home_history = [
            {"scored": 2, "conceded": 1, "rating": 7.0, "opp_rating": 6.5, "was_home": True, "days_ago": 7},
            {"scored": 1, "conceded": 1, "rating": 6.8, "opp_rating": 6.8, "was_home": False, "days_ago": 14},
            {"scored": 3, "conceded": 0, "rating": 7.2, "opp_rating": 6.2, "was_home": True, "days_ago": 21},
            {"scored": 1, "conceded": 2, "rating": 6.5, "opp_rating": 7.0, "was_home": False, "days_ago": 28},
            {"scored": 2, "conceded": 2, "rating": 6.9, "opp_rating": 6.9, "was_home": True, "days_ago": 35},
            {"scored": 2, "conceded": 1, "rating": 7.0, "opp_rating": 6.4, "was_home": False, "days_ago": 42},
            {"scored": 1, "conceded": 0, "rating": 7.1, "opp_rating": 6.3, "was_home": True, "days_ago": 49},
            {"scored": 0, "conceded": 1, "rating": 6.7, "opp_rating": 7.1, "was_home": False, "days_ago": 56},
            {"scored": 2, "conceded": 1, "rating": 7.0, "opp_rating": 6.6, "was_home": True, "days_ago": 63},
            {"scored": 1, "conceded": 1, "rating": 6.8, "opp_rating": 6.8, "was_home": False, "days_ago": 70},
        ]
    if "away_history" not in st.session_state:
        st.session_state.away_history = [
            {"scored": 1, "conceded": 2, "rating": 6.5, "opp_rating": 7.0, "was_home": False, "days_ago": 7},
            {"scored": 2, "conceded": 1, "rating": 6.8, "opp_rating": 6.6, "was_home": True, "days_ago": 14},
            {"scored": 0, "conceded": 3, "rating": 6.2, "opp_rating": 7.5, "was_home": False, "days_ago": 21},
            {"scored": 1, "conceded": 1, "rating": 6.6, "opp_rating": 6.6, "was_home": True, "days_ago": 28},
            {"scored": 1, "conceded": 2, "rating": 6.4, "opp_rating": 6.9, "was_home": False, "days_ago": 35},
            {"scored": 2, "conceded": 0, "rating": 6.9, "opp_rating": 6.3, "was_home": True, "days_ago": 42},
            {"scored": 1, "conceded": 1, "rating": 6.5, "opp_rating": 6.7, "was_home": False, "days_ago": 49},
            {"scored": 0, "conceded": 2, "rating": 6.3, "opp_rating": 7.2, "was_home": True, "days_ago": 56},
            {"scored": 1, "conceded": 1, "rating": 6.6, "opp_rating": 6.8, "was_home": False, "days_ago": 63},
            {"scored": 2, "conceded": 1, "rating": 6.7, "opp_rating": 6.5, "was_home": True, "days_ago": 70},
        ]

    # Input UI - show all 10 matches in a table-like format
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏠 Home Team - Last 10 Matches")
        for i in range(10):
            with st.container():
                st.markdown(f"**Match {i+1}**" + (" (most recent)" if i == 0 else ""))
                cols = st.columns([2, 2, 2, 2, 1, 2])
                with cols[0]:
                    st.session_state.home_history[i]["scored"] = st.number_input(
                        "Scored",
                        0,
                        10,
                        st.session_state.home_history[i]["scored"],
                        key=f"h{i}_sc",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )
                with cols[1]:
                    st.session_state.home_history[i]["conceded"] = st.number_input(
                        "Conceded",
                        0,
                        10,
                        st.session_state.home_history[i]["conceded"],
                        key=f"h{i}_co",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )
                with cols[2]:
                    st.session_state.home_history[i]["rating"] = st.number_input(
                        "Rating",
                        0.0,
                        10.0,
                        float(st.session_state.home_history[i]["rating"]),
                        step=0.1,
                        key=f"h{i}_r",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )
                with cols[3]:
                    st.session_state.home_history[i]["opp_rating"] = st.number_input(
                        "Opp Rtg",
                        0.0,
                        10.0,
                        float(st.session_state.home_history[i]["opp_rating"]),
                        step=0.1,
                        key=f"h{i}_or",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )
                with cols[4]:
                    st.session_state.home_history[i]["was_home"] = st.checkbox(
                        "Home?",
                        st.session_state.home_history[i]["was_home"],
                        key=f"h{i}_ih",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )
                with cols[5]:
                    st.session_state.home_history[i]["days_ago"] = st.number_input(
                        "Days Ago",
                        1,
                        365,
                        st.session_state.home_history[i]["days_ago"],
                        key=f"h{i}_d",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )

    with col2:
        st.markdown("#### ✈️ Away Team - Last 10 Matches")
        for i in range(10):
            with st.container():
                st.markdown(f"**Match {i+1}**" + (" (most recent)" if i == 0 else ""))
                cols = st.columns([2, 2, 2, 2, 1, 2])
                with cols[0]:
                    st.session_state.away_history[i]["scored"] = st.number_input(
                        "Scored",
                        0,
                        10,
                        st.session_state.away_history[i]["scored"],
                        key=f"a{i}_sc",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )
                with cols[1]:
                    st.session_state.away_history[i]["conceded"] = st.number_input(
                        "Conceded",
                        0,
                        10,
                        st.session_state.away_history[i]["conceded"],
                        key=f"a{i}_co",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )
                with cols[2]:
                    st.session_state.away_history[i]["rating"] = st.number_input(
                        "Rating",
                        0.0,
                        10.0,
                        float(st.session_state.away_history[i]["rating"]),
                        step=0.1,
                        key=f"a{i}_r",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )
                with cols[3]:
                    st.session_state.away_history[i]["opp_rating"] = st.number_input(
                        "Opp Rtg",
                        0.0,
                        10.0,
                        float(st.session_state.away_history[i]["opp_rating"]),
                        step=0.1,
                        key=f"a{i}_or",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )
                with cols[4]:
                    st.session_state.away_history[i]["was_home"] = st.checkbox(
                        "Home?",
                        st.session_state.away_history[i]["was_home"],
                        key=f"a{i}_ih",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )
                with cols[5]:
                    st.session_state.away_history[i]["days_ago"] = st.number_input(
                        "Days Ago",
                        1,
                        365,
                        st.session_state.away_history[i]["days_ago"],
                        key=f"a{i}_d",
                        label_visibility="collapsed" if i > 0 else "visible",
                    )

    # Convert user inputs to scaled features (10x22 - each match has 18 base + 4 aggregated features)
    features_ready = []
    for i in range(10):
        home_m = st.session_state.home_history[i]
        away_m = st.session_state.away_history[i]

        # Scale base features (18 features per match)
        home_goal_scaled = (home_m["scored"] - GOAL_MEAN) / GOAL_STD
        home_conceded_scaled = (home_m["conceded"] - GOAL_MEAN) / GOAL_STD
        home_rating_scaled = (home_m["rating"] - RATING_MEAN) / RATING_STD
        home_opp_rating_scaled = (home_m["opp_rating"] - RATING_MEAN) / RATING_STD

        away_goal_scaled = (away_m["scored"] - GOAL_MEAN) / GOAL_STD
        away_conceded_scaled = (away_m["conceded"] - GOAL_MEAN) / GOAL_STD
        away_rating_scaled = (away_m["rating"] - RATING_MEAN) / RATING_STD
        away_opp_rating_scaled = (away_m["opp_rating"] - RATING_MEAN) / RATING_STD

        match_features_18 = [
            # Home team features (9 features)
            home_goal_scaled,
            home_conceded_scaled,
            home_rating_scaled,
            home_opp_rating_scaled,
            1.0 if home_m["was_home"] else 0.0,
            0.0,  # is_cup (assume not cup match)
            float(home_coach),
            float(league_encoded),
            (home_m["days_ago"] - DAYS_MEAN) / DAYS_STD,
            # Away team features (9 features)
            away_goal_scaled,
            away_conceded_scaled,
            away_rating_scaled,
            away_opp_rating_scaled,
            1.0 if away_m["was_home"] else 0.0,
            0.0,  # is_cup
            float(away_coach),
            float(league_encoded),
            (away_m["days_ago"] - DAYS_MEAN) / DAYS_STD,
        ]

        # Add 4 aggregated features from SCALED values (matching MyDataset._add_aggregated_features)
        # Training: data is scaled first, THEN aggregated features are calculated from scaled values
        # So aggregated features themselves are NOT re-scaled
        total_goals = home_goal_scaled + home_conceded_scaled + away_goal_scaled + away_conceded_scaled
        goal_diff = home_goal_scaled - home_conceded_scaled
        avg_rating = (home_rating_scaled + home_opp_rating_scaled + away_rating_scaled + away_opp_rating_scaled) / 4
        rating_diff = home_rating_scaled - home_opp_rating_scaled

        match_features_22 = match_features_18 + [total_goals, goal_diff, avg_rating, rating_diff]
        features_ready.append(match_features_22)

# ======================
# Prediction Button
# ======================
st.markdown("---")
if st.button("⚽ Predict Match Outcome", type="primary", use_container_width=True):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features_ready},
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            probabilities = result["probabilities"]
            prediction = result["prediction"]

            # Display only the predicted outcome (simplified)
            if prediction == "home":
                st.success("🏆 **Predicted Outcome: HOME WIN**")
            elif prediction == "draw":
                st.info("🤝 **Predicted Outcome: DRAW**")
            else:
                st.success("🏆 **Predicted Outcome: AWAY WIN**")

            # Compare with actual outcome if example
            if example_data:
                actual = ["HOME WIN", "DRAW", "AWAY WIN"][example_data["actual_outcome"]]
                predicted_clean = prediction.upper().replace("HOME", "HOME WIN").replace("AWAY", "AWAY WIN")
                if predicted_clean == actual:
                    st.success(f"✅ **Correct!** Actual result was also {actual}")
                else:
                    st.error(f"❌ **Incorrect.** Actual result was {actual}")
        else:
            st.error(f"❌ API Error: {response.status_code} - {response.text}")

    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The API might be cold-starting (takes ~30s).")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("🤖 LSTM with Attention | 📊 Trained on 110K+ matches | ☁️ Deployed on Google Cloud Run")
