"""Extract real example games with their actual feature vectors from validation data."""

import json
from collections import Counter

import numpy as np
import pandas as pd

# Load processed data
df = pd.read_csv("data/processed/processed_data.csv", low_memory=False)

# Skip first 1000 rows to get more diverse examples
df = df[1000:].copy()

# Big name teams to prioritize
big_teams = [
    "Barcelona",
    "Real Madrid",
    "Atletico Madrid",
    "Bayern Munich",
    "Borussia Dortmund",
    "RB Leipzig",
    "Manchester United",
    "Manchester City",
    "Liverpool",
    "Chelsea",
    "Arsenal",
    "Tottenham",
    "Juventus",
    "AC Milan",
    "Inter Milan",
    "Napoli",
    "Roma",
    "PSG",
    "Lyon",
    "Marseille",
    "Monaco",
    "Ajax",
    "PSV",
]

# Create a filter for big team matches
df["has_big_team"] = df["home_team_name"].isin(big_teams) | df["away_team_name"].isin(big_teams)

# Check actual distribution in validation data
print("Validation data distribution:")
dist = df["target"].value_counts(normalize=True)
print(dist)
print()

# Get examples - stratified by actual distribution (prioritize big teams)
# Use realistic counts: ~7 home wins, ~4 draws, ~4 away wins (matching home advantage)
sample_counts = {
    "home": 7,  # Most common - home advantage
    "draw": 4,  # Moderate
    "away": 4,  # Less common
}

examples_by_outcome = []
for target, count in sample_counts.items():
    # First try to get big team matches
    big_team_subset = df[(df["target"] == target) & df["has_big_team"]]
    if len(big_team_subset) >= count:
        subset = big_team_subset.head(count)
    else:
        # Fall back to any matches
        subset = df[df["target"] == target].head(count)
    examples_by_outcome.append(subset)

combined = pd.concat(examples_by_outcome)
print(f"Found {len(combined)} examples")
print(f"Big team matches: {combined['has_big_team'].sum()}")

examples = []

for idx, row in combined.iterrows():
    # Extract the actual feature vector for all 10 historical matches
    # Note: aggregated features (total_goals, goal_diff, avg_rating, rating_diff) are calculated at runtime by MyDataset
    # So we only extract the 18 base features per match
    features = []
    for i in range(1, 11):  # matches 1-10 (1 is most recent)
        # IMPORTANT: Column order MUST match the actual CSV column order!
        # The CSV has columns alphabetically sorted within each category
        match_features = [
            float(row[f"home_team_history_is_play_home_{i}"]),
            float(row[f"home_team_history_is_cup_{i}"]),
            float(row[f"home_team_history_goal_{i}"]),
            float(row[f"home_team_history_opponent_goal_{i}"]),
            float(row[f"home_team_history_rating_{i}"]),
            float(row[f"home_team_history_opponent_rating_{i}"]),
            float(row[f"home_team_history_coach_{i}"]),
            float(row[f"home_team_history_league_id_{i}"]),
            float(row[f"away_team_history_is_play_home_{i}"]),
            float(row[f"away_team_history_is_cup_{i}"]),
            float(row[f"away_team_history_goal_{i}"]),
            float(row[f"away_team_history_opponent_goal_{i}"]),
            float(row[f"away_team_history_rating_{i}"]),
            float(row[f"away_team_history_opponent_rating_{i}"]),
            float(row[f"away_team_history_coach_{i}"]),
            float(row[f"away_team_history_league_id_{i}"]),
            float(row[f"home_team_history_match_days_since_{i}"]),
            float(row[f"away_team_history_match_days_since_{i}"]),
        ]
        features.append(match_features)

    # Get basic info (unscaled values from most recent match)
    outcome_map = {"home": 0, "draw": 1, "away": 2}
    example = {
        "name": f"{row['home_team_name']} vs {row['away_team_name']} ({row['match_date'][:4]})",
        "home_team": row["home_team_name"],
        "away_team": row["away_team_name"],
        "date": row["match_date"],
        "league_id": int(row["league_id"]),
        "home_coach": int(row["home_team_coach_id"]),
        "away_coach": int(row["away_team_coach_id"]),
        "actual_outcome": outcome_map[row["target"]],  # 0=home, 1=draw, 2=away
        "features": features,  # The actual 10x22 feature matrix
    }
    examples.append(example)

# Save to JSON
with open("example_games_real.json", "w") as f:
    json.dump(examples, f, indent=2)

print(f"✅ Saved {len(examples)} real example games with actual feature vectors")
print("\nExamples by outcome (reflecting home advantage):")

outcomes = [e["actual_outcome"] for e in examples]
print(f"  - Home wins (0): {outcomes.count(0)}")
print(f"  - Draws (1): {outcomes.count(1)}")
print(f"  - Away wins (2): {outcomes.count(2)}")
print("\nEach example includes the real 10x18 base feature matrix from the dataset")
print("(Aggregated features like total_goals, goal_diff, avg_rating, rating_diff are calculated at runtime)")
