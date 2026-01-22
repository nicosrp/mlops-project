"""Create mapping from raw league_id to encoded history_league_id"""

import json

import pandas as pd

# Load raw and processed data
raw_df = pd.read_csv("data/raw/train.csv", nrows=50000)
processed_df = pd.read_csv("data/processed/processed_data.csv", nrows=50000)

# Get the raw league IDs and their encoded values from the SAME rows
mapping = {}
for idx in range(min(len(raw_df), len(processed_df))):
    raw_league = raw_df.loc[idx, "league_id"]
    encoded_league = processed_df.loc[idx, "home_team_history_league_id_1"]
    if pd.notna(raw_league) and pd.notna(encoded_league):
        mapping[int(raw_league)] = int(encoded_league)

print(f"Found {len(set(mapping.values()))} unique encoded league IDs")
print("\nSample mappings:")
for raw_id, enc_id in list(mapping.items())[:20]:
    league_name = (
        raw_df[raw_df["league_id"] == raw_id]["league_name"].iloc[0]
        if len(raw_df[raw_df["league_id"] == raw_id]) > 0
        else "Unknown"
    )
    print(f"  {raw_id} ({league_name}) → {enc_id}")

with open("league_id_to_encoded.json", "w") as f:
    json.dump(mapping, f, indent=2)

print("\n✅ Saved to league_id_to_encoded.json")
