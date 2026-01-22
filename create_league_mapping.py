"""Create league_id to league_name mapping from raw data"""

import json

import pandas as pd

print("Loading raw data...")
df = pd.read_csv("data/raw/train.csv", nrows=50000)

# Create mapping from league_id to league_name
mapping = df[["league_id", "league_name"]].drop_duplicates().dropna()
mapping = mapping.set_index("league_id")["league_name"].to_dict()

# Sort by league name
mapping_sorted = dict(sorted(mapping.items(), key=lambda x: x[1]))

print(f"Found {len(mapping_sorted)} unique leagues")
print("\nSample leagues:")
for league_id, league_name in list(mapping_sorted.items())[:10]:
    print(f"  {league_id}: {league_name}")

# Save to JSON
with open("league_mapping.json", "w") as f:
    json.dump(mapping_sorted, f, indent=2)

print(f"\n✅ Saved to league_mapping.json")
