"""Create a small balanced sample for quick experimentation."""

import pandas as pd
from pathlib import Path

# Load full data
data_path = Path("data/processed/processed_data.csv")
df = pd.read_csv(data_path)

print(f"Original data size: {len(df)}")
print(f"Original distribution:\n{df['target'].value_counts()}")
print(f"Original distribution %:\n{df['target'].value_counts(normalize=True)}")

# Sample 1000 from each class (3000 total, perfectly balanced)
sample_size_per_class = 1000

balanced_sample = pd.concat([
    df[df['target'] == 'home'].sample(n=sample_size_per_class, random_state=42),
    df[df['target'] == 'draw'].sample(n=sample_size_per_class, random_state=42),
    df[df['target'] == 'away'].sample(n=sample_size_per_class, random_state=42),
])

# Shuffle
balanced_sample = balanced_sample.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nBalanced sample size: {len(balanced_sample)}")
print(f"Balanced distribution:\n{balanced_sample['target'].value_counts()}")
print(f"Balanced distribution %:\n{balanced_sample['target'].value_counts(normalize=True)}")

# Save
output_path = Path("data/processed/balanced_sample.csv")
balanced_sample.to_csv(output_path, index=False)
print(f"\n✅ Saved to {output_path}")
