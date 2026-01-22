"""
Automated hyperparameter sweep runner.
Runs multiple training experiments with different hyperparameter combinations.
"""

import random
import subprocess
import sys
from pathlib import Path

# Get the current Python executable (from venv)
PYTHON_EXE = sys.executable
print(f"Using Python: {PYTHON_EXE}")

# Define hyperparameter combinations to try
configurations = [
    # Config 1: Small model, low learning rate
    {"batch_size": 32, "lr": 0.0005, "hidden_size": 32, "num_layers": 1, "dropout": 0.2, "seq_len": 10},
    # Config 2: Medium model, medium learning rate
    {"batch_size": 16, "lr": 0.001, "hidden_size": 64, "num_layers": 2, "dropout": 0.3, "seq_len": 10},
    # Config 3: Large model, higher learning rate
    {"batch_size": 32, "lr": 0.002, "hidden_size": 128, "num_layers": 2, "dropout": 0.3, "seq_len": 10},
    # Config 4: Deep model, low learning rate
    {"batch_size": 16, "lr": 0.0005, "hidden_size": 64, "num_layers": 3, "dropout": 0.4, "seq_len": 10},
    # Config 5: Different sequence length
    {"batch_size": 32, "lr": 0.001, "hidden_size": 64, "num_layers": 2, "dropout": 0.25, "seq_len": 15},
]

EPOCHS = 10
START_FROM_CONFIG = 1  # Run all configs (change to resume from specific config)

print("=" * 80)
print(f"HYPERPARAMETER SWEEP - Running configurations {START_FROM_CONFIG}-{len(configurations)}")
print(f"Each run: {EPOCHS} epochs on full dataset")
print("=" * 80)

for i, config in enumerate(configurations[START_FROM_CONFIG - 1 :], START_FROM_CONFIG):
    print(f"\n{'='*80}")
    print(f"RUN {i}/{len(configurations)}")
    print(f"Configuration: {config}")
    print("=" * 80)

    # Build command
    cmd = [
        PYTHON_EXE,  # Use the venv Python
        "src/mlops_project/train.py",
        f'hyperparameters.batch_size={config["batch_size"]}',
        f'hyperparameters.lr={config["lr"]}',
        f'hyperparameters.hidden_size={config["hidden_size"]}',
        f'hyperparameters.num_layers={config["num_layers"]}',
        f'hyperparameters.dropout={config["dropout"]}',
        f'hyperparameters.seq_len={config["seq_len"]}',
        f"hyperparameters.epochs={EPOCHS}",
    ]

    try:
        result = subprocess.run(cmd, check=True, cwd=Path.cwd())
        print(f"\n✅ Run {i} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Run {i} failed with error: {e}")
        print("Continuing with next configuration...")

    print(f"\nCompleted {i}/{len(configurations)} runs")

print("\n" + "=" * 80)
print("🎉 ALL SWEEP RUNS COMPLETED!")
print("=" * 80)
print("\n📊 View results at:")
print("https://wandb.ai/tyranguyen7-danmarks-tekniske-universitet-dtu/football-lstm")
print("\n💡 Compare runs to find the best hyperparameters!")
print("=" * 80)
