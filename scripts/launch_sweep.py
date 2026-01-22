"""
Launch a WandB hyperparameter sweep for the football LSTM model.

This script:
1. Creates a sweep on WandB using the configuration in configs/sweep.yaml
2. Prints the sweep ID and URL
3. Shows the command to run agents

To run the sweep:
1. Execute this script: python scripts/launch_sweep.py
2. Copy the command shown and run it (or run multiple agents in parallel)
"""

from pathlib import Path

import yaml

import wandb

# Load sweep configuration
config_path = Path(__file__).parent.parent / "configs" / "sweep.yaml"
with open(config_path, "r") as f:
    sweep_config = yaml.safe_load(f)

# Load wandb settings from train config
train_config_path = Path(__file__).parent.parent / "configs" / "train.yaml"
with open(train_config_path, "r") as f:
    train_config = yaml.safe_load(f)

project = train_config["wandb"]["project"]
entity = train_config["wandb"]["entity"]

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)

print("=" * 80)
print("✅ Sweep created successfully!")
print("=" * 80)
print(f"\n📊 Sweep ID: {sweep_id}")
print(f"🔗 Sweep URL: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
print("\n" + "=" * 80)
print("To run the sweep agent, execute:")
print("=" * 80)
print(f"\n  wandb agent {entity}/{project}/{sweep_id}\n")
print("Or run from the project directory:")
print(f"\n  cd <project-dir>")
print(f'  $env:PYTHONPATH="src"')
print(f"  wandb agent {entity}/{project}/{sweep_id}\n")
print("=" * 80)
print("💡 Tip: You can run multiple agents in parallel for faster sweeps")
print("=" * 80)
