"""
Submit training job to GCP Vertex AI with GPU support.

GPU Pricing (approximate, europe-west1):
- NVIDIA T4: ~$0.35/hour (good for this task)
- NVIDIA V100: ~$2.50/hour (overkill)
- NVIDIA A100: ~$4.00/hour (overkill)

For this project, T4 is perfect: ~5-10 minutes training = $0.03-0.06 per run
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

# Configuration
PROJECT_ID = "mlops-484822"
REGION = "europe-west1"
IMAGE_NAME = f"gcr.io/{PROJECT_ID}/football-lstm-trainer"
BUCKET_DATA = f"{PROJECT_ID}-data"
BUCKET_MODELS = f"{PROJECT_ID}-models"

# Find gcloud executable
GCLOUD_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "Google", "Cloud SDK", "google-cloud-sdk", "bin", "gcloud.cmd"
)
if not os.path.exists(GCLOUD_PATH):
    GCLOUD_PATH = "gcloud"


def run_command(cmd, description):
    """Run shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")

    if cmd[0] == "gcloud":
        cmd[0] = GCLOUD_PATH

    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", shell=True)

    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    else:
        print(f"✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True


def main():
    """Submit training job with GPU to GCP Vertex AI."""

    print("\n" + "=" * 80)
    print("🚀 FOOTBALL LSTM TRAINING - GPU ACCELERATED")
    print("=" * 80)
    print("\n📊 GPU Configuration:")
    print("   - Type: NVIDIA T4")
    print("   - Cost: ~$0.35/hour")
    print("   - Estimated training time: 5-10 minutes")
    print("   - Estimated cost: $0.03-0.06 per run")
    print("=" * 80)

    # Step 1: Build Docker image
    print("\n" + "=" * 80)
    print("🐳 STEP 1: Building Docker Image")
    print("=" * 80)

    if not run_command(["docker", "build", "-f", "train.dockerfile", "-t", IMAGE_NAME, "."], "Building Docker image"):
        sys.exit(1)

    # Step 2: Push to GCR
    print("\n" + "=" * 80)
    print("📤 STEP 2: Pushing to Google Container Registry")
    print("=" * 80)

    run_command(["gcloud", "auth", "configure-docker"], "Configuring Docker for GCR")

    if not run_command(["docker", "push", IMAGE_NAME], "Pushing image to GCR"):
        sys.exit(1)

    # Step 3: Submit Vertex AI job with GPU
    print("\n" + "=" * 80)
    print("🚀 STEP 3: Submitting Vertex AI Training Job with T4 GPU")
    print("=" * 80)

    job_name = f"football-lstm-gpu-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Get WandB API key from environment
    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        print("⚠️  Warning: WANDB_API_KEY environment variable not set. Training will run without WandB logging.")
        print("   Set it with: $env:WANDB_API_KEY='your-key-here' (PowerShell)")
        print("   Or: export WANDB_API_KEY='your-key-here' (Linux/Mac)")

    # Job config with GPU
    job_config = {
        "workerPoolSpecs": [
            {
                "machineSpec": {
                    "machineType": "n1-standard-4",
                    "acceleratorType": "NVIDIA_TESLA_T4",
                    "acceleratorCount": 1,
                },
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": IMAGE_NAME,
                    "args": ["--config-name=train_gpu"],  # Use GPU config
                },
            }
        ]
    }

    if wandb_key:
        job_config["workerPoolSpecs"][0]["containerSpec"]["env"] = [{"name": "WANDB_API_KEY", "value": wandb_key}]

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(job_config, f, indent=2)
        config_file = f.name

    print(f"\n📄 Job configuration:")
    print(json.dumps(job_config, indent=2))

    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        f"--region={REGION}",
        f"--display-name={job_name}",
        f"--config={config_file}",
    ]

    if not run_command(cmd, "Submitting GPU training job"):
        os.unlink(config_file)
        sys.exit(1)

    os.unlink(config_file)

    print("\n" + "=" * 80)
    print("✅ GPU JOB SUBMITTED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📊 Monitor on WandB: https://wandb.ai/tyranguyen7-danmarks-tekniske-universitet-dtu/football-lstm")
    print(f"\n🔍 View job in GCP Console:")
    print(f"   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"\n📝 Stream logs with:")
    print(f"   gcloud ai custom-jobs stream-logs {job_name} --region={REGION}")
    print("\n💡 New features in this run:")
    print("   ✅ Attention mechanism - focuses on important matches")
    print("   ✅ Focal loss - handles class imbalance better")
    print("   ✅ Aggregated features - team strength indicators")
    print("   ✅ Learning rate scheduler - adapts during training")
    print("   ✅ Larger model (128 hidden units, 3 layers)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
