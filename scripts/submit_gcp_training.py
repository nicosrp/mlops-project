"""
Submit training job to GCP Vertex AI.
This script handles:
- Building and pushing Docker image
- Downloading data from GCS to container
- Running training with WandB logging
- Uploading trained model back to GCS
"""

import os
import subprocess
import sys
from datetime import datetime

# Configuration
PROJECT_ID = "mlops-484822"  # Replace with your project ID
REGION = "europe-west1"
IMAGE_NAME = f"gcr.io/{PROJECT_ID}/football-lstm-trainer"
BUCKET_DATA = f"{PROJECT_ID}-data"
BUCKET_MODELS = f"{PROJECT_ID}-models"

# Find gcloud executable
GCLOUD_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "Google", "Cloud SDK", "google-cloud-sdk", "bin", "gcloud.cmd"
)
if not os.path.exists(GCLOUD_PATH):
    GCLOUD_PATH = "gcloud"  # Fallback to PATH


def run_command(cmd, description):
    """Run shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")

    # Replace 'gcloud' with full path
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
    """Submit training job to GCP Vertex AI."""

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

    # Configure Docker for GCR
    run_command(["gcloud", "auth", "configure-docker"], "Configuring Docker for GCR")

    if not run_command(["docker", "push", IMAGE_NAME], "Pushing image to GCR"):
        sys.exit(1)

    # Step 3: Submit Vertex AI job
    print("\n" + "=" * 80)
    print("🚀 STEP 3: Submitting Vertex AI Training Job")
    print("=" * 80)

    job_name = f"football-lstm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Get WandB API key
    import os

    # Use environment variable or fallback to hardcoded key
    wandb_key = os.getenv("WANDB_API_KEY") or "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"

    # Create job config with environment variables  
    # Note: config file should NOT include displayName - that's passed as a flag
    job_config = {
        "workerPoolSpecs": [
            {
                "machineSpec": {
                    "machineType": "n1-standard-4"
                },
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": IMAGE_NAME,
                    "args": ["--config-name=train"]
                }
            }
        ]
    }
    
    # Add environment variables if WandB key exists
    if wandb_key:
        job_config["workerPoolSpecs"][0]["containerSpec"]["env"] = [
            {"name": "WANDB_API_KEY", "value": wandb_key}
        ]
    
    # Write config to temp file
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(job_config, f)
        config_file = f.name
    
    # Build command using config file
    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        f"--region={REGION}",
        f"--display-name={job_name}",
        f"--config={config_file}",
    ]

    if not run_command(cmd, "Submitting training job"):
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ JOB SUBMITTED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📊 Monitor on WandB: https://wandb.ai/tyranguyen7-danmarks-tekniske-universitet-dtu/football-lstm")
    print(f"\n🔍 View job in GCP Console:")
    print(f"   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"\n📝 Stream logs with:")
    print(f"   gcloud ai custom-jobs stream-logs {job_name} --region={REGION}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
