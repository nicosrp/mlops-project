"""Deploy FastAPI application to Google Cloud Run."""

import subprocess
import sys
from datetime import datetime

PROJECT_ID = "mlops-484822"
REGION = "europe-west1"
SERVICE_NAME = "football-api"
IMAGE_NAME = f"gcr.io/{PROJECT_ID}/{SERVICE_NAME}"


def run_command(cmd: str, description: str) -> None:
    """Run a shell command and handle errors."""
    print("\n" + "=" * 80)
    print(f"🔧 {description}")
    print("=" * 80)
    print(f"Command: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        sys.exit(1)

    print(f"✅ Success!")


def main():
    print("\n" + "=" * 80)
    print("🚀 DEPLOYING FOOTBALL API TO GOOGLE CLOUD RUN")
    print("=" * 80)

    # Step 1: Build Docker image
    run_command(f"docker build -f api.dockerfile -t {IMAGE_NAME}:latest .", "Building Docker image")

    # Step 2: Tag with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_command(
        f"docker tag {IMAGE_NAME}:latest {IMAGE_NAME}:{timestamp}", f"Tagging image with timestamp: {timestamp}"
    )

    # Step 3: Configure Docker for GCR
    run_command("gcloud auth configure-docker", "Configuring Docker for Google Container Registry")

    # Step 4: Push to GCR (both latest and timestamped)
    run_command(f"docker push {IMAGE_NAME}:latest", "Pushing latest image to GCR")

    run_command(f"docker push {IMAGE_NAME}:{timestamp}", "Pushing timestamped image to GCR")

    # Step 5: Deploy to Cloud Run
    deploy_cmd = f"""gcloud run deploy {SERVICE_NAME} \
        --image {IMAGE_NAME}:latest \
        --platform managed \
        --region {REGION} \
        --project {PROJECT_ID} \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 1 \
        --timeout 300 \
        --max-instances 10 \
        --port 8080"""

    run_command(deploy_cmd, "Deploying to Cloud Run")

    # Step 6: Get service URL
    print("\n" + "=" * 80)
    print("✅ DEPLOYMENT SUCCESSFUL!")
    print("=" * 80)

    result = subprocess.run(
        f"gcloud run services describe {SERVICE_NAME} --region {REGION} --project {PROJECT_ID} --format 'value(status.url)'",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        service_url = result.stdout.strip()
        print(f"\n📊 Service URL: {service_url}")
        print(f"\n🔍 Test endpoints:")
        print(f"   Health: {service_url}/health")
        print(f"   Docs: {service_url}/docs")
        print(f"   Predict: {service_url}/predict (POST)")
        print(f"\n📝 View logs:")
        print(f"   gcloud run logs read {SERVICE_NAME} --region {REGION} --project {PROJECT_ID}")
        print(f"\n🌐 Cloud Console:")
        print(f"   https://console.cloud.google.com/run/detail/{REGION}/{SERVICE_NAME}?project={PROJECT_ID}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
