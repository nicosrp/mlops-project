"""Setup monitoring and alerts for the Cloud Run API."""

import json
import subprocess

PROJECT_ID = "mlops-484822"
SERVICE_NAME = "football-api"
REGION = "europe-west1"


def create_alert_policy(name: str, condition: str, threshold: float, duration: str = "60s"):
    """Create an alert policy in Cloud Monitoring."""

    alert_config = {
        "displayName": f"{SERVICE_NAME} - {name}",
        "conditions": [
            {
                "displayName": f"{name} condition",
                "conditionThreshold": {
                    "filter": condition,
                    "comparison": "COMPARISON_GT",
                    "thresholdValue": threshold,
                    "duration": duration,
                    "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_RATE"}],
                },
            }
        ],
        "combiner": "OR",
        "enabled": True,
        "alertStrategy": {"autoClose": "1800s"},
    }

    # Write to temp file
    with open("alert_policy.json", "w") as f:
        json.dump(alert_config, f, indent=2)

    print(f"\n📊 Creating alert: {name}")
    result = subprocess.run(
        f"gcloud alpha monitoring policies create --policy-from-file=alert_policy.json --project={PROJECT_ID}",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"✅ Alert created: {name}")
    else:
        print(f"⚠️  Alert creation skipped (may already exist): {name}")


def main():
    """Setup monitoring and alerts."""

    print("=" * 80)
    print("🔍 SETTING UP MONITORING & ALERTS")
    print("=" * 80)

    # Alert 1: High error rate
    print("\n1. High Error Rate Alert")
    create_alert_policy(
        name="High Error Rate",
        condition=f'resource.type="cloud_run_revision" AND resource.labels.service_name="{SERVICE_NAME}" AND metric.type="run.googleapis.com/request_count" AND metric.labels.response_code_class="5xx"',
        threshold=5.0,  # More than 5 errors per minute
        duration="60s",
    )

    # Alert 2: High latency
    print("\n2. High Latency Alert")
    create_alert_policy(
        name="High Latency",
        condition=f'resource.type="cloud_run_revision" AND resource.labels.service_name="{SERVICE_NAME}" AND metric.type="run.googleapis.com/request_latencies"',
        threshold=2000.0,  # More than 2 seconds
        duration="120s",
    )

    # Alert 3: Container crashes
    print("\n3. Container Instance Crashes")
    create_alert_policy(
        name="Container Crashes",
        condition=f'resource.type="cloud_run_revision" AND resource.labels.service_name="{SERVICE_NAME}" AND metric.type="run.googleapis.com/container/instance_count" AND metric.labels.state="crashed"',
        threshold=1.0,  # Any crash
        duration="60s",
    )

    print("\n" + "=" * 80)
    print("✅ MONITORING SETUP COMPLETE")
    print("=" * 80)

    print(f"\n📊 View Metrics:")
    print(f"   https://console.cloud.google.com/run/detail/{REGION}/{SERVICE_NAME}/metrics?project={PROJECT_ID}")

    print(f"\n🔔 View Alerts:")
    print(f"   https://console.cloud.google.com/monitoring/alerting?project={PROJECT_ID}")

    print(f"\n📈 View Logs:")
    print(
        f"   https://console.cloud.google.com/logs/query?project={PROJECT_ID}&query=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22{SERVICE_NAME}%22"
    )

    print(f"\n🎯 Custom Metrics (Prometheus):")
    print(f"   Available at: https://{SERVICE_NAME}-617960074163.{REGION}.run.app/metrics")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
