"""Load testing for FastAPI application using Locust."""

import random

from locust import HttpUser, between, task


class FootballAPIUser(HttpUser):
    """Simulated user for load testing the football prediction API."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    @task(3)  # Weight: 3x more likely than other tasks
    def predict_match(self):
        """Test the prediction endpoint with random features."""
        # Generate random features for a match
        features = [
            round(random.uniform(0.5, 3.0), 2),  # home goals avg
            round(random.uniform(0.5, 2.5), 2),  # home conceded avg
            round(random.uniform(0.5, 3.0), 2),  # away goals avg
            round(random.uniform(0.5, 2.5), 2),  # away conceded avg
        ]

        with self.client.post("/predict", json={"features": features}, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "predictions" in data and len(data["predictions"]) == 3:
                    response.success()
                else:
                    response.failure("Invalid prediction format")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def health_check(self):
        """Test the health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def root_endpoint(self):
        """Test the root endpoint."""
        self.client.get("/")

    def on_start(self):
        """Called when a simulated user starts."""
        # Could add authentication or setup here if needed
        pass
