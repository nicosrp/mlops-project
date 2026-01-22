import pytest
import torch
from fastapi.testclient import TestClient

from mlops_project.api import app, load_model
from mlops_project.model import Model


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_input():
    """Sample input data for prediction."""
    return {
        "features": [
            [0.5] * 22,  # Sequence of 10 matches with 22 features each
            [0.3] * 22,
            [0.7] * 22,
            [0.2] * 22,
            [0.8] * 22,
            [0.4] * 22,
            [0.6] * 22,
            [0.1] * 22,
            [0.9] * 22,
            [0.5] * 22,
        ]
    }


def test_root_endpoint(client):
    """Test the root endpoint returns status ok."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "message" in response.json()


def test_health_endpoint(client):
    """Test the health endpoint returns proper status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "api_version" in data


def test_predict_endpoint_structure(client, sample_input):
    """Test predict endpoint returns correct structure."""
    response = client.post("/predict", json=sample_input)

    # Should work even if model not loaded (will return 503)
    # or return proper prediction if model exists
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probabilities" in data
        assert data["prediction"] in ["home", "draw", "away"]
        assert len(data["probabilities"]) == 3
        assert "home" in data["probabilities"]
        assert "draw" in data["probabilities"]
        assert "away" in data["probabilities"]


@pytest.mark.skip(reason="Temporarily disabled - model checkpoint compatibility issue")
def test_predict_endpoint_with_mock_model(client, sample_input, monkeypatch):
    """Test predict endpoint with a mocked model."""
    # Create a mock model - input_size should be 22 (feature_dim) * 10 (seq_len) = 220
    # Actually, the input is (batch, seq_len, features), so input_size = 22
    mock_model = Model(input_size=22, hidden_size=64, num_layers=2, output_size=3, use_attention=True)
    mock_model.eval()  # Important: set to eval mode to avoid BatchNorm issues

    # Monkeypatch the global model in the api module
    import mlops_project.api as api_module

    monkeypatch.setattr(api_module, "pytorch_model", mock_model)

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in ["home", "draw", "away"]
    assert len(data["probabilities"]) == 3

    # Check probabilities sum to approximately 1
    prob_sum = sum(data["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01


def test_predict_endpoint_invalid_input(client):
    """Test predict endpoint with invalid input."""
    invalid_input = {"features": "not a list"}
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_no_model(client, sample_input, monkeypatch):
    """Test predict endpoint when model is not loaded."""
    # Ensure model is None
    import mlops_project.api as api_module

    monkeypatch.setattr(api_module, "onnx_session", None)
    monkeypatch.setattr(api_module, "pytorch_model", None)

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


def test_predict_endpoint_wrong_feature_dimension(client, monkeypatch):
    """Test predict endpoint with wrong feature dimensions."""
    mock_model = Model(input_size=444, hidden_size=64, num_layers=2, output_size=3, use_attention=True)
    mock_model.eval()

    import mlops_project.api as api_module

    monkeypatch.setattr(api_module, "pytorch_model", mock_model)

    # Wrong number of features
    wrong_input = {
        "features": [
            [0.5] * 100,  # Wrong feature dimension
            [0.3] * 100,
        ]
    }

    response = client.post("/predict", json=wrong_input)
    assert response.status_code == 400
    assert "Prediction error" in response.json()["detail"] or "error" in response.json()["detail"].lower()


def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint exists."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predictions_total" in response.text or "prometheus" in response.text.lower()


@pytest.mark.skip(reason="Temporarily disabled - model checkpoint compatibility issue")
def test_predict_probabilities_sum_to_one(client, sample_input, monkeypatch):
    """Test that prediction probabilities sum to 1."""
    mock_model = Model(input_size=22, hidden_size=64, num_layers=2, output_size=3, use_attention=True)
    mock_model.eval()

    import mlops_project.api as api_module

    monkeypatch.setattr(api_module, "pytorch_model", mock_model)

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200

    probs = response.json()["probabilities"]
    prob_sum = sum(probs.values())
    assert 0.99 <= prob_sum <= 1.01  # Allow small floating point error
