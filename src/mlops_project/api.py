from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops_project.model import Model

app = FastAPI(title="Football Match Prediction API", version="1.0.0")


class PredictionInput(BaseModel):
    """Input features for match prediction."""

    features: list[list[float]] = Field(
        ..., description="Historical match features as a list of sequences. Shape: (seq_len, feature_dim)"
    )


class PredictionOutput(BaseModel):
    """Prediction output with class probabilities."""

    prediction: str = Field(..., description="Predicted outcome: home, draw, or away")
    probabilities: dict[str, float] = Field(..., description="Class probabilities")


# Global model instance (loaded on startup)
model: Model | None = None
label_map = {0: "home", 1: "draw", 2: "away"}


@app.on_event("startup")
async def load_model() -> None:
    """Load the trained model on application startup."""
    global model
    model_path = Path("models/best_model.pth")

    if not model_path.exists():
        print(f"Warning: Model file not found at {model_path}")
        return

    try:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        input_size = checkpoint.get("input_size", 444)  # Default from training

        model = Model(
            input_size=input_size,
            hidden_size=checkpoint.get("hidden_size", 64),
            num_layers=checkpoint.get("num_layers", 2),
            output_size=3,
            dropout=checkpoint.get("dropout", 0.3),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "message": "Football Match Prediction API"}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Detailed health check with model status."""
    return {"status": "ok", "model_loaded": model is not None, "api_version": "1.0.0"}


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput) -> PredictionOutput:
    """
    Predict match outcome based on historical match features.

    Args:
        input_data: Historical match features (seq_len, feature_dim)

    Returns:
        Predicted outcome and class probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to tensor
        features_tensor = torch.tensor(input_data.features, dtype=torch.float32)

        # Add batch dimension: (1, seq_len, feature_dim)
        features_tensor = features_tensor.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            logits = model(features_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        # Get predicted class
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = label_map[predicted_class]

        # Convert probabilities to dict
        prob_dict = {label_map[i]: float(probabilities[i]) for i in range(len(probabilities))}

        return PredictionOutput(prediction=predicted_label, probabilities=prob_dict)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
