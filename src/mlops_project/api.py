import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# Try to import ONNX Runtime, fall back to PyTorch if not available
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    # Fallback to PyTorch model for testing/development
    from mlops_project.model import Model

app = FastAPI(title="Football Match Prediction API", version="1.0.0")

# Prometheus metrics
prediction_counter = Counter("predictions_total", "Total number of predictions", ["outcome"])
prediction_probabilities = Histogram("prediction_probability", "Distribution of prediction probabilities", ["outcome"])
inference_duration = Histogram("model_inference_duration_seconds", "Time taken for model inference")
model_loaded_gauge = Gauge("model_loaded", "Whether the model is loaded (1) or not (0)")
prediction_errors = Counter("prediction_errors_total", "Total number of prediction errors", ["error_type"])

# Initialize Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


class PredictionInput(BaseModel):
    """Input features for match prediction."""

    features: list[list[float]] = Field(
        ..., description="Historical match features as a list of sequences. Shape: (seq_len, feature_dim)"
    )


class PredictionOutput(BaseModel):
    """Prediction output with class probabilities."""

    prediction: str = Field(..., description="Predicted outcome: home, draw, or away")
    probabilities: dict[str, float] = Field(..., description="Class probabilities")


# Global model instance (ONNX or PyTorch depending on availability)
onnx_session = None
pytorch_model = None
label_map = {0: "home", 1: "draw", 2: "away"}

# Database file for logging predictions
PREDICTION_DB = Path("data/prediction_database.csv")


def log_prediction_to_db(timestamp: str, features: list[list[float]], prediction: str, probabilities: dict) -> None:
    """
    Background task to log predictions to CSV database.

    Args:
        timestamp: Timestamp of the prediction
        features: Input features (flattened for storage)
        prediction: Predicted class
        probabilities: Prediction probabilities
    """
    try:
        # Create data directory if it doesn't exist
        PREDICTION_DB.parent.mkdir(parents=True, exist_ok=True)

        # Flatten features for storage (take mean across sequence)
        features_tensor = torch.tensor(features)
        feature_means = features_tensor.mean(dim=0).tolist()

        # Create row data
        row_data = {
            "time": timestamp,
            "prediction": prediction,
            "prob_home": probabilities["home"],
            "prob_draw": probabilities["draw"],
            "prob_away": probabilities["away"],
        }

        # Add feature columns (assuming 22 features)
        for i, val in enumerate(feature_means):
            row_data[f"feature_{i}"] = val

        # Append to CSV
        df = pd.DataFrame([row_data])
        if PREDICTION_DB.exists():
            df.to_csv(PREDICTION_DB, mode="a", header=False, index=False)
        else:
            df.to_csv(PREDICTION_DB, mode="w", header=True, index=False)

    except Exception as e:
        print(f"Error logging prediction: {e}")


@app.on_event("startup")
async def load_model() -> None:
    """Load the model on application startup (ONNX or PyTorch fallback)."""
    global onnx_session, pytorch_model

    # Try ONNX first
    if ONNX_AVAILABLE:
        onnx_path = Path("models/best_model.onnx")
        if onnx_path.exists():
            try:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                onnx_session = ort.InferenceSession(
                    str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"]
                )
                model_loaded_gauge.set(1)
                print(f"ONNX model loaded from {onnx_path}")
                return
            except Exception as e:
                print(f"Error loading ONNX model: {e}")

    # Fallback to PyTorch
    pytorch_path = Path("models/best_model.pth")
    if pytorch_path.exists():
        try:
            checkpoint = torch.load(pytorch_path, map_location=torch.device("cpu"))
            pytorch_model = Model(
                input_size=checkpoint.get("input_size", 22),
                hidden_size=checkpoint.get("hidden_size", 64),
                num_layers=checkpoint.get("num_layers", 2),
                output_size=3,
                dropout=checkpoint.get("dropout", 0.3),
                use_attention=checkpoint.get("use_attention", True),
            )
            pytorch_model.load_state_dict(checkpoint["model_state_dict"])
            pytorch_model.eval()
            model_loaded_gauge.set(1)
            print(f"PyTorch model loaded from {pytorch_path}")
        except Exception as e:
            model_loaded_gauge.set(0)
            print(f"Error loading PyTorch model: {e}")
    else:
        model_loaded_gauge.set(0)
        print("No model file found")


@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "message": "Football Match Prediction API"}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Detailed health check with model status."""
    model_type = "ONNX" if onnx_session is not None else ("PyTorch" if pytorch_model is not None else "None")
    return {
        "status": "ok",
        "model_loaded": (onnx_session is not None or pytorch_model is not None),
        "model_type": model_type,
        "api_version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput, background_tasks: BackgroundTasks) -> PredictionOutput:
    """
    Predict match outcome based on historical match features.

    Args:
        input_data: Historical match features (seq_len, feature_dim)
        background_tasks: FastAPI background tasks for logging

    Returns:
        Predicted outcome and class probabilities
    """
    if onnx_session is None and pytorch_model is None:
        prediction_errors.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Start timing
        start_time = time.time()

        # Use ONNX if available, otherwise PyTorch
        if onnx_session is not None:
            # ONNX inference
            features_array = np.array(input_data.features, dtype=np.float32)
            features_array = np.expand_dims(features_array, axis=0)
            input_name = onnx_session.get_inputs()[0].name
            logits = onnx_session.run(None, {input_name: features_array})[0]
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probabilities = probabilities[0]
        else:
            # PyTorch inference
            features_tensor = torch.tensor(input_data.features, dtype=torch.float32)
            features_tensor = features_tensor.unsqueeze(0)
            with torch.no_grad():
                logits = pytorch_model(features_tensor)
                probabilities = torch.softmax(logits, dim=1)[0].numpy()

        # Record inference time
        inference_duration.observe(time.time() - start_time)

        # Get predicted class
        predicted_class = int(np.argmax(probabilities))
        predicted_label = label_map[predicted_class]

        # Convert probabilities to dict
        prob_dict = {label_map[i]: float(probabilities[i]) for i in range(len(probabilities))}

        # Update metrics
        prediction_counter.labels(outcome=predicted_label).inc()
        for outcome, prob in prob_dict.items():
            prediction_probabilities.labels(outcome=outcome).observe(prob)

        # Log prediction to database (background task)
        timestamp = datetime.now().isoformat()
        background_tasks.add_task(log_prediction_to_db, timestamp, input_data.features, predicted_label, prob_dict)

        return PredictionOutput(prediction=predicted_label, probabilities=prob_dict)

    except ValueError as e:
        prediction_errors.labels(error_type="invalid_input").inc()
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        prediction_errors.labels(error_type="unknown").inc()
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring(n_samples: int = 100) -> str:
    """
    Generate and return data drift monitoring report.

    Args:
        n_samples: Number of latest predictions to analyze

    Returns:
        HTML report showing data drift analysis
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
        from evidently.report import Report

        # Load prediction database
        if not PREDICTION_DB.exists():
            return "<html><body><h1>No prediction data available</h1><p>Make some predictions first.</p></body></html>"

        current_data = pd.read_csv(PREDICTION_DB)

        if len(current_data) == 0:
            return "<html><body><h1>No prediction data available</h1><p>Database is empty.</p></body></html>"

        # Get last n samples
        current_data = current_data.tail(n_samples)

        # Load reference data (you'll need to provide this)
        # For now, we'll use the first half of data as reference if enough data exists
        if len(current_data) < 20:
            return f"<html><body><h1>Insufficient data</h1><p>Need at least 20 predictions. Current: {len(current_data)}</p></body></html>"

        # Split data for demo purposes (in production, use actual training data)
        split_idx = len(current_data) // 2
        reference_data = current_data.iloc[:split_idx].copy()
        current_data = current_data.iloc[split_idx:].copy()

        # Standardize data
        if "time" in current_data.columns:
            current_data = current_data.drop(columns=["time"])
            reference_data = reference_data.drop(columns=["time"])

        if "prediction" in current_data.columns:
            label_map_reverse = {"home": 0, "draw": 1, "away": 2}
            current_data["target"] = current_data["prediction"].map(label_map_reverse)
            reference_data["target"] = reference_data["prediction"].map(label_map_reverse)
            current_data = current_data.drop(columns=["prediction"])
            reference_data = reference_data.drop(columns=["prediction"])

        # Remove probability columns
        prob_cols = [col for col in current_data.columns if col.startswith("prob_")]
        if prob_cols:
            current_data = current_data.drop(columns=prob_cols)
            reference_data = reference_data.drop(columns=prob_cols)

        # Define column mapping
        column_mapping = ColumnMapping(target="target")

        # Generate report
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])

        report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

        # Return HTML
        return report.get_html()

    except Exception as e:
        return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
