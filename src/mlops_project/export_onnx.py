"""Export PyTorch model to ONNX format for optimized inference."""

import torch

from mlops_project.model import Model


def export_to_onnx():
    """Export model to ONNX format."""
    print("🔧 Loading PyTorch model...")

    # Load checkpoint to get model parameters
    checkpoint = torch.load("models/best_model.pth", map_location="cpu", weights_only=True)

    # Extract model parameters from checkpoint
    input_size = checkpoint.get("input_size", 22)
    hidden_size = checkpoint.get("hidden_size", 64)
    num_layers = checkpoint.get("num_layers", 2)
    dropout = checkpoint.get("dropout", 0.3)

    print(f"Model config: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")

    model = Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=3, dropout=dropout)

    # Load trained weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("✅ Model loaded successfully")

    # Create dummy input (batch_size=1, seq_len=10, features=22)
    dummy_input = torch.randn(1, 10, 22)

    print("📦 Exporting to ONNX format...")

    # Export to ONNX
    torch.onnx.export(
        model,  # model
        dummy_input,  # model input
        "models/best_model.onnx",  # where to save
        export_params=True,  # store trained weights
        opset_version=14,  # ONNX version
        do_constant_folding=True,  # optimize constant folding
        input_names=["input"],  # input names
        output_names=["output"],  # output names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # dynamic batch size
    )

    print("✅ Model exported to models/best_model.onnx")

    # Test ONNX model
    print("\n🧪 Testing ONNX model...")
    import onnxruntime as ort

    ort_session = ort.InferenceSession("models/best_model.onnx")

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Compare with PyTorch
    with torch.no_grad():
        torch_outputs = model(dummy_input)

    diff = torch.abs(torch_outputs - torch.tensor(ort_outputs[0])).max()
    print(f"✅ Max difference between PyTorch and ONNX: {diff:.6f}")

    if diff < 1e-5:
        print("✅ ONNX export successful - outputs match!")
    else:
        print("⚠️  Warning: ONNX outputs differ from PyTorch")

    # Benchmark
    print("\n⚡ Benchmarking inference speed...")
    import time

    # PyTorch
    torch_times = []
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        torch_times.append(time.time() - start)

    # ONNX
    onnx_times = []
    for _ in range(100):
        start = time.time()
        _ = ort_session.run(None, ort_inputs)
        onnx_times.append(time.time() - start)

    torch_avg = sum(torch_times) / len(torch_times) * 1000
    onnx_avg = sum(onnx_times) / len(onnx_times) * 1000
    speedup = torch_avg / onnx_avg

    print(f"PyTorch: {torch_avg:.2f}ms per inference")
    print(f"ONNX:    {onnx_avg:.2f}ms per inference")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    export_to_onnx()
