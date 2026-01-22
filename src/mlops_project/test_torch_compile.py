"""Test torch.compile for inference speedup (PyTorch 2.0+)."""

import time

import torch

from mlops_project.model import Model


def test_torch_compile():
    """Compare regular model vs compiled model inference speed."""
    print("🔧 Loading model...")

    # Load checkpoint
    checkpoint = torch.load("models/best_model.pth", map_location="cpu", weights_only=True)

    # Get model config
    input_size = checkpoint.get("input_size", 22)
    hidden_size = checkpoint.get("hidden_size", 64)
    num_layers = checkpoint.get("num_layers", 2)
    dropout = checkpoint.get("dropout", 0.3)

    # Create model
    model = Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=3, dropout=dropout)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✅ Model loaded successfully")

    # Create dummy input
    dummy_input = torch.randn(1, 10, 22)

    # Warmup
    print("\n🔥 Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark regular model
    print("\n⚡ Benchmarking regular model...")
    regular_times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(dummy_input)
            regular_times.append(time.time() - start)

    regular_avg = sum(regular_times) / len(regular_times) * 1000

    # Try torch.compile (PyTorch 2.0+)
    try:
        print("\n📦 Compiling model with torch.compile...")
        compiled_model = torch.compile(model, mode="reduce-overhead")

        # Warmup compiled
        print("🔥 Warming up compiled model...")
        with torch.no_grad():
            for _ in range(10):
                _ = compiled_model(dummy_input)

        # Benchmark compiled model
        print("⚡ Benchmarking compiled model...")
        compiled_times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = compiled_model(dummy_input)
                compiled_times.append(time.time() - start)

        compiled_avg = sum(compiled_times) / len(compiled_times) * 1000
        speedup = regular_avg / compiled_avg

        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(f"Regular model:  {regular_avg:.2f}ms per inference")
        print(f"Compiled model: {compiled_avg:.2f}ms per inference")
        print(f"Speedup:        {speedup:.2f}x")
        print("=" * 60)

        if speedup > 1.1:
            print("✅ torch.compile provides significant speedup!")
        elif speedup > 1.0:
            print("✅ torch.compile provides minor speedup")
        else:
            print("⚠️  torch.compile doesn't help (model too small or overhead dominates)")

    except Exception as e:
        print(f"\n⚠️  torch.compile not available: {e}")
        print("Note: torch.compile requires PyTorch 2.0+")
        print(f"Current PyTorch version: {torch.__version__}")


if __name__ == "__main__":
    test_torch_compile()
