import pytest
import torch

from mlops_project.model import Model


def test_model_initialization():
    """Test model initializes correctly."""
    model = Model(input_size=18, hidden_size=64, num_layers=2)
    assert model is not None


def test_model_forward_pass():
    """Test model forward pass works."""
    model = Model(input_size=18, hidden_size=64, num_layers=2)
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 18)

    output = model(x)

    assert output.shape == (batch_size, 3)


def test_model_output_shape():
    """Test model outputs correct number of classes."""
    model = Model(input_size=18, hidden_size=32, num_layers=1, output_size=3)
    x = torch.randn(2, 10, 18)

    output = model(x)

    assert output.shape[1] == 3


def test_model_with_different_hidden_size():
    """Test model works with different hidden sizes."""
    model = Model(input_size=18, hidden_size=128, num_layers=2)
    x = torch.randn(1, 10, 18)

    output = model(x)

    assert output.shape == (1, 3)


def test_model_parameters_count():
    """Test model has trainable parameters."""
    model = Model(input_size=18, hidden_size=64, num_layers=2)
    params = sum(p.numel() for p in model.parameters())

    assert params > 0
