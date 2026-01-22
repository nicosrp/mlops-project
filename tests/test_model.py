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
    model.eval()  # Set to eval mode to avoid BatchNorm issues with batch_size=1
    x = torch.randn(2, 10, 18)  # Use batch_size=2 to avoid BatchNorm issues

    output = model(x)

    assert output.shape == (2, 3)


def test_model_parameters_count():
    """Test model has trainable parameters."""
    model = Model(input_size=18, hidden_size=64, num_layers=2)
    params = sum(p.numel() for p in model.parameters())

    assert params > 0


def test_model_with_attention():
    """Test model works with attention mechanism."""
    model = Model(input_size=22, hidden_size=64, num_layers=2, use_attention=True)
    x = torch.randn(2, 10, 22)

    output = model(x)

    assert output.shape == (2, 3)


def test_model_without_attention():
    """Test model works without attention mechanism."""
    model = Model(input_size=22, hidden_size=64, num_layers=2, use_attention=False)
    x = torch.randn(2, 10, 22)

    output = model(x)

    assert output.shape == (2, 3)


def test_model_dropout_parameter():
    """Test model dropout parameter is set correctly."""
    model = Model(input_size=22, hidden_size=64, num_layers=2, dropout=0.5)
    assert model.lstm.dropout > 0


def test_model_eval_mode():
    """Test model can be set to eval mode."""
    model = Model(input_size=22, hidden_size=64, num_layers=2)
    model.eval()

    # In eval mode, dropout should be disabled
    x = torch.randn(1, 10, 22)
    output1 = model(x)
    output2 = model(x)

    # Outputs should be identical in eval mode
    assert torch.allclose(output1, output2)


def test_model_train_mode():
    """Test model can be set to train mode."""
    model = Model(input_size=22, hidden_size=64, num_layers=2, dropout=0.3)
    model.train()

    x = torch.randn(2, 10, 22)  # batch_size=2 (BatchNorm requires >1 in train mode)
    # Just check it doesn't crash in train mode
    output = model(x)
    assert output.shape == (2, 3)
