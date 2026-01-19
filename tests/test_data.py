from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from mlops_project.data import MyDataset


@pytest.fixture
def data_path():
    """Return path to processed data, skip if not available."""
    path = Path("data/processed/processed_data.csv")
    if not path.exists():
        pytest.skip("Processed data not available")
    return path


def test_my_dataset_is_dataset(data_path):
    """Test the MyDataset class is a Dataset."""
    dataset = MyDataset(data_path)
    assert isinstance(dataset, Dataset)


def test_dataset_length(data_path):
    """Test dataset returns correct length."""
    dataset = MyDataset(data_path)
    assert len(dataset) > 0


def test_dataset_getitem(data_path):
    """Test dataset __getitem__ returns correct shapes."""
    dataset = MyDataset(data_path, seq_len=10)
    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (10, dataset.input_size)
    assert y.shape == ()
    assert y.dtype == torch.long


def test_dataset_labels(data_path):
    """Test dataset returns valid labels (0, 1, or 2)."""
    dataset = MyDataset(data_path)
    x, y = dataset[0]

    assert y.item() in [0, 1, 2]


def test_dataset_input_size(data_path):
    """Test dataset has valid input size."""
    dataset = MyDataset(data_path)
    assert dataset.input_size > 0
