import torch
from torch.utils.data import Dataset

# from pathlib import Path
import pytest

# import shutil
# from unittest.mock import MagicMock
from src.mlops.data import PistachioDataset
# from mlops.dataset_statistics import dataset_statistics
# from mlops.utils import show_image_and_target


@pytest.fixture
def temp_data_dir(tmp_path):
    """Fixture to create a temporary data directory."""
    raw_data_dir = tmp_path / "data/raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    return raw_data_dir


@pytest.fixture
def mock_dataset_statistics(mocker):
    """Mock the PistachioDataset class and required methods."""
    mock_dataset = mocker.patch("src.mlops.data.PistachioDataset")
    mock_dataset.return_value.__len__.return_value = 2148
    mock_dataset.return_value.__getitem__.return_value = (torch.randn(3, 224, 224), 0)
    return mock_dataset


def test_my_dataset():
    """Test if PistachioDataset can be initialized."""
    dataset = PistachioDataset("data/raw")
    assert isinstance(dataset, Dataset)


def test_dataset_length():
    """Test if dataset length matches the expected number of samples."""
    dataset = PistachioDataset("data/raw")
    expected_length = 2148
    assert len(dataset) == expected_length, f"Expected {expected_length} samples, got {len(dataset)}"


def test_dataset_sample():
    """Test if dataset returns valid samples."""
    dataset = PistachioDataset("data/raw")
    sample = dataset[0]
    assert isinstance(sample, tuple), "Sample should be a tuple (image, label)"
    assert len(sample) == 2, "Sample should contain image and label"
    assert sample[0].shape[0] == 3, "Image should have 3 channels (RGB)"


def test_image_loading():
    """Test if all images in dataset can be loaded without errors."""
    dataset = PistachioDataset("data/raw")
    for i in range(len(dataset)):
        image, label = dataset[i]
        assert isinstance(image, torch.Tensor), "Image should be a PyTorch tensor"
        assert isinstance(label, int), "Label should be an integer"


def test_dataset_transformations():
    """Test if dataset applies transformations correctly."""
    dataset = PistachioDataset("data/raw")
    image, label = dataset[0]

    assert image.shape == (3, 224, 224), "Image should be resized to (3, 224, 224)"
    assert isinstance(label, int), "Label should be an integer"


# def test_preprocess(mocker, temp_data_dir):
#     """Test end-to-end preprocessing function."""
#     output_folder = temp_data_dir / "processed"
#     mocker.patch("src.mlops.data.download_kaggle_dataset")

#     preprocess(temp_data_dir, output_folder)

#     assert output_folder.exists(), "Output folder should be created"


"""
def test_dataset_statistics(mocker, mock_dataset_statistics):
    # Test if dataset statistics are calculated and saved correctly.
    mocker.patch("matplotlib.pyplot.savefig")

    dataset_statistics("data/raw")

    assert mock_dataset_statistics.called, "PistachioDataset should be called"
    assert os.path.exists("pistachio_images.png"), "Image plot should be saved"
    assert os.path.exists("test_label_distribution.png"), "Label distribution plot should be saved"

def test_show_image_and_target(mocker, mock_dataset_statistics):
    # Mock the image visualization function to verify it runs.
    mock_show = mocker.patch("src.mlops.utils.show_image_and_target")
    dataset_statistics("data/raw")
    mock_show.assert_called()
    """
