import torch
from torch.utils.data import Dataset
from src.mlops.data import PistachioDataset
import os
from mlops.dataset_statistics import dataset_statistics


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


def test_dataset_statistics():
    """Test if dataset statistics are written to file."""
    dataset_statistics()
    stats_file = "temp/dataset_statistics.txt"
    assert os.path.exists(stats_file), "Statistics file was not created."
    with open(stats_file, "r") as f:
        content = f.read()
        assert "Number of images:" in content, "Statistics content is incorrect."
    os.remove(stats_file)  # Cleanup
