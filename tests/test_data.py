from torch.utils.data import Dataset

from mlops.data import PistachioDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = PistachioDataset("data/raw")
    assert isinstance(dataset, Dataset)
