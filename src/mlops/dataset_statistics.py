import typer
from mlops.utils import show_image_and_target
from matplotlib import pyplot as plt
import torch
from mlops.data import PistachioDataset


def dataset_statistics(datadir: str = "data/raw/") -> None:
    """Compute dataset statistics."""
    dataset = PistachioDataset(raw_data_path=datadir)

    print("Test dataset: pistacio")
    print(f"Number of images: {len(dataset)}")
    print(f"Image shape: {dataset[0][0].shape}")

    show_image_and_target(dataset[:25], dataset[:25], show=False)
    plt.savefig("pistachio_images.png")
    plt.close()

    test_label_distribution = torch.bincount(dataset.target)

    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("test_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)
