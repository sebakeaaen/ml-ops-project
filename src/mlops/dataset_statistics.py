import typer
from mlops.utils import show_image_and_target
from matplotlib import pyplot as plt
from mlops.data import load_data, PistachioDataset


def dataset_statistics() -> None:
    """Compute dataset statistics."""
    train_dataset = PistachioDataset("data/raw")
    print("Pistachio dataset")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")

    train_dataset, test_dataset = load_data()
    images, targets = next(iter(train_dataset))
    show_image_and_target(images[:4], targets[:4], show=False)
    plt.savefig("reports/figures/pistachio_images.png")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)
