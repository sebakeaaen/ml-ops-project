import os
import typer
from mlops.utils import show_image_and_target
from matplotlib import pyplot as plt
from mlops.data import load_data, PistachioDataset


def dataset_statistics() -> None:
    """Compute dataset statistics."""
    train_dataset = PistachioDataset("data/raw")

    if not os.path.exists("temp"):
        os.makedirs("temp", exist_ok=True)

    with open("temp/dataset_statistics.txt", "w") as f:
        f.write("Pistachio dataset\n")
        f.write(f"Number of images: {len(train_dataset)}\n")
        f.write(f"Image shape: {train_dataset[0][0].shape}\n")
        f.write("![](pistachio_images.png)")

    train_dataset, test_dataset = load_data()
    images, targets = next(iter(train_dataset))
    show_image_and_target(images[:4], targets[:4], show=False)
    plt.savefig("temp/pistachio_images.png")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)
