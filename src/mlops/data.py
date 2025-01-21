from pathlib import Path
import torch
import typer
import shutil
import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import kagglehub
from typing_extensions import Annotated
from torch.utils.data import DataLoader, random_split
from torch import manual_seed


def ensure_permissions(folder: Path) -> None:
    """Ensure read and write permissions for all files in a folder."""
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o755)  # Read/write for owner, read for others
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)


class PistachioDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = Path(raw_data_path)
        self.image_paths = list(self.data_path.glob("**/*.jpg")) + list(self.data_path.glob("**/*.png"))
        print(f"Found {len(self.image_paths)} images in {self.data_path}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def process_images(self, output_folder: Path) -> None:
        """Transform and save raw images to the output folder, preserving folder structure."""
        print(f"Processing images from {self.data_path}...")
        os.makedirs(output_folder, exist_ok=True)

        ensure_permissions(self.data_path)

        if len(self.image_paths) == 0:
            print("No images found for processing.")
            return

        for image_path in self.image_paths:
            relative_path = image_path.relative_to(self.data_path)
            output_path = output_folder / relative_path.parent
            os.makedirs(output_path, exist_ok=True)

            processed_image_path = output_path / image_path.name

            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image_tensor = self.transform(image)
                image = transforms.ToPILImage()(image_tensor)
            image.save(processed_image_path, format="JPEG")

        print(f"All processed images saved to {output_folder}")


def download_kaggle_dataset(dataset_name: str, raw_data_path: Path) -> None:
    """Download a dataset from Kaggle to the raw data path."""
    print(f"Downloading dataset {dataset_name}...")
    path = kagglehub.dataset_download(dataset_name)
    print(f"Path to dataset files: {path}")

    current_path = os.getcwd()
    data_folder = os.path.join(current_path, "data", "raw")

    os.makedirs(data_folder, exist_ok=True)

    for item in os.listdir(path):
        source = os.path.join(path, item)
        destination = os.path.join(data_folder, item)
        if os.path.isdir(source):
            shutil.move(source, destination)
        else:
            shutil.move(source, destination)

    print(f"Files moved to: {os.path.abspath(data_folder)}")


def preprocess(
    raw_data_path: Annotated[Path, typer.Option(help="Path to the folder where raw data is stored")],
    output_folder: Annotated[Path, typer.Option(help="Path to the folder where processed data will be saved")],
    dataset_name: Annotated[
        str, typer.Option(help="Name of the Kaggle dataset to download")
    ] = "muratkokludataset/pistachio-image-dataset",
) -> None:
    """Download, preprocess, and save the dataset."""
    print("Starting the dataset pipeline...")

    if not raw_data_path.exists() or len([f for f in raw_data_path.iterdir() if not f.name.startswith(".")]) == 0:
        download_kaggle_dataset(dataset_name, raw_data_path)

    dataset = PistachioDataset(raw_data_path)
    dataset.process_images(output_folder)


class PreprocessedDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        print("Working dir: " + os.getcwd())
        self.file_list = os.listdir(tensor_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.tensor_dir, self.file_list[idx])
        return torch.load(file_path)


def load_data(batch_size=64, split=0.8, num_workers=0, seed=0):
    # Image transformations
    manual_seed(seed)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # Resnet18 was trained on images normalized in this fashion, so best to normalize our images the same way
        ]
    )

    # Load dataset
    data_path = "data/processed/Pistachio_Image_Dataset/Pistachio_Image_Dataset"
    tensor_dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # Split into train and validation sets
    train_size = int(split * len(tensor_dataset))
    val_size = len(tensor_dataset) - train_size
    train_dataset, val_dataset = random_split(tensor_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


if __name__ == "__main__":
    typer.run(preprocess)
