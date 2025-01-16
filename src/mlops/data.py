from pathlib import Path
import typer
import shutil
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage 
from torch.utils.data import Dataset
import kagglehub

def ensure_permissions(folder: Path) -> None:
    """Ensure read and write permissions for all files in a folder."""
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o755)  # Read/write for owner, read for others
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.image_paths = list(self.data_path.glob("**/*.jpg")) + list(self.data_path.glob("**/*.png"))
        print(f"Found {len(self.image_paths)} images in {self.data_path}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

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


def download_kaggle_dataset(dataset_name: str) -> None:
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
    raw_data_path: Path = typer.Option(..., help="Path to the folder where raw data is stored"),
    output_folder: Path = typer.Option(..., help="Path to the folder where processed data will be saved"),
    dataset_name: str = typer.Option("muratkokludataset/pistachio-image-dataset", help="Name of the Kaggle dataset to download"),
) -> None:
    """Download, preprocess, and save the dataset."""
    print("Starting the dataset pipeline...")

    if not raw_data_path.exists() or len([f for f in raw_data_path.iterdir() if not f.name.startswith('.')]) == 0:
        download_kaggle_dataset(dataset_name, raw_data_path)

    dataset = MyDataset(raw_data_path)
    dataset.process_images(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
