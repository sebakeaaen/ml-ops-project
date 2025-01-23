from matplotlib import pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid


def show_image_and_target(images: torch.Tensor, target: torch.Tensor, show: bool = True) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    if show:
        plt.show()
