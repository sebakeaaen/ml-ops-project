import matplotlib.pyplot as plt
import typer
from model import resnetSimple, MetricsTracker, load_data
import torch
import pytorch_lightning as pl
from typing_extensions import Annotated

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 1,
    model_checkpoint: Annotated[str, typer.Option(help="Path to store model checkpoint")] = "models/model.ckpt",
) -> None:
    print(f"{lr=}, {batch_size=}, {epochs=}")
    model = resnetSimple(learning_rate=lr).to(DEVICE)

    train_loader, _ = load_data(batch_size=batch_size, split=0.8)

    metrics_tracker = MetricsTracker()
    # Instantiate the model, loss, and optimizer
    trainer = pl.Trainer(
        callbacks=[metrics_tracker],
        max_epochs=epochs,
        # log_every_n_steps=9,
        logger=True,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader)

    print("Training complete")
    torch.save(model.state_dict(), "models/model.ckpt")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(metrics_tracker.train_losses)
    axs[0].set_title("Train loss")
    axs[1].plot(metrics_tracker.train_acc)
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    typer.run(train)
