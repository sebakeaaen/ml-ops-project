import torch
import typer
import pytorch_lightning as pl
from model import resnetSimple, MetricsTracker, load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    _, val_loader = load_data()

    model = resnetSimple().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    metrics_tracker = MetricsTracker()

    trainer = pl.Trainer(
        callbacks=[metrics_tracker],
        max_epochs=2,
        log_every_n_steps=9,
        logger=True,
        enable_progress_bar=True,
        num_sanity_val_steps=0
        )
    trainer.validate(model, val_loader)
    print(f"Test accuracy: {metrics_tracker.val_accuracies[0]}")


if __name__ == "__main__":
    typer.run(evaluate)
    #evaluate("models/model.pth")