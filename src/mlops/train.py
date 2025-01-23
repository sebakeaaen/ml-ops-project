import matplotlib.pyplot as plt
from mlops.model import resnetSimple, MetricsTracker
from mlops.data import load_data
import torch
import pytorch_lightning as pl
import hydra
from pytorch_lightning.loggers import CSVLogger


@hydra.main(config_path="../../configs", config_name="default_config.yaml", version_base="1.1")
def train(cfg):
    """
    (
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 1,
        model_checkpoint: Annotated[str, typer.Option(help="Path to store model checkpoint")] = 'models/model.ckpt',
        ) -> None:
    """
    logger = CSVLogger("logs/", name="training")
    # trainer = pl.Trainer(logger=logger)
    config = cfg.experiment

    logger.log_hyperparams(dict(config))

    # Model Hyperparameters
    dataset_path = config.dataset_path
    model_path = config.model_path
    cuda = config.cuda
    DEVICE = torch.device("cuda" if cuda else "cpu")
    batch_size = config.batch_size
    lr = config.lr
    epochs = config.n_epochs
    seed = config.seed
    dataset_split = config.dataset_split

    torch.manual_seed(seed)

    model = resnetSimple(learning_rate=lr).to(DEVICE)

    train_loader, _ = load_data(imgs_path=dataset_path, batch_size=batch_size, split=dataset_split)

    metrics_tracker = MetricsTracker()
    # Instantiate the model, loss, and optimizer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[metrics_tracker],
        max_epochs=epochs,
        log_every_n_steps=9,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader)

    print("Training complete")
    torch.save(model.state_dict(), model_path)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(metrics_tracker.train_losses)
    axs[0].set_title("Train loss")
    axs[1].plot(metrics_tracker.train_acc)
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    # typer.run(train)
    train()
