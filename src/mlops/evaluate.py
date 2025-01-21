import torch
import pytorch_lightning as pl
from mlops.model import resnetSimple, MetricsTracker
from mlops.data import load_data
import hydra
from pytorch_lightning.loggers import CSVLogger


@hydra.main(config_path="../../configs", config_name="default_config.yaml", version_base="1.1")
def evaluate(cfg) -> None:
    """
    (
        model_checkpoint: Annotated[
            str, typer.Option(help="Path to model checkpoint used for evaluation")
        ] = "models/model.ckpt",
    )
    -> None:
    """
    logger = CSVLogger("logs/", name="evaluating")
    trainer = pl.Trainer(logger=logger)
    config = cfg.experiment

    logger.log_hyperparams(dict(config))

    # Model Hyperparameters
    # dataset_path = config.dataset_path
    cuda = config.cuda
    DEVICE = torch.device("cuda" if cuda else "cpu")
    batch_size = config.batch_size
    epochs = config.n_epochs
    # seed = config.seed
    dataset_split = config.dataset_split

    # torch.manual_seed(seed)

    _, val_loader = load_data(batch_size=batch_size, split=dataset_split)

    model = resnetSimple().to(DEVICE)
    model.load_state_dict(torch.load("models/model.ckpt"))

    metrics_tracker = MetricsTracker()

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[metrics_tracker],
        max_epochs=epochs,
        # log_every_n_steps=9,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.validate(model, val_loader)
    print(f"Test accuracy: {metrics_tracker.val_accuracies[0]}")


if __name__ == "__main__":
    # typer.run(evaluate)
    evaluate()
