import torch
import pytest
import pytorch_lightning as pl
from src.mlops.model import resnetSimple, MetricsTracker, load_data


@pytest.fixture
def model():
    """Fixture to initialize the model."""
    return resnetSimple(num_classes=2, pretrained=False)


@pytest.fixture
def sample_input():
    """Fixture to provide a sample input tensor with batch size 4 and 3 RGB channels."""
    return torch.randn(4, 3, 224, 224)


def test_model_initialization(model):
    """Test model initialization with correct parameters."""
    assert isinstance(model, resnetSimple), "Model should be an instance of resnetSimple"
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), "Loss function should be CrossEntropyLoss"
    assert model.learning_rate == 0.003, "Default learning rate should be 0.003"


def test_forward_pass(model, sample_input):
    """Test model's forward pass outputs the correct shape."""
    output = model(sample_input)
    assert output.shape == (4, 2), "Output shape should match (batch_size, num_classes)"


def test_training_step(model, sample_input):
    """Test training step computes loss correctly."""
    labels = torch.tensor([0, 1, 0, 1])
    loss = model.training_step((sample_input, labels), 0)
    assert loss > 0, "Loss should be positive"

"""
def test_validation_step():
    #Test validation step logs loss and accuracy.
    model = resnetSimple(num_classes=2, pretrained=False)
    # Sample data
    sample_input = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1])

    trainer = pl.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False
    )
    
    trainer.validate(model, [(sample_input, labels)])

    logged_metrics = trainer.callback_metrics
    assert "val_loss" in logged_metrics, "Validation loss should be logged"
    assert "val_accuracy" in logged_metrics, "Validation accuracy should be logged"
"""
    
def test_configure_optimizers(model):
    """Test optimizer configuration."""
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be Adam"
