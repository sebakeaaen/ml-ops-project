from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
import torch
from src.mlops.model import resnetSimple

def test_model_initialization():
    """Test if the model initializes correctly."""
    model = resnetSimple(num_classes=2, pretrained=False)
    assert isinstance(model, resnetSimple), "Model should be an instance of resnetSimple"

def test_model_forward_pass():
    """Test forward pass of the model."""
    model = resnetSimple(num_classes=2, pretrained=False)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == torch.Size([1, 2]), "Output should have shape [1, 2]"

def test_model_training_step():
    model = resnetSimple(num_classes=2, pretrained=False)
    trainer = Trainer(fast_dev_run=True)
    inputs = torch.randn(10, 3, 224, 224)
    labels = torch.randint(0, 2, (10,))
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=2)
    trainer.fit(model, dataloader)

def test_model_optimizer():
    """Test optimizer initialization."""
    model = resnetSimple(num_classes=2, pretrained=False)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be Adam"
