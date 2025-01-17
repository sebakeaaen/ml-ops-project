import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class resnetSimple(pl.LightningModule):
    def __init__(self, num_classes=2, pretrained=True, learning_rate = 0.003):
        super(resnetSimple, self).__init__()
        self.model = models.resnet18(weights= models.ResNet18_Weights.DEFAULT if pretrained  else None)

        # Replace the preset num of classes to our own
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images,labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss,  on_epoch=True, prog_bar=True)
        self.log('train_accuracy', acc,  on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss,on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc,on_epoch=True,  prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    

# This is attached to the pl.trainer to save statistics for each epoch
class MetricsTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.train_acc = []
        self.val_losses = []
        self.val_accuracies = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss_epoch"].item())
        self.train_acc.append(trainer.callback_metrics["train_accuracy_epoch"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())
        self.val_accuracies.append(trainer.callback_metrics["val_accuracy"].item())

def load_data(imgs_path = "data/processed/Pistachio_Image_Dataset/Pistachio_Image_Dataset"):
    # Image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Resnet18 was trained on images normalized in this fashion, so best to normalize our images the same way
    ])
    # Load dataset
    data_path = "data/processed/Pistachio_Image_Dataset/Pistachio_Image_Dataset"
    dataset = datasets.ImageFolder(root=data_path, transform=transform)


    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader , val_loader



if __name__ == "__main__":
    model = resnetSimple()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
