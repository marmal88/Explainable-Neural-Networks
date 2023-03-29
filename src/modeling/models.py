from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule


class Backbone(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(16 * 1 * 1, 120),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classifier(x)
        return x


class ImageClassifier(LightningModule):
    def __init__(
        self, backbone: Optional[Backbone] = None, learning_rate: float = 0.0001
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        if backbone is None:
            backbone = Backbone()
        self.backbone = backbone

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
