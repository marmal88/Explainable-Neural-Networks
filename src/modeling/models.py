from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch import LightningModule


class Backbone(nn.Module):
    """
    Initialise an object of class Backbone.

    Args:
        num_classes (int): The number of output classes for the image model.
        dropout (float): Sets the dropout for neural network layers.

    Returns:
        None
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
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
        """
        Forward propagation of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch size, number of channels, height, width)

        Returns:
            x (torch.Tensor): Output after forward propagation of shape (batch size, num_classes)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classifier(x)
        return x


class ImageClassifier(LightningModule):
    """
    Initialise an object of class ImageClassifier.

    Args:
        backbone (Optional[Backbone]): An optional argument to specify the model
            used for image classification. Defaults to Backbone().
        learning_rate (float): Sets the learning rate of the optimizer used to
            train the model. Defaults to 0.0001.
        num_classes (int): The number of output classes for the image model.

    Returns:
        None
    """

    def __init__(
        self,
        backbone: Optional[Backbone] = None,
        learning_rate: float = 0.0001,  # TODO: do we want to make the learning_rate configurable
        num_classes: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        if backbone is None:
            backbone = Backbone()
        self.model = backbone

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of the model.

        Args:
            x (torch.Tensor): input tensor of shape (batch size, number of channels, height, width)

        Returns:
            x (torch.Tensor): output after forward propagation of shape (batch size, num_classes)
        """
        return self.model(x)

    def _shared_step(
        self, batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates loss and predicted labels for a batch of input data.

        Args:
            batch (Tuple): A tuple containing input features and true labels.

        Returns:
            loss (torch.Tensor): cross entropy loss between predicted and true labels.
            true_labels (torch.Tensor): class indices of the true labels
            predicted_labels (torch.Tensor): class indices of the predicted labels
        """
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        """
        Training step of model. Logs training loss and accuracy.

        Args:
            batch (Tuple): A tuple containing input features and true labels.
            batch_idx (int): batch index

        Returns:
            loss (torch.Tensor): cross entropy loss between predicted and true labels.
        """
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss, on_epoch=True)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of model. Logs validation loss and accuracy.

        Args:
            batch (Tuple): A tuple containing input features and true labels.
            batch_idx (int): batch index

        Returns:
            None
        """
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, on_step=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Test step of model. Logs test accuracy.

        Args:
            batch (Tuple): A tuple containing input features and true labels.
            batch_idx (int): batch index

        Returns:
            None
        """
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("accuracy", self.test_acc)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Predict step of model. Returns predictions.

        Args:
            batch (Tuple): A tuple containing input features and true labels.
            batch_idx (int): batch index
            dataloader_idx (int): dataloader index

        Returns:
            x (torch.Tensor): output from prediction step
        """
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        """
        Configures optimizer

        Args:
            None

        Return:
            optimizer (Callable): optimizer used for model training
        """
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
