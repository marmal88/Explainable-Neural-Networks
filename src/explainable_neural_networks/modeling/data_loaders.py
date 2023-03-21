import torchvision
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.transforms as T

class ImageClassificationDataModule(LightningDataModule):
    def __init__(
            self,
            train_transform_img: T.Compose,
            test_transform_img: T.Compose,
            batch_size: int = 32,
            train_data_path: str = "../data/train",
            validation_data_path: str = "../data/val",
            test_data_path: str = "../data/test",
            ):
        super().__init__()
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=train_data_path, transform=train_transform_img
        )
        self.validation_dataset = torchvision.datasets.ImageFolder(
            root=validation_data_path, transform=test_transform_img
        )
        self.test_dataset = torchvision.datasets.ImageFolder(
            root=test_data_path, transform=test_transform_img
        )

        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
