"""This package contains dataloaders for train, validation and test dataset."""

import pandas as pd
import torchvision.transforms as T
from lightning.pytorch import LightningDataModule
from modeling.dataset import PneumoniaDataset
from torch.utils.data import DataLoader


class ImageClassificationDataModule(LightningDataModule):
    """
    Instantialises a LightningDataModule for loading and
    preprocessing image classification datasets.

    Args:
        train_image_folder (str): Path to the folder containing training images.
        val_image_folder (str): Path to the folder containing validation images.
        test_image_folder (str): Path to the folder containing test images.
        meta_data_path (str): Path to the CSV file containing metadata for all images.
        train_transform_img (Callable): Transformations to apply to training images.
        test_transform_img (Callable): Transformations to apply to validation and test images.
        batch_size (int): Number of samples in a batch. Defaults to 32.
    """

    def __init__(
        self,
        train_image_folder: str,
        val_image_folder: str,
        test_image_folder: str,
        meta_data_path: str,
        train_transform_img: T.Compose,
        test_transform_img: T.Compose,
        batch_size: int = 32,
    ):
        super().__init__()
        full_df = pd.read_csv(meta_data_path)
        train_df = full_df[full_df["tts"] == "train"]
        val_df = full_df[full_df["tts"] == "val"]
        test_df = full_df[full_df["tts"] == "test"]

        self.train_dataset = PneumoniaDataset(
            image_folder=train_image_folder,
            data=train_df,
            inference_mode=False,
            transform=train_transform_img,
        )

        self.val_dataset = PneumoniaDataset(
            image_folder=val_image_folder,
            data=val_df,
            inference_mode=False,
            transform=test_transform_img,
        )

        self.test_dataset = PneumoniaDataset(
            image_folder=test_image_folder,
            data=test_df,
            inference_mode=False,
            transform=test_transform_img,
        )

        self.batch_size = batch_size

    def train_dataloader(self):
        """
        Creates DataLoader to iterate over the training dataset.

        Args:
            None

        Returns:
            data_loader (DataLoader): Dataloader with shuffling
        """
        data_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )
        return data_loader

    def val_dataloader(self):
        """
        Creates DataLoader to iterate over the validation dataset.

        Args:
            None

        Returns:
            data_loader (DataLoader): Dataloader without shuffling
        """
        data_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4
        )
        return data_loader

    def test_dataloader(self):
        """
        Creates DataLoader to iterate over the test dataset.

        Args:
            None

        Returns:
            data_loader (DataLoader): Dataloader without shuffling
        """
        data_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=4
        )
        return data_loader

    def predict_dataloader(self):
        """
        Creates DataLoader to iterate over the dataset.

        Args:
            None

        Returns:
            data_loader (DataLoader): Dataloader without shuffling
        """
        data_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=4
        )
        return data_loader
