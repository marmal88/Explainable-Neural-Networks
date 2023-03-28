from typing import Optional

import pandas as pd
import torchvision
import torchvision.transforms as T
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import PneumoniaDataset


class ImageClassificationDataModule(LightningDataModule):
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
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
