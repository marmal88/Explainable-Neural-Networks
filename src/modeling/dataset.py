"""This package contains datasets for train, validation and test dataset."""

import os
from typing import Callable, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PneumoniaDataset(Dataset):
    """
    A custom PyTorch dataset for loading images and their labels.

    Args:
        image_folder (str): The path to the directory containing the images.
        data (pd.DataFrame): A pandas DataFrame containing the file paths
            of the images and their labels.
        inference_mode (bool): Specify if the dataset is used for inference
            or training. Defaults to False.
        transform (Callable): A callable object that applies image augmentation
            on the input image. Defaults to None.

    Returns:
        None
    """

    def __init__(
        self,
        image_folder: str,
        data: pd.DataFrame,
        inference_mode: bool = False,
        transform: Callable = None,
    ) -> None:
        self.image_folder = image_folder
        self.data = data
        self.inference_mode = inference_mode
        self.transform = transform

    def __len__(self) -> int:
        """
        Gets the number of rows in the dataset

        Args:
            None

        Returns:

        """
        len_data = len(self.data)
        return len_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Gets data with the specified index from the dataset.

        Args:
            idx (int): Index of the data to retrieve

        Returns:
            item (Tuple): A tuple containing the image tensor and label if not in inference mode,
            else a tuple containing only the image tensor.
        """
        row = self.data.iloc[idx]

        image_path = os.path.join(self.image_folder, row["file_path"])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.inference_mode is False:
            label = row["label"]

        item = image, label
        return item
