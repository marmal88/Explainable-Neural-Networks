import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PneumoniaDataset(Dataset):
    def __init__(
        self,
        image_folder: str,
        data: pd.DataFrame,
        inference_mode: bool = False,
        transform=None,
    ):
        self.image_folder = image_folder
        self.data = data
        self.inference_mode = inference_mode
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = os.path.join(self.image_folder, row["file_path"])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.inference_mode is False:
            label = row["label"]

        return image, label
