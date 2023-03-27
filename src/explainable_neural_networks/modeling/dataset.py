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

        print(f" transform is {self.transform is not None}")
        if self.transform is not None:
            image = self.transform(image)
        print(image)

        item = {"image": image}

        if self.inference_mode is False:
            label = row["class"]
            item["label"] = label

        return item
