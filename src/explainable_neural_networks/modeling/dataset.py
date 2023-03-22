import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os 

class PneumoniaDataset(Dataset):

    def __init__(self, 
                 data: pd.DataFrame,
                 image_folder: str):
        self.data = data
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = row.label
        image_path = os.path.join(self.image_folder, row.image_file_name)
        image = Image.open(image_path).convert("RGB")
        # return a dictionary containing image and label
        item = {"image": image,
                "label": label}
        return item