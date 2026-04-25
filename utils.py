import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from torchvision import transforms

class DrivingDataset(Dataset):
    """
    X : frame
    y = steering, throttle, brake
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.frames_dir = df["Frame"]

        # shrinks image down to 128x128 from 512x512
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        X = Image.open(row["Frame"]).convert("RGB")
        X = self.transform(X)

        y = torch.tensor([
            row["Steering"],
            row["Throttle"],
            row["Brake"],
        ], dtype=torch.float32)
        return X, y
        