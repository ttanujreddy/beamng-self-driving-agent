import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from torchvision import transforms


class DrivingDataset(Dataset):
    """
    X = distFromCenter, headingAngle, xCurvature, yCurvature, roadWidth, drivability
    y = steering, throttle, brake
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        X = torch.tensor([
            row["distFromCenter"],
            row["headingAngle"],
            row["xCurvature"],
            row["yCurvature"],
            row["roadWidth"],
            row["drivability"]
        ])

        y = torch.tensor([
            row["Steering"],
            row["Throttle"],
            row["Brake"],
        ], dtype=torch.float32)

        return X, y