"""
Author(s): Matt Gallenberger, Brendel
Class: CS450-01
Date: 04/29/26
"""

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from torchvision import transforms


class DrivingDataset(Dataset):
    """
    X = distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed
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
            row["curvature"],
            row["roadWidth"],
            row["drivability"],
            row["Speed"]
        ])

        y = torch.tensor([
            row["Steering"],
            row["Throttle"],
            row["Brake"],
        ], dtype=torch.float32)

        return X, y