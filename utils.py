"""utils.py - PyTorch dataset wrapper for BeamNG driving rows.

Author(s): Matt Gallenberger, Brendel
Class: CS450-01
Date: 04/29/26
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from state_schema import FEATURE_COLUMNS, TARGET_COLUMNS, row_to_features, row_to_targets


class DrivingDataset(Dataset):
    """Dataset for supervised behavior-cloning training.

    Each item is one recorded driving example:

        X: road/electrics state vector, shape (6,)
        y: control label vector, shape (3,)

    Args:
        df: pandas DataFrame containing all columns listed in state_schema.py.
    """

    def __init__(self, df: pd.DataFrame):
        # Resetting the index lets __getitem__ use iloc safely even after
        # train_test_split produces non-contiguous original indices.
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        """Return number of examples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """Return one training example as float32 tensors.

        Args:
            idx: Integer row index.

        Returns:
            Tuple (X, y) where:
                X is a torch.float32 tensor of length 6.
                y is a torch.float32 tensor of length 3.
        """
        row = self.df.iloc[idx]

        X = torch.tensor(row_to_features(row), dtype=torch.float32)
        y = torch.tensor(row_to_targets(row), dtype=torch.float32)

        return X, y


# Re-export the column names so notebook users can inspect them from utils.
__all__ = ["DrivingDataset", "FEATURE_COLUMNS", "TARGET_COLUMNS"]
