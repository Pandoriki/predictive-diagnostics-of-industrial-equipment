import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from configs.config import cfg

class RULDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]

    @staticmethod
    def create_dataloaders(X_train_data: np.ndarray, y_train_data: np.ndarray, 
                           X_val_data: np.ndarray, y_val_data: np.ndarray, 
                           batch_size: int = cfg.BATCH_SIZE) -> (DataLoader, DataLoader):
        train_dataset = RULDataset(X_train_data, y_train_data)
        val_dataset = RULDataset(X_val_data, y_val_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader