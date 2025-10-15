import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import configs.config as cfg # ИЗМЕНЕНИЕ!
# ... (везде где было cfg.XXX, теперь останется cfg.XXX)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., noise_factor=cfg.AUGMENTATION_NOISE_FACTOR):
        self.mean = mean
        self.std = std
        self.noise_factor = noise_factor

    def __call__(self, sample):
        features, target = sample
        
        if self.noise_factor > 0:
            std_scaled = self.noise_factor * (features.max() - features.min()).mean()
            noise = torch.randn(features.size()) * std_scaled

            features_augmented = (features + noise).clamp(0, 1)
            return features_augmented, target
        else:
            return features, target

class RULDataset(Dataset):
    """PyTorch Dataset для предсказания Remaining Useful Life (RUL)."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, transform=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        sample = self.features[idx], self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def create_dataloaders(X_train_data: np.ndarray, y_train_data: np.ndarray, 
                           X_val_data: np.ndarray, y_val_data: np.ndarray, 
                           batch_size: int = cfg.BATCH_SIZE, 
                           use_augmentation: bool = False) -> (DataLoader, DataLoader):
        
        train_transform = None
        if use_augmentation:
            train_transform = AddGaussianNoise(noise_factor=cfg.AUGMENTATION_NOISE_FACTOR)

        train_dataset = RULDataset(X_train_data, y_train_data, transform=train_transform)
        val_dataset = RULDataset(X_val_data, y_val_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader