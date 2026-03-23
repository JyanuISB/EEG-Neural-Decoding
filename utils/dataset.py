"""
dataset.py — PyTorch Dataset class for EEG epochs.

Wraps numpy epoch arrays into a format compatible with PyTorch DataLoader.
Each epoch is independently normalized (zero mean, unit variance per channel)
to remove amplitude differences between subjects and sessions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG motor imagery epochs.

    Parameters
    ----------
    epochs : np.ndarray, shape (n_epochs, n_channels, n_times)
        Raw EEG data in microvolts
    labels : np.ndarray, shape (n_epochs,)
        Integer class labels (0 = left hand, 1 = right hand)
    model_type : str
        'eegnet'  -> adds a channel dimension -> (1, n_channels, n_times)
        'cnn_lstm' -> keeps as (n_channels, n_times)
    """

    def __init__(self, epochs: np.ndarray, labels: np.ndarray, model_type: str = 'eegnet'):
        self.epochs = epochs.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.model_type = model_type.lower()

        # Validate shapes
        assert self.epochs.ndim == 3, \
            f"Expected epochs shape (n, C, T), got {self.epochs.shape}"
        assert len(self.epochs) == len(self.labels), \
            "Number of epochs and labels must match"

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        epoch = self.epochs[idx].copy()  # shape: (n_channels, n_times)

        # -- Per-channel normalization ----------------------------------------
        # Subtract mean and divide by std for each channel independently.
        # This handles amplitude differences between electrodes and subjects.
        mean = epoch.mean(axis=1, keepdims=True)   # (n_channels, 1)
        std  = epoch.std(axis=1, keepdims=True)    # (n_channels, 1)
        std  = np.where(std < 1e-8, 1e-8, std)    # avoid division by zero
        epoch = (epoch - mean) / std

        label = self.labels[idx]

        # EEGNet expects 4D input: (batch, 1, n_channels, n_times)
        # so we add a dummy "image channel" dimension here
        if self.model_type == 'eegnet':
            epoch = epoch[np.newaxis, ...]  # -> (1, n_channels, n_times)

        return torch.tensor(epoch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def make_data_loaders(train_epochs, train_labels, test_epochs, test_labels,
                      model_type='eegnet', batch_size=32, num_workers=0):
    """
    Convenience function: create train and test DataLoaders.

    Parameters
    ----------
    train_epochs, train_labels : np.ndarray  — training split
    test_epochs, test_labels   : np.ndarray  — test split
    model_type : str  — 'eegnet' or 'cnn_lstm'
    batch_size : int
    num_workers : int  — parallel data loading workers (0 = main process)

    Returns
    -------
    train_loader, test_loader : torch.utils.data.DataLoader
    """
    train_ds = EEGDataset(train_epochs, train_labels, model_type=model_type)
    test_ds  = EEGDataset(test_epochs,  test_labels,  model_type=model_type)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Quick smoke test with random data
    np.random.seed(42)
    fake_epochs = np.random.randn(100, 64, 481).astype(np.float32)
    fake_labels = np.random.randint(0, 2, size=100).astype(np.int64)

    ds = EEGDataset(fake_epochs, fake_labels, model_type='eegnet')
    x, y = ds[0]
    print(f"EEGNet mode — x shape: {x.shape}, label: {y.item()}")

    ds2 = EEGDataset(fake_epochs, fake_labels, model_type='cnn_lstm')
    x2, y2 = ds2[0]
    print(f"CNN-LSTM mode — x shape: {x2.shape}, label: {y2.item()}")
    print("Dataset test PASSED.")
