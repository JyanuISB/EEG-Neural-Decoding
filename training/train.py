"""
train.py — Training loop for EEGNet and CNN-LSTM on PhysioNet EEGBCI data.

Cross-subject split:
  Train: subjects S001–S007  (70%)
  Test : subjects S008–S010  (30%)

Usage:
  python training/train.py

Outputs saved to data/:
  - eegnet_best.pt        — best EEGNet checkpoint
  - cnn_lstm_best.pt      — best CNN-LSTM checkpoint
  - eegnet_history.npy    — training curves
  - cnn_lstm_history.npy  — training curves
  - eegnet_curves.png     — loss/accuracy plot
  - cnn_lstm_curves.png   — loss/accuracy plot
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from utils.preprocess import load_multiple_subjects
from utils.dataset import make_data_loaders
from models.eegnet import EEGNet
from models.cnn_lstm import CNNLSTM

# -- Reproducibility -----------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -- Hyperparameters -----------------------------------------------------------
TRAIN_SUBJECTS = list(range(1, 8))   # S001–S007
TEST_SUBJECTS  = list(range(8, 11))  # S008–S010
EPOCHS         = 50
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
N_CLASSES      = 2
DATA_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data')

os.makedirs(DATA_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# -- Helpers -------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    """Run one full pass over the training set, return avg loss + accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += len(y)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    """Evaluate model on a loader, return avg loss + accuracy."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += len(y)

    return total_loss / total, correct / total


def save_curves(history: dict, model_name: str):
    """Plot and save training / validation loss + accuracy curves."""
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{model_name} — Training Curves', fontsize=14)

    epochs_range = range(1, len(history['train_loss']) + 1)

    # Loss subplot
    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss',  color='steelblue')
    axes[0].plot(epochs_range, history['val_loss'],   label='Val Loss',    color='tomato')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss'); axes[0].legend()

    # Accuracy subplot
    axes[1].plot(epochs_range, history['train_acc'], label='Train Acc', color='steelblue')
    axes[1].plot(epochs_range, history['val_acc'],   label='Val Acc',   color='tomato')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy'); axes[1].legend()

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    out_path = os.path.join(DATA_DIR, f'{safe_name}_curves.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves -> {out_path}")


def train_model(model, model_name: str, train_loader, val_loader, model_type: str):
    """
    Full training loop with early stopping via best-val-accuracy checkpoint.

    Returns history dict with lists of train/val loss and accuracy per epoch.
    """
    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=7, factor=0.5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    ckpt_path = os.path.join(DATA_DIR, f'{model_type}_best.pt')

    print(f"\n{'='*55}")
    print(f"  Training: {model_name}")
    print(f"{'='*55}")

    for epoch in tqdm(range(1, EPOCHS + 1), desc=f'{model_name}'):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step(vl_acc)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        # Save checkpoint whenever validation accuracy improves
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': vl_acc,
                'history': history
            }, ckpt_path)

    print(f"  Best val accuracy: {best_val_acc:.4f}  (saved to {ckpt_path})")
    return history


def main():
    # -- Load data -------------------------------------------------------------
    print("\nLoading training subjects S001–S007...")
    train_epochs, train_labels, ch_names, info = load_multiple_subjects(TRAIN_SUBJECTS)

    print("\nLoading test subjects S008–S010...")
    test_epochs,  test_labels, _, _ = load_multiple_subjects(TEST_SUBJECTS)

    # Save info for later use in GradCAM and dashboard
    np.save(os.path.join(DATA_DIR, 'ch_names.npy'), np.array(ch_names))
    np.save(os.path.join(DATA_DIR, 'test_epochs.npy'),  test_epochs)
    np.save(os.path.join(DATA_DIR, 'test_labels.npy'),  test_labels)
    np.save(os.path.join(DATA_DIR, 'train_epochs.npy'), train_epochs)
    np.save(os.path.join(DATA_DIR, 'train_labels.npy'), train_labels)
    print(f"\nData saved to {DATA_DIR}/")

    n_channels = train_epochs.shape[1]
    n_times    = train_epochs.shape[2]
    print(f"  Train: {train_epochs.shape}  |  Test: {test_epochs.shape}")
    print(f"  Channels: {n_channels}, Time points: {n_times}")

    # -- Train EEGNet ----------------------------------------------------------
    eegnet_train_loader, eegnet_val_loader = make_data_loaders(
        train_epochs, train_labels, test_epochs, test_labels,
        model_type='eegnet', batch_size=BATCH_SIZE
    )
    eegnet = EEGNet(n_classes=N_CLASSES, n_channels=n_channels, n_times=n_times)
    eegnet_history = train_model(
        eegnet, 'EEGNet', eegnet_train_loader, eegnet_val_loader, 'eegnet'
    )
    np.save(os.path.join(DATA_DIR, 'eegnet_history.npy'), eegnet_history)
    save_curves(eegnet_history, 'EEGNet')

    # -- Train CNN-LSTM --------------------------------------------------------
    cnn_train_loader, cnn_val_loader = make_data_loaders(
        train_epochs, train_labels, test_epochs, test_labels,
        model_type='cnn_lstm', batch_size=BATCH_SIZE
    )
    cnnlstm = CNNLSTM(n_classes=N_CLASSES, n_channels=n_channels, n_times=n_times)
    cnn_history = train_model(
        cnnlstm, 'CNN-LSTM', cnn_train_loader, cnn_val_loader, 'cnn_lstm'
    )
    np.save(os.path.join(DATA_DIR, 'cnn_lstm_history.npy'), cnn_history)
    save_curves(cnn_history, 'CNN-LSTM')

    print("\n All training complete!")


if __name__ == "__main__":
    main()
