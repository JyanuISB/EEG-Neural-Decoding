"""
evaluate.py -- Load saved model checkpoints and compute evaluation metrics.

Metrics computed:
  - Overall accuracy
  - Cohen's kappa coefficient
  - Confusion matrix (plotted)
  - Per-subject accuracy breakdown

Usage:
  python training/evaluate.py

Outputs saved to data/:
  - confusion_eegnet.png
  - confusion_cnn_lstm.png
  - per_subject_accuracy.png
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from utils.preprocess import load_subject
from utils.dataset import EEGDataset
from models.eegnet import EEGNet
from models.cnn_lstm import CNNLSTM

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
TEST_SUBS  = list(range(8, 11))   # S008, S009, S010
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_type: str, n_channels: int, n_times: int):
    """Load a saved model checkpoint from data/."""
    ckpt_path = os.path.join(DATA_DIR, f'{model_type}_best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun train.py first.")

    if model_type == 'eegnet':
        model = EEGNet(n_classes=2, n_channels=n_channels, n_times=n_times)
    else:
        model = CNNLSTM(n_classes=2, n_channels=n_channels, n_times=n_times)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def get_predictions(model, epochs: np.ndarray, model_type: str) -> np.ndarray:
    """Run inference on a numpy array of epochs, return predicted labels."""
    dataset = EEGDataset(epochs, np.zeros(len(epochs), dtype=np.int64), model_type=model_type)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    return np.concatenate(all_preds)


def plot_confusion(cm: np.ndarray, model_name: str):
    """Plot and save a normalised confusion matrix heatmap."""
    sns.set_style('white')
    fig, ax = plt.subplots(figsize=(5, 4))

    # Normalise to percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(
        cm_norm, annot=True, fmt='.1f',
        cmap='Blues', ax=ax,
        xticklabels=['Left Hand', 'Right Hand'],
        yticklabels=['Left Hand', 'Right Hand'],
        cbar_kws={'label': '%'}
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{model_name} -- Confusion Matrix')

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    out_path = os.path.join(DATA_DIR, f'confusion_{safe_name}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix -> {out_path}")
    return out_path


def evaluate_model(model, model_type: str, model_name: str):
    """
    Evaluate a model on the test subjects and print metrics.

    Returns
    -------
    dict with keys: accuracy, kappa, confusion_matrix, per_subject_acc
    """
    print(f"\n{'-'*45}")
    print(f"  Evaluating: {model_name}")

    all_true, all_pred = [], []
    per_subject = {}

    for sid in TEST_SUBS:
        try:
            epochs, labels, _, _ = load_subject(sid, verbose=False)
            preds = get_predictions(model, epochs, model_type)

            acc = accuracy_score(labels, preds)
            per_subject[f'S{sid:03d}'] = acc

            all_true.extend(labels.tolist())
            all_pred.extend(preds.tolist())
        except Exception as e:
            print(f"  [WARNING] Could not evaluate S{sid:03d}: {e}")

    if len(all_true) == 0:
        print("  No predictions generated.")
        return {}

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    acc   = accuracy_score(all_true, all_pred)
    kappa = cohen_kappa_score(all_true, all_pred)
    cm    = confusion_matrix(all_true, all_pred)

    print(f"  Accuracy      : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Cohen's Kappa : {kappa:.4f}")
    print(f"  Per-subject   :")
    for subj, a in per_subject.items():
        print(f"    {subj} : {a:.4f}")

    plot_confusion(cm, model_name)

    return {
        'accuracy': acc,
        'kappa': kappa,
        'confusion_matrix': cm,
        'per_subject_acc': per_subject
    }


def plot_per_subject_comparison(eegnet_results: dict, cnn_results: dict):
    """Bar chart comparing per-subject accuracy for both models."""
    subjects = list(eegnet_results['per_subject_acc'].keys())
    eegnet_accs = [eegnet_results['per_subject_acc'][s] for s in subjects]
    cnn_accs    = [cnn_results['per_subject_acc'][s]    for s in subjects]

    x = np.arange(len(subjects))
    width = 0.35

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, eegnet_accs, width, label='EEGNet',   color='steelblue')
    bars2 = ax.bar(x + width/2, cnn_accs,    width, label='CNN-LSTM', color='coral')

    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-Subject Test Accuracy Comparison')
    ax.legend()
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')

    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, 'per_subject_accuracy.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved per-subject plot -> {out_path}")


def main():
    # Load data dims from saved arrays
    test_epochs = np.load(os.path.join(DATA_DIR, 'test_epochs.npy'))
    n_channels  = test_epochs.shape[1]
    n_times     = test_epochs.shape[2]

    try:
        eegnet  = load_model('eegnet',   n_channels, n_times)
        cnnlstm = load_model('cnn_lstm', n_channels, n_times)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    eegnet_results = evaluate_model(eegnet,  'eegnet',   'EEGNet')
    cnn_results    = evaluate_model(cnnlstm, 'cnn_lstm', 'CNN-LSTM')

    if eegnet_results and cnn_results:
        plot_per_subject_comparison(eegnet_results, cnn_results)

    print("\n Summary")
    print(f"  EEGNet   -- Acc: {eegnet_results.get('accuracy',0):.4f}  Kappa: {eegnet_results.get('kappa',0):.4f}")
    print(f"  CNN-LSTM -- Acc: {cnn_results.get('accuracy',0):.4f}  Kappa: {cnn_results.get('kappa',0):.4f}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
