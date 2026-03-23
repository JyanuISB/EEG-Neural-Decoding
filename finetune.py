"""
finetune.py — Calibration curve: how accuracy scales with number of calibration epochs.

Simulates a realistic BCI deployment scenario:
  - A pre-trained model (EEGNet) is adapted on small amounts of subject-specific data.
  - We measure how quickly accuracy improves as more calibration epochs are added.
  - This tells us: "How many trials does the user need to sit through to reach X% accuracy?"

Also generates a reliability (calibration) plot showing whether the model's
confidence scores are well-calibrated (i.e., 70% confidence -> 70% accuracy).

Usage:
  python finetune.py

Outputs saved to data/:
  - calibration_curve.png     — accuracy vs. number of calibration epochs
  - reliability_curve.png     — confidence vs. actual accuracy (reliability diagram)
  - finetuned_eegnet_S{id}.pt — fine-tuned checkpoints per test subject
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve

from utils.preprocess import load_subject
from utils.dataset import EEGDataset
from models.eegnet import EEGNet

# -- Constants -----------------------------------------------------------------
SEED        = 42
TEST_SUBS   = list(range(8, 11))  # S008–S010
DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FINETUNE_LR = 5e-4
FINETUNE_EP = 20    # fine-tuning epochs per calibration batch
CLASS_NAMES = ['Left Hand', 'Right Hand']

# Number of calibration epochs to try (simulate increasing amounts of labeled data)
CALIB_SIZES = [5, 10, 15, 20, 30, 45]

np.random.seed(SEED)
torch.manual_seed(SEED)


def load_pretrained_eegnet(n_channels: int, n_times: int) -> EEGNet:
    """Load the best EEGNet checkpoint from training."""
    ckpt_path = os.path.join(DATA_DIR, 'eegnet_best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"EEGNet checkpoint not found at {ckpt_path}.\n"
            "Run python training/train.py first."
        )
    model = EEGNet(n_classes=2, n_channels=n_channels, n_times=n_times)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def get_predictions_and_probs(model, epochs: np.ndarray, model_type='eegnet'):
    """Return (predicted_labels, max_confidence) arrays for a set of epochs."""
    ds     = EEGDataset(epochs, np.zeros(len(epochs), dtype=np.int64), model_type=model_type)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

    all_preds, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_preds.append(preds)
            all_probs.append(probs)

    return np.concatenate(all_preds), np.concatenate(all_probs, axis=0)


def finetune_model(base_model: EEGNet, calib_epochs: np.ndarray,
                   calib_labels: np.ndarray) -> EEGNet:
    """
    Fine-tune EEGNet on a small set of subject-specific calibration epochs.

    We freeze Block 1 (low-level features are generic) and only update
    Block 2 + classifier (high-level, subject-specific features).
    """
    import copy
    model = copy.deepcopy(base_model).to(DEVICE)

    # Freeze Block 1 — temporal and depthwise conv layers are generic
    for param in model.block1.parameters():
        param.requires_grad = False

    # Only fine-tune Block 2 and classifier
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    ds     = EEGDataset(calib_epochs, calib_labels, model_type='eegnet')
    loader = torch.utils.data.DataLoader(ds, batch_size=min(16, len(ds)), shuffle=True)

    model.train()
    for _ in range(FINETUNE_EP):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    return model


def plot_calibration_curve(results: dict):
    """
    Plot accuracy vs. number of calibration epochs for each test subject + mean.
    """
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['steelblue', 'coral', 'seagreen']
    for i, (subj, accs) in enumerate(results['per_subject'].items()):
        ax.plot(CALIB_SIZES, accs, marker='o', label=subj,
                color=colors[i % len(colors)], linewidth=1.5)

    # Mean across subjects
    mean_accs = np.mean(list(results['per_subject'].values()), axis=0)
    ax.plot(CALIB_SIZES, mean_accs, marker='D', label='Mean',
            color='black', linewidth=2.5, linestyle='--')

    # Baseline (no fine-tuning)
    ax.axhline(results['baseline_acc'], color='gray', linestyle=':',
               linewidth=1.5, label=f'Zero-shot baseline ({results["baseline_acc"]:.2%})')

    ax.set_xlabel('Number of calibration epochs per subject')
    ax.set_ylabel('Test accuracy')
    ax.set_title('EEGNet Fine-tuning: Calibration Curve\n(Accuracy vs. Subject-Specific Training Data)')
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right')
    plt.tight_layout()

    out = os.path.join(DATA_DIR, 'calibration_curve.png')
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved calibration curve -> {out}")


def plot_reliability_diagram(all_confidences: np.ndarray, all_correct: np.ndarray):
    """
    Reliability diagram: plots model confidence vs. observed accuracy.
    A perfectly calibrated model follows the diagonal.
    """
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(5, 5))

    frac_pos, mean_pred = calibration_curve(all_correct, all_confidences,
                                            n_bins=10, strategy='uniform')

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1.5)
    ax.plot(mean_pred, frac_pos, marker='o', color='steelblue',
            linewidth=2, label='EEGNet')

    ax.set_xlabel('Mean predicted confidence')
    ax.set_ylabel('Fraction of correct predictions')
    ax.set_title('Reliability Diagram (Confidence Calibration)')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    out = os.path.join(DATA_DIR, 'reliability_curve.png')
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved reliability diagram -> {out}")


def main():
    print("\n" + "="*55)
    print("  Fine-tuning / Calibration Curve Generator")
    print("="*55)

    # -- Load base model + find data dimensions --------------------------------
    test_epochs_all = np.load(os.path.join(DATA_DIR, 'test_epochs.npy'))
    test_labels_all = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
    n_channels = test_epochs_all.shape[1]
    n_times    = test_epochs_all.shape[2]

    base_model = load_pretrained_eegnet(n_channels, n_times)
    base_model.to(DEVICE)

    # -- Zero-shot baseline (no fine-tuning) -----------------------------------
    print("\nComputing zero-shot baseline (pre-trained, no fine-tuning)...")
    preds_zero, probs_zero = get_predictions_and_probs(base_model, test_epochs_all)
    baseline_acc = accuracy_score(test_labels_all, preds_zero)
    print(f"  Zero-shot accuracy: {baseline_acc:.4f}")

    # Collect confidence and correctness for reliability diagram
    all_confidences = probs_zero.max(axis=1)
    all_correct     = (preds_zero == test_labels_all).astype(float)
    plot_reliability_diagram(all_confidences, all_correct)

    # -- Per-subject fine-tuning loop ------------------------------------------
    per_subject_accs = {}

    for sid in TEST_SUBS:
        print(f"\n  Subject S{sid:03d}:")
        try:
            ep, lb, _, _ = load_subject(sid, verbose=False)
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

        subject_accs = []
        np.random.seed(SEED)
        shuffled_idx = np.random.permutation(len(ep))

        for n_calib in CALIB_SIZES:
            if n_calib > len(ep):
                # Not enough epochs — repeat last valid accuracy
                subject_accs.append(subject_accs[-1] if subject_accs else baseline_acc)
                continue

            # Use first n_calib epochs as calibration data, rest as test
            calib_idx = shuffled_idx[:n_calib]
            test_idx  = shuffled_idx[n_calib:]

            if len(test_idx) == 0:
                subject_accs.append(subject_accs[-1] if subject_accs else baseline_acc)
                continue

            calib_ep = ep[calib_idx]
            calib_lb = lb[calib_idx]
            test_ep  = ep[test_idx]
            test_lb  = lb[test_idx]

            # Fine-tune on calibration epochs
            ft_model = finetune_model(base_model, calib_ep, calib_lb)
            preds, _ = get_predictions_and_probs(ft_model, test_ep)
            acc = accuracy_score(test_lb, preds)
            subject_accs.append(acc)

            print(f"    n_calib={n_calib:3d}  ->  acc={acc:.4f}")

            # Save the best fine-tuned model (most data = last calib size)
            if n_calib == CALIB_SIZES[-1]:
                ft_path = os.path.join(DATA_DIR, f'finetuned_eegnet_S{sid:03d}.pt')
                torch.save(ft_model.state_dict(), ft_path)

        per_subject_accs[f'S{sid:03d}'] = subject_accs

    # -- Save results + plots --------------------------------------------------
    results = {
        'per_subject': per_subject_accs,
        'baseline_acc': baseline_acc,
        'calib_sizes': CALIB_SIZES
    }
    np.save(os.path.join(DATA_DIR, 'finetune_results.npy'), results)

    plot_calibration_curve(results)

    print("\n" + "="*55)
    print("  Fine-tuning complete!")
    print(f"  Zero-shot baseline : {baseline_acc:.4f}")
    if per_subject_accs:
        max_calib_accs = [v[-1] for v in per_subject_accs.values()]
        print(f"  After {CALIB_SIZES[-1]} calib epochs : {np.mean(max_calib_accs):.4f} (mean)")
    print("="*55)


if __name__ == "__main__":
    main()
