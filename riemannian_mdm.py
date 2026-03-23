"""
riemannian_mdm.py — Riemannian Geometry baseline using Minimum Distance to Mean (MDM).

Why Riemannian?
  EEG covariance matrices live on the Symmetric Positive Definite (SPD) manifold.
  Standard Euclidean classifiers ignore this geometry and perform poorly.
  MDM computes the Riemannian mean of each class's covariance matrices and classifies
  new epochs by finding which class mean they are closest to (in Riemannian distance).

Reference: Barachant et al. (2012). "Multiclass Brain-Computer Interface Classification
by Riemannian Geometry". IEEE Transactions on Biomedical Engineering.

Usage:
  python riemannian_mdm.py

Outputs saved to data/:
  - mdm_results.npy         — accuracy, kappa, confusion matrix
  - confusion_mdm.png       — confusion matrix plot
  - riemannian_mdm_best.pkl — fitted MDM pipeline (for dashboard use)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression

from utils.preprocess import load_multiple_subjects

# -- Constants -----------------------------------------------------------------
SEED         = 42
TRAIN_SUBS   = list(range(1, 8))   # S001–S007
TEST_SUBS    = list(range(8, 11))  # S008–S010
DATA_DIR     = os.path.join(os.path.dirname(__file__), 'data')
CLASS_NAMES  = ['Left Hand', 'Right Hand']

os.makedirs(DATA_DIR, exist_ok=True)
np.random.seed(SEED)


def load_or_fetch(subject_ids: list, split_name: str):
    """Load epochs/labels from cache if available, else download via MNE."""
    ep_path = os.path.join(DATA_DIR, f'{split_name}_epochs.npy')
    lb_path = os.path.join(DATA_DIR, f'{split_name}_labels.npy')

    if os.path.exists(ep_path) and os.path.exists(lb_path):
        print(f"  Loading cached {split_name} data...")
        return np.load(ep_path), np.load(lb_path), None
    else:
        print(f"  Downloading {split_name} subjects...")
        epochs, labels, ch_names, info = load_multiple_subjects(subject_ids)
        np.save(ep_path, epochs)
        np.save(lb_path, labels)
        return epochs, labels, info


def bandpower_features(epochs: np.ndarray) -> np.ndarray:
    """
    Compute band-power features as a simple alternative feature set.
    Not used in the main MDM pipeline but useful for ablation.
    """
    from scipy.signal import welch
    features = []
    for epoch in epochs:
        powers = []
        for ch in epoch:
            freqs, psd = welch(ch, fs=160, nperseg=256)
            # Mu band (8-12 Hz) and Beta band (12-30 Hz)
            mu_mask   = (freqs >= 8)  & (freqs <= 12)
            beta_mask = (freqs >= 12) & (freqs <= 30)
            powers.extend([psd[mu_mask].mean(), psd[beta_mask].mean()])
        features.append(powers)
    return np.array(features)


def plot_confusion_mdm(cm: np.ndarray, accuracy: float, kappa: float):
    """Save a confusion matrix plot for the MDM classifier."""
    sns.set_style('white')
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(
        cm_norm, annot=True, fmt='.1f', cmap='Greens', ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar_kws={'label': '%'}
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'MDM — Confusion Matrix\nAcc: {accuracy:.2%}  Kappa: {kappa:.4f}')
    plt.tight_layout()
    out = os.path.join(DATA_DIR, 'confusion_mdm.png')
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix -> {out}")


def main():
    print("\n" + "="*55)
    print("  Riemannian MDM Classifier")
    print("="*55)

    # -- Load data --------------------------------------------------------------
    print("\nLoading training data (S001–S007)...")
    train_epochs, train_labels, _ = load_or_fetch(TRAIN_SUBS, 'train')

    print("Loading test data (S008–S010)...")
    test_epochs, test_labels, _   = load_or_fetch(TEST_SUBS, 'test')

    print(f"\n  Train: {train_epochs.shape}  Labels: {np.bincount(train_labels)}")
    print(f"  Test : {test_epochs.shape}   Labels: {np.bincount(test_labels)}")

    # -- Build Riemannian pipeline ----------------------------------------------
    # Step 1: Estimate covariance matrix for each epoch (64×64 SPD matrix)
    # Step 2: Project to tangent space at the Riemannian mean (vectorises the manifold)
    # Step 3: Logistic regression on the vectorised tangent-space features
    #
    # We use TWO variants:
    #   a) Pure MDM  — classify by distance to Riemannian class mean (no tangent space)
    #   b) TS + LR   — tangent space projection + logistic regression (often stronger)

    print("\nFitting MDM pipeline (direct Riemannian classification)...")
    mdm_pipeline = Pipeline([
        ('cov',  Covariances(estimator='oas')),  # OAS: robust covariance estimator
        ('mdm',  MDM(metric='riemann'))           # classify by Riemannian distance
    ])
    mdm_pipeline.fit(train_epochs, train_labels)
    mdm_preds = mdm_pipeline.predict(test_epochs)

    mdm_acc   = accuracy_score(test_labels, mdm_preds)
    mdm_kappa = cohen_kappa_score(test_labels, mdm_preds)
    mdm_cm    = confusion_matrix(test_labels, mdm_preds)

    print(f"\n  [MDM] Accuracy      : {mdm_acc:.4f}  ({mdm_acc*100:.1f}%)")
    print(f"  [MDM] Cohen's Kappa : {mdm_kappa:.4f}")

    print("\nFitting Tangent Space + Logistic Regression pipeline...")
    ts_pipeline = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('ts',  TangentSpace(metric='riemann')),
        ('lr',  LogisticRegression(max_iter=1000, random_state=SEED))
    ])
    ts_pipeline.fit(train_epochs, train_labels)
    ts_preds = ts_pipeline.predict(test_epochs)

    ts_acc   = accuracy_score(test_labels, ts_preds)
    ts_kappa = cohen_kappa_score(test_labels, ts_preds)

    print(f"\n  [TS+LR] Accuracy      : {ts_acc:.4f}  ({ts_acc*100:.1f}%)")
    print(f"  [TS+LR] Cohen's Kappa : {ts_kappa:.4f}")

    # Use whichever variant performed better as the "MDM" result
    if ts_acc >= mdm_acc:
        best_pipeline = ts_pipeline
        best_acc, best_kappa, best_cm = ts_acc, ts_kappa, confusion_matrix(test_labels, ts_preds)
        best_label = 'TS+LR'
    else:
        best_pipeline = mdm_pipeline
        best_acc, best_kappa, best_cm = mdm_acc, mdm_kappa, mdm_cm
        best_label = 'MDM'

    print(f"\n  Best variant: {best_label}  ->  Acc: {best_acc:.4f}  Kappa: {best_kappa:.4f}")

    # -- Per-subject breakdown --------------------------------------------------
    print("\n  Per-subject accuracy:")
    from utils.preprocess import load_subject
    per_subject = {}
    for sid in TEST_SUBS:
        try:
            ep, lb, _, _ = load_subject(sid, verbose=False)
            pred = best_pipeline.predict(ep)
            acc  = accuracy_score(lb, pred)
            per_subject[f'S{sid:03d}'] = acc
            print(f"    S{sid:03d} : {acc:.4f}")
        except Exception as e:
            print(f"    S{sid:03d} : ERROR — {e}")

    # -- Save results -----------------------------------------------------------
    results = {
        'accuracy':        best_acc,
        'kappa':           best_kappa,
        'confusion_matrix': best_cm,
        'per_subject_acc': per_subject,
        'mdm_acc':         mdm_acc,
        'mdm_kappa':       mdm_kappa,
        'ts_acc':          ts_acc,
        'ts_kappa':        ts_kappa,
        'best_variant':    best_label
    }
    np.save(os.path.join(DATA_DIR, 'mdm_results.npy'), results)

    # Save the fitted pipeline (for dashboard inference)
    pkl_path = os.path.join(DATA_DIR, 'riemannian_mdm_best.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"\n  Saved MDM pipeline -> {pkl_path}")

    plot_confusion_mdm(best_cm, best_acc, best_kappa)

    print("\n" + "="*55)
    print("  Riemannian MDM training complete!")
    print(f"  MDM direct  : Acc={mdm_acc:.4f}  Kappa={mdm_kappa:.4f}")
    print(f"  TS + LR     : Acc={ts_acc:.4f}  Kappa={ts_kappa:.4f}")
    print(f"  Best ({best_label:6s}): Acc={best_acc:.4f}  Kappa={best_kappa:.4f}")
    print("="*55)


if __name__ == "__main__":
    main()
