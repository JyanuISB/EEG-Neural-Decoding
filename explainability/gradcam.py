"""
gradcam.py — GradCAM explainability for EEGNet.

GradCAM (Gradient-weighted Class Activation Mapping) highlights which parts
of the input most influenced the model's prediction. Here we adapt it to EEG:
the output is a per-channel importance score (one value per electrode).

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.

Usage:
  from explainability.gradcam import GradCAM, plot_topomap
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import warnings
warnings.filterwarnings('ignore')


class GradCAM:
    """
    GradCAM adapted for EEGNet's 2D convolutional architecture.

    Hooks into the target layer, computes gradient of the predicted class
    score w.r.t. the feature maps, then pools spatially to get channel scores.

    Parameters
    ----------
    model      : EEGNet instance (already loaded, in eval mode)
    target_layer : nn.Module — the conv layer to target (e.g. model.get_last_conv_layer())
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        # Storage for forward activations and backward gradients
        self._activations = None
        self._gradients   = None

        # Register forward and backward hooks on the target layer
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook: store the layer's output activations during the forward pass."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook: store the gradients flowing back through the layer."""
        self._gradients = grad_output[0].detach()

    def compute(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Compute GradCAM channel-importance scores for one epoch.

        Parameters
        ----------
        input_tensor : torch.Tensor, shape (1, 1, n_channels, n_times)
            Single EEG epoch (already normalised), with batch dim added.
        class_idx : int or None
            Target class to explain. If None, uses the predicted class.

        Returns
        -------
        channel_scores : np.ndarray, shape (n_channels,)
            Non-negative importance score per EEG electrode.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # -- Forward pass -----------------------------------------------------
        logits = self.model(input_tensor)   # (1, n_classes)
        probs  = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # -- Backward pass for target class -----------------------------------
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # -- Compute GradCAM ---------------------------------------------------
        # activations: (1, n_filters, 1, T') — feature maps from target layer
        # gradients  : (1, n_filters, 1, T') — gradient of score w.r.t. maps
        acts  = self._activations  # (1, F, 1, T')
        grads = self._gradients    # (1, F, 1, T')

        # Global average pooling over spatial dims -> importance weight per filter
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, F, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=True)   # (1, 1, 1, T')
        cam = F.relu(cam)                                  # keep positive contributions

        # -- Map back to EEG channels ------------------------------------------
        # The depthwise conv in Block 1 operates per-channel, so we can extract
        # per-channel importance from the Block 1 depthwise layer's activations.
        # As a proxy, we use the absolute value of the input gradient averaged
        # over time, which gives a direct per-channel sensitivity measure.
        input_grad = input_tensor.grad  # (1, 1, n_channels, n_times)
        if input_grad is not None:
            channel_scores = input_grad.abs().mean(dim=(0, 1, 3)).cpu().numpy()
        else:
            # Fallback: replicate CAM score to all channels
            channel_scores = np.ones(input_tensor.shape[2])

        # Normalise to [0, 1]
        if channel_scores.max() > 0:
            channel_scores = channel_scores / channel_scores.max()

        return channel_scores, class_idx, probs[0].detach().cpu().numpy()

    def remove_hooks(self):
        """Clean up hooks to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def plot_topomap(channel_scores: np.ndarray, info: mne.Info,
                 class_idx: int, save_path: str = None,
                 title: str = None) -> str:
    """
    Visualise per-channel GradCAM scores as a scalp topographic map using MNE.

    Parameters
    ----------
    channel_scores : np.ndarray, shape (n_channels,)
    info           : mne.Info — contains electrode positions
    class_idx      : int — 0=left hand, 1=right hand
    save_path      : str — if provided, save the figure here
    title          : str — custom title; defaults to class name

    Returns
    -------
    save_path : str — path where the image was saved (or None)
    """
    class_names = ['Left Hand', 'Right Hand']
    if title is None:
        title = f'GradCAM — {class_names[class_idx]}'

    fig, ax = plt.subplots(figsize=(5, 5))

    # mne.viz.plot_topomap draws the scalp map
    im, cn = mne.viz.plot_topomap(
        channel_scores,
        info,
        axes=ax,
        show=False,
        cmap='RdBu_r',
        vlim=(0, 1),
        contours=4,
        sensors=True
    )

    plt.colorbar(im, ax=ax, label='Importance (0–1)')
    ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"  Saved topomap -> {save_path}")

    plt.close()
    return save_path


if __name__ == "__main__":
    # Demo: load EEGNet, compute GradCAM on first test epoch
    import numpy as np
    from models.eegnet import EEGNet
    from utils.dataset import EEGDataset
    from utils.preprocess import load_subject

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    DEVICE   = torch.device('cpu')

    # Load test data
    test_epochs = np.load(os.path.join(DATA_DIR, 'test_epochs.npy'))
    test_labels = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))

    n_channels = test_epochs.shape[1]
    n_times    = test_epochs.shape[2]

    # Load model
    model = EEGNet(n_classes=2, n_channels=n_channels, n_times=n_times)
    ckpt  = torch.load(os.path.join(DATA_DIR, 'eegnet_best.pt'), map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Prepare one epoch
    ds = EEGDataset(test_epochs[:1], test_labels[:1], model_type='eegnet')
    x, y = ds[0]
    x = x.unsqueeze(0)  # add batch dim -> (1, 1, C, T)

    # Compute GradCAM
    gradcam = GradCAM(model, model.get_last_conv_layer())
    scores, pred_class, probs = gradcam.compute(x)
    gradcam.remove_hooks()

    print(f"Predicted class: {pred_class} ({['Left','Right'][pred_class]})")
    print(f"Confidence: {probs[pred_class]:.4f}")
    print(f"Top 5 channels: {np.argsort(scores)[-5:][::-1]}")

    # Load MNE info for topomap
    _, _, ch_names, info = load_subject(8, verbose=False)
    save_path = os.path.join(DATA_DIR, 'gradcam_topomap.png')
    plot_topomap(scores, info, pred_class, save_path)
    print("GradCAM demo complete.")
