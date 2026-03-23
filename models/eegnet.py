"""
eegnet.py — EEGNet model in PyTorch.

Reference: Lawhern et al., "EEGNet: A Compact Convolutional Neural Network
for EEG-based Brain-Computer Interfaces", J. Neural Engineering, 2018.

Architecture overview:
  Block 1 -> Temporal convolution + Depthwise spatial convolution
  Block 2 -> Separable convolution (depthwise + pointwise)
  Classifier -> Flatten -> Dense -> Softmax

Input shape: (batch, 1, n_channels, n_times)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet: compact CNN for EEG classification.

    Parameters
    ----------
    n_classes  : int   — number of output classes (default 2)
    n_channels : int   — number of EEG channels (default 64)
    n_times    : int   — number of time samples per epoch (default 481 = 3s @ 160Hz)
    F1         : int   — number of temporal filters (default 8)
    D          : int   — depth multiplier for depthwise conv (default 2)
    F2         : int   — number of pointwise filters in Block 2 (= F1 * D = 16)
    dropout    : float — dropout rate (default 0.5)
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 64,
        n_times: int = 481,
        F1: int = 8,
        D: int = 2,
        dropout: float = 0.5
    ):
        super(EEGNet, self).__init__()

        F2 = F1 * D  # number of separable conv filters = 16

        # -- Block 1: Temporal Convolution ------------------------------------
        # Learn frequency-specific features across time.
        # Kernel width = sampling_rate / 2 = 64 samples for 160 Hz data.
        self.block1 = nn.Sequential(
            # Temporal conv: operates along the time axis only (kernel size 1 × 64)
            # padding='same' keeps the time dimension unchanged
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),

            # Depthwise conv: each temporal filter learns a spatial (channel) filter
            # groups=F1 means each input channel is processed independently
            # D filters per channel -> F1*D = 16 feature maps total
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),

            # Average pooling: downsample by 4× along time axis
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout)
        )

        # -- Block 2: Separable Convolution -----------------------------------
        # Separable conv = depthwise + pointwise (approximates full conv cheaply).
        # Learns how to combine the features from Block 1 across time.
        self.block2 = nn.Sequential(
            # Depthwise: kernel (1 × 16) operates on each feature map independently
            nn.Conv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8),
                      groups=F2, bias=False),
            # Pointwise: 1×1 conv to mix feature maps
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),

            # Average pooling: downsample by 8× along time axis
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout)
        )

        # -- Classifier -------------------------------------------------------
        # Dynamically compute the flattened feature size by doing a forward pass
        # with a dummy input so we don't hard-code it
        self._feature_size = self._compute_feature_size(n_channels, n_times, F1, D)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size, n_classes)
        )

    def _compute_feature_size(self, n_channels, n_times, F1, D):
        """Run a dummy tensor through blocks 1 & 2 to get flattened size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.block1(dummy)
            out = self.block2(out)
            return out.numel()  # total elements after blocks

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 1, n_channels, n_times)

        Returns
        -------
        logits : torch.Tensor, shape (batch, n_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x  # raw logits (CrossEntropyLoss applies softmax internally)

    def get_last_conv_layer(self):
        """Return the last Conv2d layer — used as GradCAM target."""
        # The last conv layer is the pointwise conv inside block2
        return self.block2[1]


if __name__ == "__main__":
    model = EEGNet(n_classes=2, n_channels=64, n_times=481)
    print(model)

    # Test forward pass
    x = torch.randn(8, 1, 64, 481)  # batch=8
    out = model(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")  # expect (8, 2)
    print("EEGNet forward pass PASSED.")
