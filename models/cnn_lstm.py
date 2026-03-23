"""
cnn_lstm.py — CNN-LSTM hybrid model for EEG motor imagery classification.

Architecture:
  - 1D convolutional layers extract spatial features from each time step
  - LSTM layers capture temporal dynamics across the sequence
  - Fully connected output layer for class prediction

Input shape: (batch, n_channels, n_times)
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN-LSTM model for EEG time-series classification.

    Parameters
    ----------
    n_classes  : int   — number of output classes (default 2)
    n_channels : int   — number of EEG input channels (default 64)
    n_times    : int   — number of time samples (default 481)
    cnn_filters: list  — number of filters per Conv1d layer
    lstm_hidden: int   — number of hidden units per LSTM layer
    lstm_layers: int   — number of stacked LSTM layers
    dropout    : float — dropout rate
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 64,
        n_times: int = 481,
        cnn_filters: list = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super(CNNLSTM, self).__init__()

        if cnn_filters is None:
            cnn_filters = [64, 128]  # two conv layers by default

        # -- CNN Feature Extractor ---------------------------------------------
        # Conv1d treats the channel axis as "features" and time as the sequence.
        # Each conv layer slides along the time axis to detect local patterns.
        cnn_layers = []
        in_ch = n_channels
        for out_ch in cnn_filters:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2)   # halve the time dimension
            ]
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # Compute CNN output time length after pooling
        cnn_out_len = n_times
        for _ in cnn_filters:
            cnn_out_len = cnn_out_len // 2

        # -- LSTM Temporal Encoder ---------------------------------------------
        # After CNN, data shape is (batch, cnn_filters[-1], cnn_out_len).
        # LSTM expects (seq_len, batch, features), so we'll permute in forward().
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],   # feature dim from CNN
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,              # input: (batch, seq_len, features)
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False
        )

        # -- Classifier --------------------------------------------------------
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_channels, n_times)

        Returns
        -------
        logits : torch.Tensor, shape (batch, n_classes)
        """
        # CNN feature extraction: (batch, n_channels, n_times) -> (batch, C', T')
        x = self.cnn(x)

        # Reshape for LSTM: (batch, C', T') -> (batch, T', C')
        # LSTM processes T' time steps, each with C' features
        x = x.permute(0, 2, 1)

        # LSTM temporal encoding; we use only the last hidden state
        lstm_out, (h_n, _) = self.lstm(x)
        # h_n shape: (n_layers, batch, hidden) — take the last layer's output
        x = h_n[-1]  # (batch, hidden)

        # Dropout + classification
        x = self.dropout(x)
        x = self.fc(x)
        return x  # raw logits


if __name__ == "__main__":
    model = CNNLSTM(n_classes=2, n_channels=64, n_times=481)
    print(model)

    x = torch.randn(8, 64, 481)  # batch=8
    out = model(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")  # expect (8, 2)
    print("CNN-LSTM forward pass PASSED.")
