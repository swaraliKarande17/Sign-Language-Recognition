import torch
import torch.nn as nn


class CSLRModel(nn.Module):
    """
    Continuous Sign Language Recognition Model.

    Architecture:
        Pre-extracted i3d features (1024-dim) -> feature projection
        Bidirectional LSTM (2 layers)
        CTC output head

    Inputs:
        frames:    (batch, T, 1024)  — pre-extracted i3d features
        keypoints: ignored (kept for API compatibility)

    Output:
        log_probs: (batch, T, vocab_size) — log probabilities per timestep
    """

    I3D_FEATURE_DIM = 1024

    def __init__(self, vocab_size, hidden_size=512,
                 num_layers=2, dropout=0.3, use_keypoints=False):
        super().__init__()

        # ── 1. i3d Feature Projection ────────────────────────────────
        # Project from 1024 (i3d output) to hidden_size
        self.feat_proj = nn.Sequential(
            nn.Linear(self.I3D_FEATURE_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)