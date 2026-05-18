new_model = '''import torch
import torch.nn as nn


class CSLRModel(nn.Module):

    I3D_FEATURE_DIM = 1024

    def __init__(self, vocab_size, hidden_size=1024,
                 num_layers=3, dropout=0.3, use_keypoints=False):
        super().__init__()

        self.feat_proj = nn.Sequential(
            nn.Linear(self.I3D_FEATURE_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.ctc_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )

    def freeze_backbone(self, freeze=True):
        pass

    def forward(self, frames, keypoints=None):
        feats = self.feat_proj(frames)
        lstm_out, _ = self.bilstm(feats)
        logits = self.ctc_head(lstm_out)
        return logits.log_softmax(dim=-1)
'''

import os
os.chdir("/kaggle/working/Sign-Language-Recognition")

with open("src/models/ctc_model.py", "wb") as f:
    f.write(new_model.encode("utf-8"))

print("✅ ctc_model.py fixed!")