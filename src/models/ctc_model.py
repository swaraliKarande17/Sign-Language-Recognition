import torch	
import torch.nn as nn	
from torchvision.models import resnet50, ResNet50_Weights	
	
	
class CSLRModel(nn.Module):	
    """	
    Continuous Sign Language Recognition Model.	
	
    Architecture:	
        ResNet-50 CNN (pretrained) → feature projection	
        MediaPipe keypoints → keypoint projection (optional)	
        Feature fusion (concat + linear)	
        Bidirectional LSTM (2 layers)	
        CTC output head	
	
    Inputs:	
        frames:    (batch, T, 3, 224, 224)  — video frames tensor	
        keypoints: (batch, T, 258)          — MediaPipe features (optional)	
	
    Output:	
        log_probs: (batch, T, vocab_size)   — log probabilities per timestep	
    """	
	
    def __init__(self, vocab_size, hidden_size=512,	
                 num_layers=2, dropout=0.3, use_keypoints=True):	
        super().__init__()	
        self.use_keypoints = use_keypoints	
	
        # ── 1. CNN Backbone ──────────────────────────────────────────	
        # Load pretrained ResNet-50 (trained on ImageNet)	
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)	
        # Remove the last classification layer — we want features only	
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])	
        # Project from 2048 (ResNet output) to hidden_size	
        self.cnn_proj = nn.Sequential(	
            nn.Linear(2048, hidden_size),	
            nn.LayerNorm(hidden_size),	
            nn.ReLU(),	
            nn.Dropout(dropout)	
        )	
	
        # ── 2. Keypoint Branch ───────────────────────────────────────	
        if use_keypoints:	
            # Project 258-dim keypoints to 256 dimensions	
            self.kp_proj = nn.Sequential(	
                nn.Linear(258, 256),	
                nn.LayerNorm(256),	
                nn.ReLU(),	
                nn.Dropout(dropout)	
            )	
            # Fuse CNN features (512) + keypoint features (256) -> 512	
            self.fusion = nn.Sequential(	
                nn.Linear(hidden_size + 256, hidden_size),	
                nn.LayerNorm(hidden_size),	
                nn.ReLU()	
            )	
	
        # ── 3. Temporal model — Bidirectional LSTM ───────────────────	
        # bidirectional=True means hidden_size//2 per direction x 2 = hidden_size	
        self.bilstm = nn.LSTM(	
            input_size=hidden_size,	
            hidden_size=hidden_size // 2,	
            num_layers=num_layers,	
            bidirectional=True,	
            dropout=dropout if num_layers > 1 else 0,	
            batch_first=True	
        )	
	
        # ── 4. CTC Output Head ───────────────────────────────────────	
        self.ctc_head = nn.Sequential(	
            nn.Dropout(dropout),	
            nn.Linear(hidden_size, vocab_size)	
        )	
	
    def freeze_backbone(self, freeze=True):	
        """Freeze or unfreeze the CNN backbone weights.	
        We freeze it at the start and unfreeze after 10 epochs	
        so the BiLSTM learns first, then fine-tune everything together.	
        """	
        for param in self.cnn.parameters():	
            param.requires_grad = not freeze	
	
    def forward(self, frames, keypoints=None):	
        """	
        Forward pass through the entire model.	
	
        frames:    tensor (B, T, C, H, W)	
        keypoints: tensor (B, T, 258) or None	
        """	
        B, T, C, H, W = frames.shape	
	
        # Process all T frames at once through CNN	
        # Reshape: merge batch and time -> (B*T, C, H, W)	
        frames_flat = frames.view(B * T, C, H, W)	
        cnn_out = self.cnn(frames_flat)              # (B*T, 2048, 1, 1)	
        cnn_out = cnn_out.squeeze(-1).squeeze(-1)   # (B*T, 2048)	
        cnn_feats = self.cnn_proj(cnn_out)           # (B*T, hidden)	
        cnn_feats = cnn_feats.view(B, T, -1)         # (B, T, hidden)	
	
        # Fuse with keypoints if provided	
        if self.use_keypoints and keypoints is not None:	
            kp_feats = self.kp_proj(keypoints)       # (B, T, 256)	
            combined = torch.cat([cnn_feats, kp_feats], dim=-1)	
            feats = self.fusion(combined)             # (B, T, hidden)	
        else:	
            feats = cnn_feats	
	
        # Temporal modeling with BiLSTM	
        lstm_out, _ = self.bilstm(feats)             # (B, T, hidden)	
	
        # CTC output logits	
        logits = self.ctc_head(lstm_out)             # (B, T, vocab_size)	
	
        # log_softmax is required as input for nn.CTCLoss	
        return logits.log_softmax(dim=-1)	