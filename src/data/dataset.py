import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PhoenixDataset(Dataset):
    """
    Dataset using pre-extracted i3d .npy features + translation labels.
    Works with RWTH-PHOENIX-2014T i3d + mediapipe Kaggle dataset.

    Expected data structure (on Kaggle):
    - i3d .npy features: i3d_features_rwth phoenix 2014t/{train,dev,test}/<id>.npy
    - Annotations: tsv files/{train,val,test}.how2sign.tsv

    Each .npy file has shape (T, 1024) where T = number of frames.
    Labels are word-level tokens from the German translation column.
    """

    I3D_BASE = (
        "/kaggle/input/datasets/rabeyaakter23/"
        "rwth-phoenix-2014t-i3d-features-mediapipe-features/"
        "i3d_features_rwth phoenix 2014t/"
        "i3d_features_rwth phoenix 2014t"
    )
    TSV_BASE = (
        "/kaggle/input/datasets/rabeyaakter23/"
        "rwth-phoenix-2014t-i3d-features-mediapipe-features/"
        "tsv files_rwth phoenix 2014t/tsv files"
    )
    SPLIT_MAP = {
        "train": "cvpr23.fairseq.i3d.train.how2sign.tsv",
        "val":   "cvpr23.fairseq.i3d.val.how2sign.tsv",
        "test":  "cvpr23.fairseq.i3d.test.how2sign.tsv",
    }

    def __init__(self, root_dir=None, split="train",
                 max_frames=300, img_size=224, use_keypoints=False):
        """
        Args:
            root_dir:      ignored (paths are resolved from Kaggle input)
            split:         "train", "val", or "test"
            max_frames:    max frames to keep per sample (truncates longer)
            img_size:      unused (kept for API compatibility)
            use_keypoints: unused (kept for API compatibility)
        """
        self.split      = split
        self.max_frames = max_frames

        # Load annotation TSV
        tsv_path = os.path.join(self.TSV_BASE, self.SPLIT_MAP[split])
        self.data = pd.read_csv(tsv_path, sep="\t")
        self.data = self.data[self.data["translation"].notna()].reset_index(drop=True)

        # Build vocabulary from translations (word-level tokens)
        # Index 0 = <blank> reserved for CTC loss
        # Index 1 = <unk> for out-of-vocabulary words at inference
        # OLD — word level (causes 99% WER)
        all_words = " ".join(self.data["translation"].tolist()).split()
        vocab = sorted(set(all_words))
        self.gloss2idx = {"<blank>": 0, "<unk>": 1}
        self.gloss2idx.update({w: i + 2 for i, w in enumerate(vocab)})

        # NEW — character level (fixes the mismatch problem)
        all_chars = set(" ".join(self.data["translation"].tolist()))
        vocab = sorted(all_chars)
        self.gloss2idx = {"<blank>": 0, "<unk>": 1, " ": 2}
        self.gloss2idx.update({c: i + 3 for i, c in enumerate(vocab) if c != " "})
        self.idx2gloss = {v: k for k, v in self.gloss2idx.items()}
        
        # i3d feature folder for this split
        self.npy_dir = os.path.join(self.I3D_BASE, split)

        print(f"[{split}] Loaded {len(self.data)} samples | vocab size: {len(self.gloss2idx)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load pre-extracted i3d features → (T, 1024)
        npy_path = os.path.join(self.npy_dir, row["id"] + ".npy")
        features = np.load(npy_path)

        # Truncate to max_frames if needed
        if features.shape[0] > self.max_frames:
            features = features[:self.max_frames]

        frames_tensor = torch.tensor(features, dtype=torch.float32)  # (T, 1024)

        # Tokenize translation into integer label sequence
        chars = list(row["translation"].strip())
        label = torch.tensor(
                [self.gloss2idx.get(c, 1) for c in chars],
                dtype=torch.long
)

        # Return (frames, None, label) — None keeps collate_fn signature intact
        return frames_tensor, None, label


def collate_fn(batch):
    """
    Custom collate for variable-length i3d sequences.
    Pads frames to same length; concatenates labels for CTC loss.
    """
    frames, _, labels = zip(*batch)

    frame_lengths = torch.tensor([f.shape[0] for f in frames])
    label_lengths = torch.tensor([l.shape[0] for l in labels])
    frames_padded = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)
    labels_concat = torch.cat(labels)

    return frames_padded, None, labels_concat, frame_lengths, label_lengths