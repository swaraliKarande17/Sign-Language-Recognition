import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PhoenixDataset(Dataset):

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
        self.split      = split
        self.max_frames = max_frames

        # ── Build vocab from ALL splits so train/val/test share same vocab ──
        # This fixes the train(2889 words) vs val(953 words) mismatch
        all_words = set()
        for s, fname in self.SPLIT_MAP.items():
            tsv_path = os.path.join(self.TSV_BASE, fname)
            df_tmp = pd.read_csv(tsv_path, sep="\t")
            df_tmp = df_tmp[df_tmp["translation"].notna()]
            for text in df_tmp["translation"].tolist():
                all_words.update(text.strip().split())

        vocab = sorted(all_words)
        self.gloss2idx = {"<blank>": 0, "<unk>": 1}
        self.gloss2idx.update({w: i + 2 for i, w in enumerate(vocab)})
        self.idx2gloss = {v: k for k, v in self.gloss2idx.items()}

        # ── Load this split ──────────────────────────────────────────
        tsv_path = os.path.join(self.TSV_BASE, self.SPLIT_MAP[split])
        self.data = pd.read_csv(tsv_path, sep="\t")
        self.data = self.data[self.data["translation"].notna()].reset_index(drop=True)

        # ── i3d feature folder ───────────────────────────────────────
        npy_split = "val" if split == "val" else split
        self.npy_dir = os.path.join(self.I3D_BASE, npy_split)

        # ── Filter bad samples where T < num_words (CTC requirement) ─
        # CTC needs at least as many frames as label tokens
        valid_indices = []
        for i, row in self.data.iterrows():
            npy_path = os.path.join(self.npy_dir, row["id"] + ".npy")
            if not os.path.exists(npy_path):
                continue
            T = np.load(npy_path).shape[0]
            L = len(row["translation"].strip().split())  # word count not char count
            if T >= L:
                valid_indices.append(i)

        self.data = self.data.loc[valid_indices].reset_index(drop=True)

        print(f"[{split}] Loaded {len(self.data)} samples | vocab size: {len(self.gloss2idx)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load pre-extracted i3d features
        npy_path = os.path.join(self.npy_dir, row["id"] + ".npy")
        features = np.load(npy_path)

        if features.shape[0] > self.max_frames:
            features = features[:self.max_frames]

        frames_tensor = torch.tensor(features, dtype=torch.float32)

        # Word-level tokenization — T(frames) >> L(words) so CTC works fine
        words = row["translation"].strip().split()
        label = torch.tensor(
            [self.gloss2idx.get(w, 1) for w in words],
            dtype=torch.long
        )

        return frames_tensor, None, label


def collate_fn(batch):
    """
    Pads frames to same length.
    Concatenates labels for CTC loss.
    """
    frames, _, labels = zip(*batch)
    frame_lengths = torch.tensor([f.shape[0] for f in frames])
    label_lengths = torch.tensor([l.shape[0] for l in labels])
    frames_padded = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)
    labels_concat = torch.cat(labels)
    return frames_padded, None, labels_concat, frame_lengths, label_lengths