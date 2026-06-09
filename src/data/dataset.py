from cProfile import label
import os
from pyexpat import features
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

    def __init__(self, split, root_dir, max_frames=400):
        self.data = pd.read_csv(csv_path, sep="\t")

    # --- ADD THIS BLOCK ---
        npy_dir = os.path.join(root_dir, f"i3d_features_{split}")
        valid_rows = []
        for _, row in self.data.iterrows():
            npy_path = os.path.join(npy_dir, row["id"] + ".npy")
            if not os.path.exists(npy_path):
                continue           # skip missing files
        T = np.load(npy_path, mmap_mode="r").shape[0]
        words = row["translation"].strip().split()
        L = len(words)         # word-level length
        if T >= L:             # CTC requires T >= L
            valid_rows.append(row)
        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[{split}] {len(self.data)} valid samples after filtering")
    # ----------------------
        # ── Build vocab from ALL splits so train/val/test share same vocab ──
        # This fixes the train(2889 words) vs val(953 words) mismatch
       # Build vocab from all splits ONCE (shared vocab)
    def build_vocab(self ,root_dir):
        all_words = set(["<blank>"])   # index 0 = CTC blank
        for split in ["train", "dev", "test"]:
            tsv = os.path.join(root_dir, f"cvpr23.fairseq.i3d.{split}.how2sign.tsv")
            if not os.path.exists(tsv): continue
            df = pd.read_csv(tsv, sep="\t")
            for sentence in df["translation"]:
                all_words.update(sentence.strip().split())
        vocab = {"<blank>": 0}
        for i, w in enumerate(sorted(all_words - {"<blank>"}), start=1):
            vocab[w] = i
        return vocab

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        words = row["translation"].strip().split()

        label = torch.tensor(
            [self.vocab[w] for w in words if w in self.vocab],
            dtype=torch.long
        )

        return features, label
def collate_fn(batch):
    """Pad frames and labels to max length in batch."""
    frames, labels = zip(*[(b[0], b[2]) for b in batch])

        # Pad frame sequences
    T_max = max(f.shape[0] for f in frames)
    padded_frames = torch.zeros(len(frames), T_max, frames[0].shape[1])
    input_lengths = []
    for i, f in enumerate(frames):
        padded_frames[i, :f.shape[0]] = f
        input_lengths.append(f.shape[0])

        # Concatenate labels (CTC takes flat labels + lengths)
    label_lengths = [len(l) for l in labels]
    flat_labels   = torch.cat(labels)

    return (
        padded_frames,                              # (B, T_max, 1024)
        torch.tensor(input_lengths),                # (B,)
        flat_labels,                                # (sum of L_i,)
        torch.tensor(label_lengths)                 # (B,)
        )