import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PhoenixDataset(Dataset):

    TSV_BASE = (
        "/kaggle/input/datasets/rabeyaakter23/"
        "rwth-phoenix-2014t-i3d-features-mediapipe-features/"
        "tsv files_rwth phoenix 2014t/tsv files"
    )
    I3D_BASE = (
        "/kaggle/input/datasets/rabeyaakter23/"
        "rwth-phoenix-2014t-i3d-features-mediapipe-features/"
        "i3d_features_rwth phoenix 2014t/"
        "i3d_features_rwth phoenix 2014t"
    )
    SPLIT_MAP = {
        "train": "cvpr23.fairseq.i3d.train.how2sign.tsv",
        "val":   "cvpr23.fairseq.i3d.val.how2sign.tsv",
        "test":  "cvpr23.fairseq.i3d.test.how2sign.tsv",
    }

    def __init__(self, split, root_dir, max_frames=400):
        self.split      = split
        self.root_dir   = root_dir
        self.max_frames = max_frames

        # Load TSV
        csv_path = os.path.join(self.TSV_BASE, self.SPLIT_MAP[split])
        self.data = pd.read_csv(csv_path, sep="\t")

        # I3D feature directory for this split
        self.npy_dir = os.path.join(
            self.I3D_BASE, split
        )

        # Build shared word vocabulary from all splits
        self.vocab = self._build_vocab()

        # Filter out bad samples where T < L
        valid_rows = []
        for _, row in self.data.iterrows():
            npy_path = os.path.join(self.npy_dir, str(row["id"]) + ".npy")
            if not os.path.exists(npy_path):
                continue
            T = np.load(npy_path, mmap_mode="r").shape[0]
            words = str(row["translation"]).strip().split()
            L = len(words)
            if T >= L:
                valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[{split}] {len(self.data)} valid samples after filtering")

    def _build_vocab(self):
        vocab = {"<blank>": 0}
        for split_name, tsv_file in self.SPLIT_MAP.items():
            tsv_path = os.path.join(self.TSV_BASE, tsv_file)
            if not os.path.exists(tsv_path):
                continue
            df = pd.read_csv(tsv_path, sep="\t")
            for sentence in df["translation"].dropna():
                for word in str(sentence).strip().split():
                    if word not in vocab:
                        vocab[word] = len(vocab)
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load I3D features
        npy_path = os.path.join(self.npy_dir, str(row["id"]) + ".npy")
        frames = np.load(npy_path)
        frames = torch.tensor(frames, dtype=torch.float32)

        # Trim to max_frames
        if frames.shape[0] > self.max_frames:
            frames = frames[:self.max_frames]

        # Encode label as word-level indices
        words = str(row["translation"]).strip().split()
        label = torch.tensor(
            [self.vocab[w] for w in words if w in self.vocab],
            dtype=torch.long
        )

        return frames, None, label


def collate_fn(batch):
    frames_list, _, labels_list = zip(*batch)

    # Pad frames to max T in batch
    T_max = max(f.shape[0] for f in frames_list)
    feat_dim = frames_list[0].shape[1]
    padded_frames = torch.zeros(len(frames_list), T_max, feat_dim)
    input_lengths = []
    for i, f in enumerate(frames_list):
        padded_frames[i, :f.shape[0]] = f
        input_lengths.append(f.shape[0])

    # Flatten labels for CTC
    label_lengths = [len(l) for l in labels_list]
    flat_labels   = torch.cat(labels_list)

    return (
        padded_frames,
        torch.tensor(input_lengths),
        flat_labels,
        torch.tensor(label_lengths)
    )