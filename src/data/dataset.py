import os	
import torch	
import numpy as np	
import pandas as pd	
from torch.utils.data import Dataset	
from .preprocessing import extract_frames, get_transform	
from .keypoint_extractor import KeypointExtractor	
	
	
class PhoenixDataset(Dataset):	
    """	
    PyTorch Dataset for RWTH PHOENIX-Weather 2014T.	
	
    Expected folder structure:	
    phoenix2014T/	
    ├── annotations/manual/	
    │   ├── train.corpus.csv   (columns separated by |)	
    │   ├── dev.corpus.csv     (validation set)	
    │   └── test.corpus.csv	
    └── features/fullFrame-210x260px/	
        ├── train/<video_folder>/<frame>.png	
        ├── dev/	
        └── test/	
	
    CSV columns: name | video | start | end | speaker | folder | annotation	
    annotation = space-separated German glosses e.g. "HEUTE WETTER GUT"	
    """	
	
    def __init__(self, root_dir, split="train",	
                 max_frames=300, img_size=224, use_keypoints=True):	
        """	
        Args:	
            root_dir:      path to phoenix2014T/ root folder	
            split:         "train", "dev" (validation), or "test"	
            max_frames:    max frames to use from each video	
            img_size:      pixel size to resize each frame to	
            use_keypoints: whether to extract MediaPipe features	
        """	
        self.root_dir      = root_dir	
        self.split         = split	
        self.max_frames    = max_frames	
        self.use_keypoints = use_keypoints	
        self.transform     = get_transform(train=(split == "train"))	
	
        # Load annotation CSV	
        csv_path = os.path.join(	
            root_dir, f"annotations/manual/{split}.corpus.csv"	
        )	
        self.data = pd.read_csv(csv_path, sep="|")	
	
        # Build vocabulary: each unique gloss word gets an integer index	
        # Index 0 is reserved for <blank> which CTC loss requires	
        # Index 1 is <unk> for unknown words at inference time	
        all_glosses = " ".join(self.data["annotation"].tolist()).split()	
        vocab = sorted(set(all_glosses))	
	
        self.gloss2idx = {"<blank>": 0, "<unk>": 1}	
        self.gloss2idx.update({g: i + 2 for i, g in enumerate(vocab)})	
        self.idx2gloss = {v: k for k, v in self.gloss2idx.items()}	
	
        if self.use_keypoints:	
            self.kp_extractor = KeypointExtractor()	
	
        print(f"[{split}] Loaded {len(self.data)} samples | vocab size: {len(self.gloss2idx)}")	
	
    def __len__(self):	
        return len(self.data)	
	
    def __getitem__(self, idx):	
        row = self.data.iloc[idx]	
	
        # Build path to frame folder for this sample	
        video_dir = os.path.join(	
            self.root_dir,	
            "features/fullFrame-210x260px",	
            self.split,	
            row["folder"]	
        )	
	
        # Extract frames as numpy array (T, H, W, 3)	
        frames = extract_frames(video_dir, self.max_frames)	
	
        # Apply transforms and convert to tensor (T, C, H, W)	
        frames_tensor = torch.stack([self.transform(f) for f in frames])	
	
        # Extract keypoints if enabled → tensor (T, 258)	
        keypoints_tensor = None	
        if self.use_keypoints:	
            kps = self.kp_extractor.extract_sequence(frames)	
            keypoints_tensor = torch.tensor(kps, dtype=torch.float32)	
	
        # Convert gloss annotation string to integer list	
        glosses = row["annotation"].split()	
        label = torch.tensor(	
            [self.gloss2idx.get(g, 1) for g in glosses],	
            dtype=torch.long	
        )	
	
        if self.use_keypoints:	
            return frames_tensor, keypoints_tensor, label	
        return frames_tensor, label	
	
	
def collate_fn(batch):	
    """	
    Custom collate function for the DataLoader.	
    Videos have different lengths so we pad them to the same length.	
    CTC loss requires all label sequences to be concatenated.	
    """	
    if len(batch[0]) == 3:	
        frames, keypoints, labels = zip(*batch)	
        kp_padded = torch.nn.utils.rnn.pad_sequence(	
            keypoints, batch_first=True	
        )	
    else:	
        frames, labels = zip(*batch)	
        kp_padded = None	
	
    frame_lengths = torch.tensor([f.shape[0] for f in frames])	
    label_lengths = torch.tensor([l.shape[0] for l in labels])	
    frames_padded = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)	
    labels_concat = torch.cat(labels)	
	
    return frames_padded, kp_padded, labels_concat, frame_lengths, label_lengths	