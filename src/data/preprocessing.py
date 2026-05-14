import cv2	
import torch	
import numpy as np	
import os	
from torchvision import transforms	
	
	
def extract_frames(video_path, max_frames=300, img_size=224):	
    """	
    Read a video or folder of images and return evenly spaced frames.	
	
    NOTE: PHOENIX dataset stores frames as PNG images inside folders,	
    NOT as .mp4 files. This function handles both cases.	
	
    Args:	
        video_path: path to .mp4/.avi file OR path to a folder of images	
        max_frames: how many frames to sample (300 is a good default)	
        img_size:   resize each frame to this square size in pixels	
	    Returns:	
        numpy array of shape (num_frames, img_size, img_size, 3)	
        values are uint8 (0-255) in RGB format	
    """	
    frames = []	
	
    if os.path.isdir(video_path):	
        # PHOENIX stores frames as sorted image files in a folder	
        img_files = sorted([	
            f for f in os.listdir(video_path)	
            if f.lower().endswith((".png", ".jpg", ".jpeg"))	
        ])	
        if not img_files:	
            return np.array(frames, dtype=np.uint8)	
	
        # Sample evenly across the full sequence	
        indices = np.linspace(0, len(img_files) - 1,	
                              min(max_frames, len(img_files)), dtype=int)	
        for idx in indices:	
            img = cv2.imread(os.path.join(video_path, img_files[idx]))	
            if img is not None:	
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	
                img = cv2.resize(img, (img_size, img_size))	
                frames.append(img)	
    else:	
        # Standard video file	
        cap = cv2.VideoCapture(video_path)	
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))	
        if total == 0:	
            cap.release()	
            return np.array(frames, dtype=np.uint8)	
	
        indices = np.linspace(0, total - 1,	
                              min(max_frames, total), dtype=int)	
        for idx in indices:	
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)	
            ret, frame = cap.read()	
            if ret:	
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)	
                frame = cv2.resize(frame, (img_size, img_size))	
                frames.append(frame)	
        cap.release()	
	
    return np.array(frames, dtype=np.uint8)	
	
	
def get_transform(train=True):	
    """	
    Returns the torchvision transform pipeline for preprocessing.	
	
    Training mode: includes random augmentations (flip, color jitter,	
    small rotation) to help the model generalize better.	
	
    Inference mode: only resize and normalize — no random changes.	
	
    The mean/std values [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225]	
    are ImageNet statistics. Use these exact numbers because ResNet-50	
    was pretrained on ImageNet with these normalization values.	
    """	
    imagenet_mean = [0.485, 0.456, 0.406]	
    imagenet_std  = [0.229, 0.224, 0.225]	
	
    if train:	
        return transforms.Compose([	
            transforms.ToPILImage(),	
            transforms.RandomHorizontalFlip(p=0.5),   # flip left-right	
            transforms.ColorJitter(	
                brightness=0.2, contrast=0.2,	
                saturation=0.1, hue=0.05	
            ),	
            transforms.RandomRotation(degrees=10),    # small rotation	
            transforms.ToTensor(),	
            transforms.Normalize(imagenet_mean, imagenet_std)	
        ])	
    else:	
        return transforms.Compose([	
            transforms.ToPILImage(),	
            transforms.ToTensor(),	
            transforms.Normalize(imagenet_mean, imagenet_std)	
        ])	