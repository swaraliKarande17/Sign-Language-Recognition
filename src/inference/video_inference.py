import torch	
import numpy as np	
from src.models.ctc_model import CSLRModel	
from src.data.preprocessing import extract_frames, get_transform	
from src.data.keypoint_extractor import KeypointExtractor	
from src.utils.metrics import decode_predictions	
from src.utils.translation import GermanToEnglishTranslator	
	
	
class VideoInference:	
    """	
    Full inference pipeline for a video file.	
    Pipeline: video file -> frames -> CNN+LSTM -> CTC decode -> German glosses -> English	
    """	
	
    def __init__(self, checkpoint_path, device="cpu"):	
        self.device    = torch.device(device)	
        self.transform = get_transform(train=False)	
        self.kp_ext    = KeypointExtractor()	
        self.translator = GermanToEnglishTranslator()	
	
        # Load checkpoint — contains model weights AND vocabulary	
        print(f"Loading model from: {checkpoint_path}")	
        checkpoint = torch.load(checkpoint_path, map_location=self.device)	
        self.vocab     = checkpoint["vocab"]	
        self.idx2gloss = {v: k for k, v in self.vocab.items()}	
	
        self.model = CSLRModel(vocab_size=len(self.vocab))	
        self.model.load_state_dict(checkpoint["model_state"])	
        self.model.eval().to(self.device)	
        print(f"Model loaded. Vocabulary size: {len(self.vocab)}")	
	
    def run(self, video_path, max_frames=300):	
        """	
        Run the full pipeline on a video file.	
	
        Args:	
            video_path: path to video file or folder of frames	
            max_frames: maximum frames to process	
	
        Returns:	
            dict with keys:	
                "glosses":    list of German gloss strings	
                "english":    English translation string	
                "num_frames": number of frames processed	
        """	
        # Step 1: Extract frames from video	
        frames = extract_frames(video_path, max_frames)	
        if len(frames) == 0:	
            return {"error": "Could not read video or no frames found"}	
	
        # Step 2: Extract MediaPipe keypoints for all frames	
        keypoints = self.kp_ext.extract_sequence(frames)	
	
        # Step 3: Apply transforms and convert to tensors	
        frames_t    = torch.stack([self.transform(f) for f in frames])	
        frames_t    = frames_t.unsqueeze(0).to(self.device)     # (1, T, C, H, W)	
        keypoints_t = torch.tensor(keypoints, dtype=torch.float32)	
        keypoints_t = keypoints_t.unsqueeze(0).to(self.device)  # (1, T, 258)	
	
        # Step 4: Run model forward pass	
        with torch.no_grad():	
            log_probs = self.model(frames_t, keypoints_t)  # (1, T, V)	
	
        # Step 5: Greedy CTC decode -> list of gloss strings	
        decoded = decode_predictions(log_probs, self.idx2gloss)	
        glosses = decoded[0].split() if decoded[0] else []	
	
        # Step 6: Translate German glosses to English	
        english = self.translator.translate(glosses)	
	
        return {	
            "glosses":    glosses,	
            "english":    english,	
            "num_frames": len(frames)	
        }	