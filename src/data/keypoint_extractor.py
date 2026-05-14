import mediapipe as mp	
import numpy as np	
import cv2	
	
	
class KeypointExtractor:	
    """	
    Extracts hand and body pose landmarks per frame using MediaPipe.	
	
    Output vector per frame = 258 values:	
      - Hand keypoints:  2 hands x 21 landmarks x 3 coords = 126 values	
      - Pose keypoints: 33 body landmarks x 4 values       = 132 values	
      - Total: 258 values	
	
    If no hands or body detected in a frame, that section is all zeros.	
    """	
	
    def __init__(self):	
        self.mp_hands = mp.solutions.hands	
        self.mp_pose  = mp.solutions.pose	
	
        # static_image_mode=True: we process one frame at a time (not video)	
        self.hands = self.mp_hands.Hands(	
            static_image_mode=True,	
            max_num_hands=2,	
            min_detection_confidence=0.5	
        )	
        self.pose = self.mp_pose.Pose(	
            static_image_mode=True,	
            min_detection_confidence=0.5	
        )	
	
    def extract(self, frame_rgb):	
        """	
        Process a single frame and return a 258-dim feature vector.	
	
        Args:	
            frame_rgb: numpy array (H, W, 3) in RGB format	
	
        Returns:	
            numpy array of shape (258,), dtype float32	
            All zeros if no person detected in the frame.	
        """	
        # ── Hand keypoints ───────────────────────────────────────────	
        hand_feats = np.zeros(126, dtype=np.float32)	
        hand_result = self.hands.process(frame_rgb)	
	
        if hand_result.multi_hand_landmarks:	
            for i, hand_lm in enumerate(hand_result.multi_hand_landmarks[:2]):	
                for j, lm in enumerate(hand_lm.landmark):	
                    hand_feats[i * 63 + j * 3]     = lm.x	
                    hand_feats[i * 63 + j * 3 + 1] = lm.y	
                    hand_feats[i * 63 + j * 3 + 2] = lm.z	
	
        # ── Pose keypoints ───────────────────────────────────────────	
        pose_feats = np.zeros(132, dtype=np.float32)	
        pose_result = self.pose.process(frame_rgb)	
	
        if pose_result.pose_landmarks:	
            for j, lm in enumerate(pose_result.pose_landmarks.landmark):	
                pose_feats[j * 4]     = lm.x	
                pose_feats[j * 4 + 1] = lm.y	
                pose_feats[j * 4 + 2] = lm.z	
                pose_feats[j * 4 + 3] = lm.visibility	
	
        # Concatenate: 126 + 132 = 258 total values	
        return np.concatenate([hand_feats, pose_feats])	
	
    def extract_sequence(self, frames):	
        """	
        Process a full sequence of frames.	
	
        Args:	
            frames: numpy array (T, H, W, 3)	
	
        Returns:	
            numpy array (T, 258)	
        """	
        return np.array([self.extract(f) for f in frames], dtype=np.float32)	
	
    def close(self):	
        """Release MediaPipe resources when done."""	
        self.hands.close()	
        self.pose.close()	