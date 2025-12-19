"""
FaceFeatureExtractor for extracting facial features using MediaPipe Face Mesh.

This module provides the FaceFeatureExtractor class that processes video frames
and extracts normalized facial features including eye aspect ratios, mouth features,
eyebrow positions, and head pose estimation.
"""

import time
from typing import Optional, Tuple
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
except ImportError:
    mp = None
    mp_face_mesh = None

from .features import FaceFeatures


# MediaPipe Face Mesh landmark indices
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Eye landmarks for aspect ratio calculation
LEFT_EYE_LANDMARKS = {
    'top': [159, 158, 157, 173],      # Upper eyelid
    'bottom': [145, 144, 153, 154],   # Lower eyelid
    'left': [33],                      # Left corner
    'right': [133],                    # Right corner
}

RIGHT_EYE_LANDMARKS = {
    'top': [386, 385, 384, 398],      # Upper eyelid
    'bottom': [374, 373, 380, 381],   # Lower eyelid
    'left': [362],                     # Left corner
    'right': [263],                    # Right corner
}

# Iris landmarks for gaze estimation
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eyebrow landmarks
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]

# Mouth landmarks
MOUTH_OUTER = {
    'top': [13],           # Upper lip center
    'bottom': [14],        # Lower lip center
    'left': [61],          # Left corner
    'right': [291],        # Right corner
}

MOUTH_INNER = {
    'top': [13, 312, 311, 310, 82, 81, 80],
    'bottom': [14, 317, 402, 318, 87, 178, 88],
}

# Lip landmarks for pucker detection
UPPER_LIP_CENTER = [0, 267, 269, 270, 37, 39, 40]
LOWER_LIP_CENTER = [17, 84, 181, 91, 314, 405, 321]

# Face oval for bounding box
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Key points for head pose estimation
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291


class FaceFeatureExtractor:
    """从 MediaPipe Face Mesh 提取关键面部特征
    
    Extracts 14 facial features from video frames using MediaPipe Face Mesh:
    - Eye aspect ratios (2): left and right eye openness
    - Eye gaze (2): horizontal and vertical gaze direction
    - Eyebrow features (3): left/right height and furrow
    - Mouth features (4): openness, width, pucker, smile
    - Head pose (3): pitch, yaw, roll
    
    All features are normalized relative to face bounding box.
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 refine_landmarks: bool = True):
        """
        Initialize MediaPipe Face Mesh.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection [0, 1]
            min_tracking_confidence: Minimum confidence for landmark tracking [0, 1]
            refine_landmarks: Whether to refine landmarks around eyes and lips
        """
        if mp is None:
            raise ImportError(
                "MediaPipe is not installed. Install with: pip install mediapipe"
            )
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        # Camera matrix for head pose estimation (will be set on first frame)
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1))
        
        # 3D model points for head pose estimation
        self._model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye outer corner
            (225.0, 170.0, -135.0),      # Right eye outer corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0),     # Right mouth corner
        ], dtype=np.float64)
    
    def extract(self, frame: np.ndarray) -> Optional[FaceFeatures]:
        """
        从视频帧提取面部特征
        
        Args:
            frame: BGR format video frame (H, W, 3)
            
        Returns:
            FaceFeatures object with normalized facial features,
            or None if no face is detected
        """
        if frame is None or frame.size == 0:
            return None
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = frame[:, :, ::-1]
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array (478 landmarks x 3 coordinates)
        h, w = frame.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in face_landmarks.landmark
        ])
        
        # Initialize camera matrix if needed
        if self._camera_matrix is None:
            self._camera_matrix = np.array([
                [w, 0, w / 2],
                [0, w, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)
        
        # Extract features
        timestamp = time.time()
        
        # Eye features
        left_ear = self._compute_eye_aspect_ratio(landmarks, LEFT_EYE_LANDMARKS)
        right_ear = self._compute_eye_aspect_ratio(landmarks, RIGHT_EYE_LANDMARKS)
        gaze_h, gaze_v = self._compute_gaze(landmarks)
        
        # Eyebrow features
        left_brow_h, right_brow_h = self._compute_eyebrow_heights(landmarks)
        furrow = self._compute_eyebrow_furrow(landmarks)
        
        # Mouth features
        mouth_open = self._compute_mouth_open_ratio(landmarks)
        mouth_width = self._compute_mouth_width_ratio(landmarks)
        pucker = self._compute_lip_pucker(landmarks)
        smile = self._compute_smile_intensity(landmarks)
        
        # Head pose
        pitch, yaw, roll = self._compute_head_pose(landmarks, w, h)
        
        return FaceFeatures(
            left_eye_aspect_ratio=left_ear,
            right_eye_aspect_ratio=right_ear,
            eye_gaze_horizontal=gaze_h,
            eye_gaze_vertical=gaze_v,
            left_eyebrow_height=left_brow_h,
            right_eyebrow_height=right_brow_h,
            eyebrow_furrow=furrow,
            mouth_open_ratio=mouth_open,
            mouth_width_ratio=mouth_width,
            lip_pucker=pucker,
            smile_intensity=smile,
            head_pitch=pitch,
            head_yaw=yaw,
            head_roll=roll,
            landmarks=landmarks,
            timestamp=timestamp,
        )
    
    def get_feature_vector(self, features: FaceFeatures) -> np.ndarray:
        """
        将 FaceFeatures 转换为模型输入向量
        
        Args:
            features: FaceFeatures object
            
        Returns:
            Shape (14,) feature vector
        """
        return features.to_array()

    def _compute_eye_aspect_ratio(self, landmarks: np.ndarray, 
                                   eye_landmarks: dict) -> float:
        """
        Compute Eye Aspect Ratio (EAR) for blink/openness detection.
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        
        Returns value in [0, 1] where 0 is closed and 1 is fully open.
        """
        # Get vertical distances (top to bottom)
        top_pts = landmarks[eye_landmarks['top']]
        bottom_pts = landmarks[eye_landmarks['bottom']]
        
        # Average vertical distance
        vertical_dist = np.mean(np.linalg.norm(top_pts[:, :2] - bottom_pts[:, :2], axis=1))
        
        # Horizontal distance (corner to corner)
        left_pt = landmarks[eye_landmarks['left'][0]][:2]
        right_pt = landmarks[eye_landmarks['right'][0]][:2]
        horizontal_dist = np.linalg.norm(left_pt - right_pt)
        
        if horizontal_dist < 1e-6:
            return 0.0
        
        # EAR normalized to [0, 1]
        ear = vertical_dist / horizontal_dist
        return float(np.clip(ear, 0.0, 1.0))
    
    def _compute_gaze(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Compute gaze direction from iris position relative to eye corners.
        
        Returns:
            (horizontal, vertical) gaze in [-1, 1] where:
            - horizontal: -1 = left, 0 = center, 1 = right
            - vertical: -1 = down, 0 = center, 1 = up
        """
        # Left eye
        left_iris_center = np.mean(landmarks[LEFT_IRIS], axis=0)[:2]
        left_eye_left = landmarks[LEFT_EYE_LANDMARKS['left'][0]][:2]
        left_eye_right = landmarks[LEFT_EYE_LANDMARKS['right'][0]][:2]
        left_eye_center = (left_eye_left + left_eye_right) / 2
        left_eye_width = np.linalg.norm(left_eye_right - left_eye_left)
        
        # Right eye
        right_iris_center = np.mean(landmarks[RIGHT_IRIS], axis=0)[:2]
        right_eye_left = landmarks[RIGHT_EYE_LANDMARKS['left'][0]][:2]
        right_eye_right = landmarks[RIGHT_EYE_LANDMARKS['right'][0]][:2]
        right_eye_center = (right_eye_left + right_eye_right) / 2
        right_eye_width = np.linalg.norm(right_eye_right - right_eye_left)
        
        # Average gaze from both eyes
        if left_eye_width < 1e-6 or right_eye_width < 1e-6:
            return 0.0, 0.0
        
        # Horizontal gaze (normalized by eye width)
        left_h = (left_iris_center[0] - left_eye_center[0]) / (left_eye_width / 2)
        right_h = (right_iris_center[0] - right_eye_center[0]) / (right_eye_width / 2)
        gaze_h = float(np.clip((left_h + right_h) / 2, -1.0, 1.0))
        
        # Vertical gaze
        left_eye_top = np.mean(landmarks[LEFT_EYE_LANDMARKS['top']], axis=0)[:2]
        left_eye_bottom = np.mean(landmarks[LEFT_EYE_LANDMARKS['bottom']], axis=0)[:2]
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        
        right_eye_top = np.mean(landmarks[RIGHT_EYE_LANDMARKS['top']], axis=0)[:2]
        right_eye_bottom = np.mean(landmarks[RIGHT_EYE_LANDMARKS['bottom']], axis=0)[:2]
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        
        if left_eye_height < 1e-6 or right_eye_height < 1e-6:
            return gaze_h, 0.0
        
        left_v_center = (left_eye_top[1] + left_eye_bottom[1]) / 2
        right_v_center = (right_eye_top[1] + right_eye_bottom[1]) / 2
        
        # Note: y increases downward in image coordinates
        left_v = -(left_iris_center[1] - left_v_center) / (left_eye_height / 2)
        right_v = -(right_iris_center[1] - right_v_center) / (right_eye_height / 2)
        gaze_v = float(np.clip((left_v + right_v) / 2, -1.0, 1.0))
        
        return gaze_h, gaze_v
    
    def _compute_eyebrow_heights(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Compute eyebrow heights relative to eyes.
        
        Returns values in [0, 1] where 0 is low (frown) and 1 is high (surprised).
        """
        # Left eyebrow height relative to left eye
        left_brow_y = np.mean(landmarks[LEFT_EYEBROW][:, 1])
        left_eye_y = np.mean(landmarks[LEFT_EYE_LANDMARKS['top']][:, 1])
        
        # Right eyebrow height relative to right eye
        right_brow_y = np.mean(landmarks[RIGHT_EYEBROW][:, 1])
        right_eye_y = np.mean(landmarks[RIGHT_EYE_LANDMARKS['top']][:, 1])
        
        # Get face height for normalization
        face_top = landmarks[10, 1]  # Forehead
        face_bottom = landmarks[152, 1]  # Chin
        face_height = abs(face_bottom - face_top)
        
        if face_height < 1e-6:
            return 0.5, 0.5
        
        # Distance from eye to brow, normalized by face height
        # Note: y increases downward, so brow above eye means brow_y < eye_y
        left_dist = (left_eye_y - left_brow_y) / face_height
        right_dist = (right_eye_y - right_brow_y) / face_height
        
        # Normalize to [0, 1] - typical range is 0.05 to 0.15
        left_height = float(np.clip((left_dist - 0.05) / 0.10, 0.0, 1.0))
        right_height = float(np.clip((right_dist - 0.05) / 0.10, 0.0, 1.0))
        
        return left_height, right_height

    def _compute_eyebrow_furrow(self, landmarks: np.ndarray) -> float:
        """
        Compute eyebrow furrow (how close the inner eyebrows are).
        
        Returns value in [0, 1] where 0 is relaxed and 1 is furrowed.
        """
        # Inner eyebrow points (closest to nose)
        left_inner = landmarks[107][:2]   # Left eyebrow inner
        right_inner = landmarks[336][:2]  # Right eyebrow inner
        
        # Distance between inner eyebrows
        inner_dist = np.linalg.norm(left_inner - right_inner)
        
        # Normalize by face width
        face_left = landmarks[234, 0]   # Left face edge
        face_right = landmarks[454, 0]  # Right face edge
        face_width = abs(face_right - face_left)
        
        if face_width < 1e-6:
            return 0.0
        
        # Normalized distance - typical range is 0.15 to 0.25
        norm_dist = inner_dist / face_width
        
        # Invert so that smaller distance = higher furrow
        furrow = float(np.clip(1.0 - (norm_dist - 0.10) / 0.20, 0.0, 1.0))
        
        return furrow
    
    def _compute_mouth_open_ratio(self, landmarks: np.ndarray) -> float:
        """
        Compute mouth openness ratio.
        
        Returns value in [0, 1] where 0 is closed and 1 is fully open.
        """
        # Vertical distance (top lip to bottom lip)
        top_lip = landmarks[MOUTH_OUTER['top'][0]][:2]
        bottom_lip = landmarks[MOUTH_OUTER['bottom'][0]][:2]
        vertical_dist = np.linalg.norm(top_lip - bottom_lip)
        
        # Horizontal distance (mouth width)
        left_corner = landmarks[MOUTH_OUTER['left'][0]][:2]
        right_corner = landmarks[MOUTH_OUTER['right'][0]][:2]
        horizontal_dist = np.linalg.norm(left_corner - right_corner)
        
        if horizontal_dist < 1e-6:
            return 0.0
        
        # Mouth aspect ratio
        mar = vertical_dist / horizontal_dist
        
        # Normalize to [0, 1] - typical range is 0 to 0.7
        return float(np.clip(mar / 0.7, 0.0, 1.0))
    
    def _compute_mouth_width_ratio(self, landmarks: np.ndarray) -> float:
        """
        Compute mouth width relative to face width.
        
        Returns value in [0, 1] representing relative mouth width.
        """
        # Mouth width
        left_corner = landmarks[MOUTH_OUTER['left'][0]][:2]
        right_corner = landmarks[MOUTH_OUTER['right'][0]][:2]
        mouth_width = np.linalg.norm(left_corner - right_corner)
        
        # Face width
        face_left = landmarks[234, 0]
        face_right = landmarks[454, 0]
        face_width = abs(face_right - face_left)
        
        if face_width < 1e-6:
            return 0.5
        
        # Normalized width - typical range is 0.3 to 0.5
        ratio = mouth_width / face_width
        return float(np.clip((ratio - 0.25) / 0.30, 0.0, 1.0))
    
    def _compute_lip_pucker(self, landmarks: np.ndarray) -> float:
        """
        Compute lip pucker intensity (lips pushed forward).
        
        Returns value in [0, 1] where 0 is relaxed and 1 is puckered.
        """
        # Use z-coordinate (depth) of lip center relative to face
        upper_lip_z = np.mean(landmarks[UPPER_LIP_CENTER][:, 2])
        lower_lip_z = np.mean(landmarks[LOWER_LIP_CENTER][:, 2])
        lip_z = (upper_lip_z + lower_lip_z) / 2
        
        # Reference: nose tip z
        nose_z = landmarks[NOSE_TIP, 2]
        
        # Pucker pushes lips forward (more negative z in MediaPipe)
        pucker_depth = nose_z - lip_z
        
        # Also consider mouth width reduction as indicator of pucker
        left_corner = landmarks[MOUTH_OUTER['left'][0]][:2]
        right_corner = landmarks[MOUTH_OUTER['right'][0]][:2]
        mouth_width = np.linalg.norm(left_corner - right_corner)
        
        face_left = landmarks[234, 0]
        face_right = landmarks[454, 0]
        face_width = abs(face_right - face_left)
        
        if face_width < 1e-6:
            return 0.0
        
        width_ratio = mouth_width / face_width
        
        # Combine depth and width signals
        # Pucker: lips forward (positive depth) and narrow mouth
        depth_signal = float(np.clip(pucker_depth / face_width * 10, 0.0, 1.0))
        width_signal = float(np.clip(1.0 - (width_ratio - 0.20) / 0.20, 0.0, 1.0))
        
        return float(np.clip((depth_signal + width_signal) / 2, 0.0, 1.0))
    
    def _compute_smile_intensity(self, landmarks: np.ndarray) -> float:
        """
        Compute smile intensity based on mouth corner positions.
        
        Returns value in [0, 1] where 0 is neutral and 1 is full smile.
        """
        # Mouth corners
        left_corner = landmarks[MOUTH_OUTER['left'][0]]
        right_corner = landmarks[MOUTH_OUTER['right'][0]]
        
        # Mouth center (between lips)
        mouth_center_y = (landmarks[MOUTH_OUTER['top'][0], 1] + 
                         landmarks[MOUTH_OUTER['bottom'][0], 1]) / 2
        
        # Smile raises corners relative to center
        left_raise = mouth_center_y - left_corner[1]
        right_raise = mouth_center_y - right_corner[1]
        avg_raise = (left_raise + right_raise) / 2
        
        # Normalize by face height
        face_top = landmarks[10, 1]
        face_bottom = landmarks[152, 1]
        face_height = abs(face_bottom - face_top)
        
        if face_height < 1e-6:
            return 0.0
        
        # Normalized raise - typical range is -0.02 to 0.05
        norm_raise = avg_raise / face_height
        
        # Also consider mouth width increase
        mouth_width = np.linalg.norm(left_corner[:2] - right_corner[:2])
        face_width = abs(landmarks[454, 0] - landmarks[234, 0])
        
        if face_width < 1e-6:
            return 0.0
        
        width_ratio = mouth_width / face_width
        width_signal = float(np.clip((width_ratio - 0.35) / 0.15, 0.0, 1.0))
        
        # Combine signals
        raise_signal = float(np.clip((norm_raise + 0.01) / 0.05, 0.0, 1.0))
        
        return float(np.clip((raise_signal + width_signal) / 2, 0.0, 1.0))

    def _compute_head_pose(self, landmarks: np.ndarray, 
                           img_w: int, img_h: int) -> Tuple[float, float, float]:
        """
        Compute head pose (pitch, yaw, roll) using solvePnP.
        
        Returns:
            (pitch, yaw, roll) in degrees where:
            - pitch: positive = looking up, negative = looking down
            - yaw: positive = looking right, negative = looking left
            - roll: positive = tilting right, negative = tilting left
        """
        try:
            import cv2
        except ImportError:
            # Fallback: estimate from landmarks without cv2
            return self._estimate_head_pose_simple(landmarks)
        
        # 2D image points
        image_points = np.array([
            landmarks[NOSE_TIP][:2],
            landmarks[CHIN][:2],
            landmarks[LEFT_EYE_OUTER][:2],
            landmarks[RIGHT_EYE_OUTER][:2],
            landmarks[LEFT_MOUTH_CORNER][:2],
            landmarks[RIGHT_MOUTH_CORNER][:2],
        ], dtype=np.float64)
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self._model_points,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
        
        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # Get Euler angles from rotation matrix
        # Using the convention: pitch (x), yaw (y), roll (z)
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
        
        if sy > 1e-6:
            pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = 0.0
        
        # Convert to degrees
        pitch = float(np.degrees(pitch))
        yaw = float(np.degrees(yaw))
        roll = float(np.degrees(roll))
        
        # Clamp to reasonable range
        pitch = float(np.clip(pitch, -90.0, 90.0))
        yaw = float(np.clip(yaw, -90.0, 90.0))
        roll = float(np.clip(roll, -90.0, 90.0))
        
        return pitch, yaw, roll
    
    def _estimate_head_pose_simple(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """
        Simple head pose estimation without cv2.solvePnP.
        
        Uses geometric relationships between facial landmarks.
        """
        # Yaw: horizontal asymmetry of face
        nose = landmarks[NOSE_TIP][:2]
        left_eye = landmarks[LEFT_EYE_OUTER][:2]
        right_eye = landmarks[RIGHT_EYE_OUTER][:2]
        
        eye_center = (left_eye + right_eye) / 2
        left_dist = np.linalg.norm(nose - left_eye)
        right_dist = np.linalg.norm(nose - right_eye)
        
        if left_dist + right_dist > 1e-6:
            yaw = (right_dist - left_dist) / (left_dist + right_dist) * 90
        else:
            yaw = 0.0
        
        # Pitch: vertical position of nose relative to eyes and chin
        chin = landmarks[CHIN][:2]
        nose_to_eyes = eye_center[1] - nose[1]
        nose_to_chin = chin[1] - nose[1]
        
        if abs(nose_to_chin) > 1e-6:
            pitch_ratio = nose_to_eyes / nose_to_chin
            pitch = (pitch_ratio - 0.5) * 60  # Approximate mapping
        else:
            pitch = 0.0
        
        # Roll: angle of eye line
        eye_vec = right_eye - left_eye
        roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
        
        return (
            float(np.clip(pitch, -90.0, 90.0)),
            float(np.clip(yaw, -90.0, 90.0)),
            float(np.clip(roll, -90.0, 90.0))
        )
    
    def close(self):
        """Release MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
