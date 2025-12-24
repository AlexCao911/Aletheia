#!/usr/bin/env python3
"""
MediaPipe 面部特征提取演示应用 (使用新版 Tasks API)

实时显示摄像头画面，并可视化 14 维面部特征向量
"""

import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 特征名称
FEATURE_NAMES = [
    ("左眼开合", "L Eye"),
    ("右眼开合", "R Eye"),
    ("水平视线", "Gaze H"),
    ("垂直视线", "Gaze V"),
    ("左眉高度", "L Brow"),
    ("右眉高度", "R Brow"),
    ("眉头皱起", "Furrow"),
    ("嘴巴张开", "Mouth Open"),
    ("嘴巴宽度", "Mouth W"),
    ("噘嘴程度", "Pucker"),
    ("微笑强度", "Smile"),
    ("俯仰角", "Pitch"),
    ("偏航角", "Yaw"),
    ("翻滚角", "Roll"),
]

# Landmark indices
LEFT_EYE = {'top': [159, 158, 157, 173], 'bottom': [145, 144, 153, 154], 'left': [33], 'right': [133]}
RIGHT_EYE = {'top': [386, 385, 384, 398], 'bottom': [374, 373, 380, 381], 'left': [362], 'right': [263]}
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]
MOUTH_OUTER = {'top': [13], 'bottom': [14], 'left': [61], 'right': [291]}


@dataclass
class FaceFeatures:
    """14维面部特征"""
    left_eye_aspect_ratio: float = 0.0
    right_eye_aspect_ratio: float = 0.0
    eye_gaze_horizontal: float = 0.0
    eye_gaze_vertical: float = 0.0
    left_eyebrow_height: float = 0.0
    right_eyebrow_height: float = 0.0
    eyebrow_furrow: float = 0.0
    mouth_open_ratio: float = 0.0
    mouth_width_ratio: float = 0.0
    lip_pucker: float = 0.0
    smile_intensity: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    landmarks: Optional[np.ndarray] = None

    def to_array(self) -> np.ndarray:
        return np.array([
            self.left_eye_aspect_ratio, self.right_eye_aspect_ratio,
            self.eye_gaze_horizontal, self.eye_gaze_vertical,
            self.left_eyebrow_height, self.right_eyebrow_height, self.eyebrow_furrow,
            self.mouth_open_ratio, self.mouth_width_ratio, self.lip_pucker, self.smile_intensity,
            self.head_pitch, self.head_yaw, self.head_roll,
        ], dtype=np.float32)


class FaceFeatureExtractor:
    """使用 MediaPipe Face Landmarker 提取面部特征"""
    
    def __init__(self):
        # 下载模型（首次运行）
        model_path = self._get_model_path()
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def _get_model_path(self) -> str:
        """获取或下载模型文件"""
        import os
        import urllib.request
        
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading face landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded.")
        return model_path
    
    def extract(self, frame: np.ndarray) -> Optional[FaceFeatures]:
        """从帧中提取特征"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.detector.detect(mp_image)
        
        if not result.face_landmarks:
            return None
        
        h, w = frame.shape[:2]
        landmarks = np.array([[lm.x * w, lm.y * h, lm.z * w] 
                              for lm in result.face_landmarks[0]])
        
        features = FaceFeatures(landmarks=landmarks)
        
        # 计算眼睛开合度
        features.left_eye_aspect_ratio = self._eye_aspect_ratio(landmarks, LEFT_EYE)
        features.right_eye_aspect_ratio = self._eye_aspect_ratio(landmarks, RIGHT_EYE)
        
        # 计算视线方向
        features.eye_gaze_horizontal, features.eye_gaze_vertical = self._compute_gaze(landmarks)
        
        # 计算眉毛特征
        features.left_eyebrow_height, features.right_eyebrow_height = self._eyebrow_heights(landmarks)
        features.eyebrow_furrow = self._eyebrow_furrow(landmarks)
        
        # 计算嘴巴特征
        features.mouth_open_ratio = self._mouth_open(landmarks)
        features.mouth_width_ratio = self._mouth_width(landmarks)
        features.lip_pucker = self._lip_pucker(landmarks)
        features.smile_intensity = self._smile_intensity(landmarks)
        
        # 头部姿态（从变换矩阵）
        if result.facial_transformation_matrixes:
            features.head_pitch, features.head_yaw, features.head_roll = \
                self._head_pose_from_matrix(result.facial_transformation_matrixes[0])
        
        return features
    
    def _eye_aspect_ratio(self, lm: np.ndarray, eye: dict) -> float:
        top = lm[eye['top']]
        bottom = lm[eye['bottom']]
        left = lm[eye['left'][0]][:2]
        right = lm[eye['right'][0]][:2]
        
        v_dist = np.mean(np.linalg.norm(top[:, :2] - bottom[:, :2], axis=1))
        h_dist = np.linalg.norm(left - right)
        
        if h_dist < 1e-6:
            return 0.0
        return float(np.clip(v_dist / h_dist, 0.0, 1.0))
    
    def _compute_gaze(self, lm: np.ndarray) -> Tuple[float, float]:
        # 简化的视线计算
        left_iris = np.mean(lm[LEFT_IRIS], axis=0)[:2]
        right_iris = np.mean(lm[RIGHT_IRIS], axis=0)[:2]
        
        left_center = (lm[LEFT_EYE['left'][0]][:2] + lm[LEFT_EYE['right'][0]][:2]) / 2
        right_center = (lm[RIGHT_EYE['left'][0]][:2] + lm[RIGHT_EYE['right'][0]][:2]) / 2
        
        left_w = np.linalg.norm(lm[LEFT_EYE['right'][0]][:2] - lm[LEFT_EYE['left'][0]][:2])
        right_w = np.linalg.norm(lm[RIGHT_EYE['right'][0]][:2] - lm[RIGHT_EYE['left'][0]][:2])
        
        if left_w < 1e-6 or right_w < 1e-6:
            return 0.0, 0.0
        
        h = ((left_iris[0] - left_center[0]) / (left_w/2) + 
             (right_iris[0] - right_center[0]) / (right_w/2)) / 2
        v = -((left_iris[1] - left_center[1]) / (left_w/2) + 
              (right_iris[1] - right_center[1]) / (right_w/2)) / 2
        
        return float(np.clip(h, -1, 1)), float(np.clip(v, -1, 1))
    
    def _eyebrow_heights(self, lm: np.ndarray) -> Tuple[float, float]:
        face_h = abs(lm[152, 1] - lm[10, 1])
        if face_h < 1e-6:
            return 0.5, 0.5
        
        left_dist = (np.mean(lm[LEFT_EYE['top']][:, 1]) - np.mean(lm[LEFT_EYEBROW][:, 1])) / face_h
        right_dist = (np.mean(lm[RIGHT_EYE['top']][:, 1]) - np.mean(lm[RIGHT_EYEBROW][:, 1])) / face_h
        
        return (float(np.clip((left_dist - 0.05) / 0.10, 0, 1)),
                float(np.clip((right_dist - 0.05) / 0.10, 0, 1)))
    
    def _eyebrow_furrow(self, lm: np.ndarray) -> float:
        inner_dist = np.linalg.norm(lm[107][:2] - lm[336][:2])
        face_w = abs(lm[454, 0] - lm[234, 0])
        if face_w < 1e-6:
            return 0.0
        norm_dist = inner_dist / face_w
        return float(np.clip(1.0 - (norm_dist - 0.10) / 0.20, 0, 1))
    
    def _mouth_open(self, lm: np.ndarray) -> float:
        v = np.linalg.norm(lm[13][:2] - lm[14][:2])
        h = np.linalg.norm(lm[61][:2] - lm[291][:2])
        if h < 1e-6:
            return 0.0
        return float(np.clip((v / h) / 0.7, 0, 1))
    
    def _mouth_width(self, lm: np.ndarray) -> float:
        mouth_w = np.linalg.norm(lm[61][:2] - lm[291][:2])
        face_w = abs(lm[454, 0] - lm[234, 0])
        if face_w < 1e-6:
            return 0.5
        ratio = mouth_w / face_w
        return float(np.clip((ratio - 0.25) / 0.30, 0, 1))
    
    def _lip_pucker(self, lm: np.ndarray) -> float:
        mouth_w = np.linalg.norm(lm[61][:2] - lm[291][:2])
        face_w = abs(lm[454, 0] - lm[234, 0])
        if face_w < 1e-6:
            return 0.0
        width_ratio = mouth_w / face_w
        return float(np.clip(1.0 - (width_ratio - 0.20) / 0.20, 0, 1))
    
    def _smile_intensity(self, lm: np.ndarray) -> float:
        left_corner = lm[61]
        right_corner = lm[291]
        mouth_center_y = (lm[13, 1] + lm[14, 1]) / 2
        
        avg_raise = mouth_center_y - (left_corner[1] + right_corner[1]) / 2
        face_h = abs(lm[152, 1] - lm[10, 1])
        
        if face_h < 1e-6:
            return 0.0
        
        norm_raise = avg_raise / face_h
        return float(np.clip((norm_raise + 0.01) / 0.05, 0, 1))
    
    def _head_pose_from_matrix(self, matrix) -> Tuple[float, float, float]:
        """从变换矩阵提取欧拉角"""
        # matrix 是 4x4 变换矩阵
        r = np.array(matrix)[:3, :3]
        
        sy = np.sqrt(r[0, 0]**2 + r[1, 0]**2)
        if sy > 1e-6:
            pitch = np.arctan2(r[2, 1], r[2, 2])
            yaw = np.arctan2(-r[2, 0], sy)
            roll = np.arctan2(r[1, 0], r[0, 0])
        else:
            pitch = np.arctan2(-r[1, 2], r[1, 1])
            yaw = np.arctan2(-r[2, 0], sy)
            roll = 0
        
        return (float(np.clip(np.degrees(pitch), -90, 90)),
                float(np.clip(np.degrees(yaw), -90, 90)),
                float(np.clip(np.degrees(roll), -90, 90)))
    
    def close(self):
        pass


def draw_feature_bars(frame: np.ndarray, features: FaceFeatures, x: int = 10, y: int = 30):
    """绘制特征条形图"""
    arr = features.to_array()
    bar_w, bar_h, spacing = 150, 16, 20
    
    for i, (cn, en) in enumerate(FEATURE_NAMES):
        py = y + i * spacing
        val = arr[i]
        
        cv2.putText(frame, f"{en}: {val:.2f}", (x, py), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        bar_x = x + 95
        cv2.rectangle(frame, (bar_x, py - 10), (bar_x + bar_w, py), (50, 50, 50), -1)
        
        if i in [2, 3]:  # 视线 [-1, 1]
            center = bar_x + bar_w // 2
            fill = int(abs(val) * bar_w / 2)
            if val >= 0:
                cv2.rectangle(frame, (center, py - 8), (center + fill, py - 2), (0, 200, 200), -1)
            else:
                cv2.rectangle(frame, (center - fill, py - 8), (center, py - 2), (0, 200, 200), -1)
            cv2.line(frame, (center, py - 10), (center, py), (100, 100, 100), 1)
        elif i >= 11:  # 头部角度
            norm = val / 90.0
            center = bar_x + bar_w // 2
            fill = int(abs(norm) * bar_w / 2)
            color = [(200, 100, 100), (100, 200, 100), (100, 100, 200)][i - 11]
            if norm >= 0:
                cv2.rectangle(frame, (center, py - 8), (center + fill, py - 2), color, -1)
            else:
                cv2.rectangle(frame, (center - fill, py - 8), (center, py - 2), color, -1)
            cv2.line(frame, (center, py - 10), (center, py), (100, 100, 100), 1)
        else:  # [0, 1]
            fill = int(val * bar_w)
            color = (0, 255, 0) if i < 2 else (255, 165, 0) if i < 7 else (255, 0, 255)
            cv2.rectangle(frame, (bar_x, py - 8), (bar_x + fill, py - 2), color, -1)


def draw_face_mesh(frame: np.ndarray, landmarks: np.ndarray):
    """绘制简化面部网格"""
    if landmarks is None:
        return
    
    # 面部轮廓
    oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    pts = landmarks[oval, :2].astype(np.int32)
    cv2.polylines(frame, [pts], True, (100, 100, 100), 1)
    
    # 眼睛
    for eye in [[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
                [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]]:
        pts = landmarks[eye, :2].astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 1)
    
    # 嘴巴
    mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    pts = landmarks[mouth, :2].astype(np.int32)
    cv2.polylines(frame, [pts], True, (255, 0, 255), 1)


def main():
    print("=" * 50)
    print("MediaPipe Face Feature Extraction Demo")
    print("=" * 50)
    print("\nInitializing...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    extractor = FaceFeatureExtractor()
    
    print("Ready! Press 'q' to quit, 's' to print features.\n")
    
    fps_history = deque(maxlen=30)
    last_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            features = extractor.extract(frame)
            
            # FPS
            now = time.time()
            fps_history.append(1.0 / (now - last_time + 1e-6))
            last_time = now
            fps = np.mean(fps_history)
            
            display = frame.copy()
            h, w = display.shape[:2]
            
            if features is not None:
                draw_face_mesh(display, features.landmarks)
                draw_feature_bars(display, features)
            
            # Info
            cv2.putText(display, f"FPS: {fps:.1f}", (w - 100, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            status = "Face OK" if features else "No Face"
            color = (0, 255, 0) if features else (0, 0, 255)
            cv2.putText(display, status, (w - 100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(display, "Q: Quit | S: Print", (10, h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Face Features Demo', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and features:
                print("\n14-dim Feature Vector:")
                print(features.to_array().tolist())
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nDone.")


if __name__ == "__main__":
    main()
