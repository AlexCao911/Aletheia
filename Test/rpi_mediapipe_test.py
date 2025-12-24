#!/usr/bin/env python3
"""
树莓派 MediaPipe 面部特征提取测试

在终端实时输出 14 维特征向量，用于验证 MediaPipe 在树莓派上的运行情况。
支持 CSI 摄像头和 USB 摄像头。

使用方法:
    # USB 摄像头
    python rpi_mediapipe_test.py
    
    # CSI 摄像头 (使用 libcamera)
    python rpi_mediapipe_test.py --csi
    
    # 指定摄像头 ID
    python rpi_mediapipe_test.py --camera 1

按 Ctrl+C 退出
"""

import argparse
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# MediaPipe 导入
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False
    print("错误: MediaPipe 未安装，请运行: pip install mediapipe")
    sys.exit(1)


# ============== 特征提取器 ==============

# Landmark indices
LEFT_EYE = {'top': [159, 158, 157, 173], 'bottom': [145, 144, 153, 154], 'left': [33], 'right': [133]}
RIGHT_EYE = {'top': [386, 385, 384, 398], 'bottom': [374, 373, 380, 381], 'left': [362], 'right': [263]}
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]
MOUTH_OUTER = {'top': [13], 'bottom': [14], 'left': [61], 'right': [291]}
UPPER_LIP_CENTER = [0, 267, 269, 270, 37, 39, 40]
LOWER_LIP_CENTER = [17, 84, 181, 91, 314, 405, 321]
NOSE_TIP = 1

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"


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

    def to_array(self) -> np.ndarray:
        return np.array([
            self.left_eye_aspect_ratio, self.right_eye_aspect_ratio,
            self.eye_gaze_horizontal, self.eye_gaze_vertical,
            self.left_eyebrow_height, self.right_eyebrow_height, self.eyebrow_furrow,
            self.mouth_open_ratio, self.mouth_width_ratio, self.lip_pucker, self.smile_intensity,
            self.head_pitch, self.head_yaw, self.head_roll,
        ], dtype=np.float32)


class SimpleFeatureExtractor:
    """简化的特征提取器"""
    
    def __init__(self):
        model_path = self._ensure_model()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def _ensure_model(self) -> str:
        if os.path.exists(MODEL_PATH):
            return MODEL_PATH
        print(f"下载模型文件 {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("下载完成")
        return MODEL_PATH
    
    def extract(self, frame: np.ndarray) -> Optional[FaceFeatures]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        
        if not result.face_landmarks:
            return None
        
        h, w = frame.shape[:2]
        lm = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]])
        
        f = FaceFeatures()
        f.left_eye_aspect_ratio = self._ear(lm, LEFT_EYE)
        f.right_eye_aspect_ratio = self._ear(lm, RIGHT_EYE)
        f.eye_gaze_horizontal, f.eye_gaze_vertical = self._gaze(lm)
        f.left_eyebrow_height, f.right_eyebrow_height = self._brow_heights(lm)
        f.eyebrow_furrow = self._furrow(lm)
        f.mouth_open_ratio = self._mouth_open(lm)
        f.mouth_width_ratio = self._mouth_width(lm)
        f.lip_pucker = self._pucker(lm)
        f.smile_intensity = self._smile(lm)
        
        if result.facial_transformation_matrixes:
            f.head_pitch, f.head_yaw, f.head_roll = self._head_pose(result.facial_transformation_matrixes[0])
        
        return f
    
    def _ear(self, lm, eye):
        top, bottom = lm[eye['top']], lm[eye['bottom']]
        v = np.mean(np.linalg.norm(top[:,:2] - bottom[:,:2], axis=1))
        h = np.linalg.norm(lm[eye['left'][0]][:2] - lm[eye['right'][0]][:2])
        return float(np.clip(v / h, 0, 1)) if h > 1e-6 else 0.0
    
    def _gaze(self, lm):
        li = np.mean(lm[LEFT_IRIS], axis=0)[:2]
        ri = np.mean(lm[RIGHT_IRIS], axis=0)[:2]
        lc = (lm[LEFT_EYE['left'][0]][:2] + lm[LEFT_EYE['right'][0]][:2]) / 2
        rc = (lm[RIGHT_EYE['left'][0]][:2] + lm[RIGHT_EYE['right'][0]][:2]) / 2
        lw = np.linalg.norm(lm[LEFT_EYE['right'][0]][:2] - lm[LEFT_EYE['left'][0]][:2])
        rw = np.linalg.norm(lm[RIGHT_EYE['right'][0]][:2] - lm[RIGHT_EYE['left'][0]][:2])
        if lw < 1e-6 or rw < 1e-6:
            return 0.0, 0.0
        gh = ((li[0]-lc[0])/(lw/2) + (ri[0]-rc[0])/(rw/2)) / 2
        gv = -((li[1]-lc[1])/(lw/2) + (ri[1]-rc[1])/(rw/2)) / 2
        return float(np.clip(gh, -1, 1)), float(np.clip(gv, -1, 1))
    
    def _brow_heights(self, lm):
        fh = abs(lm[152, 1] - lm[10, 1])
        if fh < 1e-6:
            return 0.5, 0.5
        ld = (np.mean(lm[LEFT_EYE['top']][:,1]) - np.mean(lm[LEFT_EYEBROW][:,1])) / fh
        rd = (np.mean(lm[RIGHT_EYE['top']][:,1]) - np.mean(lm[RIGHT_EYEBROW][:,1])) / fh
        return float(np.clip((ld-0.05)/0.10, 0, 1)), float(np.clip((rd-0.05)/0.10, 0, 1))
    
    def _furrow(self, lm):
        d = np.linalg.norm(lm[107][:2] - lm[336][:2])
        fw = abs(lm[454, 0] - lm[234, 0])
        return float(np.clip(1 - (d/fw - 0.10)/0.20, 0, 1)) if fw > 1e-6 else 0.0
    
    def _mouth_open(self, lm):
        v = np.linalg.norm(lm[13][:2] - lm[14][:2])
        h = np.linalg.norm(lm[61][:2] - lm[291][:2])
        return float(np.clip(v/h/0.7, 0, 1)) if h > 1e-6 else 0.0
    
    def _mouth_width(self, lm):
        mw = np.linalg.norm(lm[61][:2] - lm[291][:2])
        fw = abs(lm[454, 0] - lm[234, 0])
        return float(np.clip((mw/fw - 0.25)/0.30, 0, 1)) if fw > 1e-6 else 0.5
    
    def _pucker(self, lm):
        mw = np.linalg.norm(lm[61][:2] - lm[291][:2])
        fw = abs(lm[454, 0] - lm[234, 0])
        return float(np.clip(1 - (mw/fw - 0.20)/0.20, 0, 1)) if fw > 1e-6 else 0.0
    
    def _smile(self, lm):
        mc = (lm[13, 1] + lm[14, 1]) / 2
        cr = mc - (lm[61, 1] + lm[291, 1]) / 2
        fh = abs(lm[152, 1] - lm[10, 1])
        return float(np.clip((cr/fh + 0.01)/0.05, 0, 1)) if fh > 1e-6 else 0.0
    
    def _head_pose(self, matrix):
        r = np.array(matrix)[:3, :3]
        sy = np.sqrt(r[0,0]**2 + r[1,0]**2)
        if sy > 1e-6:
            pitch = np.arctan2(r[2,1], r[2,2])
            yaw = np.arctan2(-r[2,0], sy)
            roll = np.arctan2(r[1,0], r[0,0])
        else:
            pitch = np.arctan2(-r[1,2], r[1,1])
            yaw = np.arctan2(-r[2,0], sy)
            roll = 0
        return (float(np.clip(np.degrees(pitch), -90, 90)),
                float(np.clip(np.degrees(yaw), -90, 90)),
                float(np.clip(np.degrees(roll), -90, 90)))
    
    def close(self):
        self.detector.close()


# ============== 摄像头封装 ==============

def open_camera(camera_id: int = 0, use_csi: bool = False, width: int = 640, height: int = 480):
    """打开摄像头"""
    if use_csi:
        # 使用 libcamera (树莓派 CSI 摄像头)
        pipeline = (
            f"libcamerasrc ! "
            f"video/x-raw,width={width},height={height},framerate=30/1 ! "
            f"videoconvert ! appsink"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        # USB 摄像头
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    return cap


# ============== 终端显示 ==============

FEATURE_NAMES = [
    "L_Eye    ", "R_Eye    ", "Gaze_H   ", "Gaze_V   ",
    "L_Brow   ", "R_Brow   ", "Furrow   ",
    "Mouth_O  ", "Mouth_W  ", "Pucker   ", "Smile    ",
    "Pitch    ", "Yaw      ", "Roll     "
]

def clear_screen():
    """清屏"""
    os.system('clear' if os.name != 'nt' else 'cls')

def print_features(features: FaceFeatures, fps: float, frame_count: int):
    """在终端打印特征"""
    arr = features.to_array()
    
    # 移动光标到开头 (ANSI escape)
    print("\033[H", end="")
    
    print("=" * 60)
    print("  树莓派 MediaPipe 面部特征测试")
    print("=" * 60)
    print(f"  帧数: {frame_count:6d}  |  FPS: {fps:5.1f}  |  按 Ctrl+C 退出")
    print("-" * 60)
    print()
    
    # 眼睛特征
    print("  【眼睛特征】")
    for i in range(4):
        bar = "█" * int(abs(arr[i]) * 20) if i < 2 else "█" * int((arr[i] + 1) * 10)
        print(f"    {FEATURE_NAMES[i]}: {arr[i]:+7.3f}  [{bar:<20}]")
    print()
    
    # 眉毛特征
    print("  【眉毛特征】")
    for i in range(4, 7):
        bar = "█" * int(arr[i] * 20)
        print(f"    {FEATURE_NAMES[i]}: {arr[i]:+7.3f}  [{bar:<20}]")
    print()
    
    # 嘴巴特征
    print("  【嘴巴特征】")
    for i in range(7, 11):
        bar = "█" * int(arr[i] * 20)
        print(f"    {FEATURE_NAMES[i]}: {arr[i]:+7.3f}  [{bar:<20}]")
    print()
    
    # 头部姿态
    print("  【头部姿态】")
    for i in range(11, 14):
        norm = (arr[i] + 90) / 180  # 归一化到 [0, 1]
        bar = "█" * int(norm * 20)
        print(f"    {FEATURE_NAMES[i]}: {arr[i]:+7.1f}°  [{bar:<20}]")
    print()
    
    print("-" * 60)
    print("  14维特征向量:")
    print(f"  {arr.tolist()}")
    print("=" * 60)


def print_no_face(fps: float, frame_count: int):
    """未检测到人脸时的显示"""
    print("\033[H", end="")
    print("=" * 60)
    print("  树莓派 MediaPipe 面部特征测试")
    print("=" * 60)
    print(f"  帧数: {frame_count:6d}  |  FPS: {fps:5.1f}  |  按 Ctrl+C 退出")
    print("-" * 60)
    print()
    print("  ⚠️  未检测到人脸，请面对摄像头")
    print()
    print("=" * 60)


# ============== 主程序 ==============

def main():
    parser = argparse.ArgumentParser(description="树莓派 MediaPipe 测试")
    parser.add_argument("--camera", type=int, default=0, help="摄像头 ID (默认 0)")
    parser.add_argument("--csi", action="store_true", help="使用 CSI 摄像头 (libcamera)")
    parser.add_argument("--width", type=int, default=640, help="分辨率宽度")
    parser.add_argument("--height", type=int, default=480, help="分辨率高度")
    args = parser.parse_args()
    
    print("初始化中...")
    
    # 打开摄像头
    cap = open_camera(args.camera, args.csi, args.width, args.height)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        print("  - USB 摄像头: 检查是否连接")
        print("  - CSI 摄像头: 使用 --csi 参数，并确保已启用摄像头")
        return
    
    # 初始化特征提取器
    extractor = SimpleFeatureExtractor()
    
    print("初始化完成，开始检测...")
    time.sleep(1)
    clear_screen()
    
    frame_count = 0
    fps_history = []
    last_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取帧")
                break
            
            # 提取特征
            features = extractor.extract(frame)
            
            # 计算 FPS
            now = time.time()
            fps_history.append(1.0 / (now - last_time + 1e-6))
            if len(fps_history) > 30:
                fps_history.pop(0)
            fps = sum(fps_history) / len(fps_history)
            last_time = now
            frame_count += 1
            
            # 显示
            if features:
                print_features(features, fps, frame_count)
            else:
                print_no_face(fps, frame_count)
    
    except KeyboardInterrupt:
        print("\n\n检测结束")
    
    finally:
        cap.release()
        extractor.close()
        print(f"总帧数: {frame_count}")


if __name__ == "__main__":
    main()
