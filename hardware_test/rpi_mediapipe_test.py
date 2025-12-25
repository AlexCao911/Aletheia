#!/usr/bin/env python3
"""
树莓派 MediaPipe 测试 - 直接使用 expression_control.extractor

使用方法:
    python Test/rpi_mediapipe_test.py           # USB 摄像头
    python Test/rpi_mediapipe_test.py --csi     # CSI 摄像头

按 Ctrl+C 退出
"""

import argparse
import os
import sys
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from expression_control.extractor import FaceFeatureExtractor


FEATURE_NAMES = [
    "L_Eye", "R_Eye", "Gaze_H", "Gaze_V",
    "L_Brow", "R_Brow", "Furrow",
    "Mouth_O", "Mouth_W", "Pucker", "Smile",
    "Pitch", "Yaw", "Roll"
]


def open_camera(camera_id: int, use_csi: bool, width: int, height: int):
    """打开摄像头"""
    if use_csi:
        pipeline = (
            f"libcamerasrc ! video/x-raw,width={width},height={height},framerate=30/1 ! "
            f"videoconvert ! appsink"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    parser = argparse.ArgumentParser(description="树莓派 MediaPipe 测试")
    parser.add_argument("--camera", type=int, default=0, help="摄像头 ID")
    parser.add_argument("--csi", action="store_true", help="使用 CSI 摄像头")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    print("初始化摄像头...")
    cap = open_camera(args.camera, args.csi, args.width, args.height)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    print("初始化 FaceFeatureExtractor...")
    extractor = FaceFeatureExtractor()

    print("开始检测，按 Ctrl+C 退出\n")

    frame_count = 0
    fps_list = []
    last_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            features = extractor.extract(frame)

            # FPS
            now = time.time()
            fps_list.append(1.0 / (now - last_time + 1e-6))
            if len(fps_list) > 30:
                fps_list.pop(0)
            fps = np.mean(fps_list)
            last_time = now
            frame_count += 1

            # 输出
            print(f"\033[2J\033[H", end="")  # 清屏
            print(f"帧: {frame_count}  FPS: {fps:.1f}")
            print("-" * 50)

            if features:
                arr = features.to_array()
                for i, name in enumerate(FEATURE_NAMES):
                    print(f"  {name:8s}: {arr[i]:+8.3f}")
                print("-" * 50)
                print(f"向量: {arr.tolist()}")
            else:
                print("  未检测到人脸")

    except KeyboardInterrupt:
        print("\n结束")
    finally:
        cap.release()
        extractor.close()


if __name__ == "__main__":
    main()
