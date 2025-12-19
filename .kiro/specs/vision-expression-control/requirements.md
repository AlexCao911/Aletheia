# Requirements Document

## Introduction

本项目旨在升级现有的机器人表情控制系统，通过集成摄像头和树莓派5，实现基于视觉输入的实时表情生成。系统将使用 LNN-S4（Liquid Neural Network with S4 layers）模型处理视频流，输出21个舵机的角度控制数据，驱动机器人面部表情与外界视觉刺激实时同步。

## Glossary

- **Raspberry Pi 5 (RPi5)**: 主控制单元，负责视觉数据处理和模型推理
- **Pico**: Raspberry Pi Pico 微控制器，负责舵机 PWM 控制
- **LNN-S4**: Liquid Neural Network with Structured State Space (S4) layers，一种适合时序数据处理的轻量级神经网络架构
- **Servo**: 舵机，通过 PWM 信号控制角度的执行器
- **Expression Controller**: 表情控制系统，包含眉毛(4舵机)、眼睛(6舵机)、嘴巴(11舵机)共21个舵机
- **Video Stream**: 摄像头采集的实时视频数据流
- **Inference Latency**: 模型推理延迟，从输入到输出的时间
- **UART**: 通用异步收发传输器，用于 RPi5 与 Pico 之间的串行通信
- **Frame Rate**: 视频帧率，每秒处理的图像帧数

## Requirements

### Requirement 1: 硬件架构设计

**User Story:** 作为系统集成工程师，我希望有一个清晰的硬件架构方案，以便正确连接摄像头、树莓派5和现有的Pico舵机控制系统。

#### Acceptance Criteria

1. THE Hardware Architecture SHALL define camera module selection compatible with Raspberry Pi 5 CSI interface
2. THE Hardware Architecture SHALL use hierarchical communication where Raspberry Pi 5 connects only to MouthMaster Pico via USB serial
3. THE Hardware Architecture SHALL maintain the existing GPIO-based communication from MouthMaster Pico to Eyes Pico and Brows Pico
4. THE Hardware Architecture SHALL preserve existing PCB and circuit designs without modification
5. WHEN Raspberry Pi 5 sends servo angle commands THEN the MouthMaster Pico SHALL distribute commands to Eyes and Brows Picos via GPIO signals
6. THE Hardware Architecture SHALL include power distribution design supporting Raspberry Pi 5, camera, and all three Pico boards simultaneously
7. THE Software Architecture SHALL provide an abstraction layer to support future direct RPi5-to-Pico communication without hardware changes

### Requirement 2: 通信协议设计

**User Story:** 作为嵌入式开发者，我希望有一个高效的通信协议，以便树莓派5能够通过MouthMaster实时控制21个舵机的角度。

#### Acceptance Criteria

1. THE Communication Protocol SHALL define a text-based command format compatible with existing USB serial interface on MouthMaster Pico
2. THE Communication Protocol SHALL extend the existing command set to support batch servo angle updates
3. THE Communication Protocol SHALL define a new command format "angles:A1,A2,...,A21" for transmitting all 21 servo angles
4. WHEN MouthMaster receives angle commands THEN it SHALL update local servos and translate commands to GPIO signals for Eyes and Brows Picos
5. THE Communication Protocol SHALL support a minimum update rate of 30 commands per second
6. THE Communication Protocol SHALL define servo angle range as 0-180 degrees with 1-degree resolution
7. WHEN parsing angle command strings THEN the system SHALL validate format and extract exactly 21 angle values
8. WHEN serializing angle commands THEN the system SHALL produce strings that can be parsed back to identical angle values

### Requirement 3: 数据采集系统

**User Story:** 作为机器学习工程师，我希望有一个数据采集系统，以便收集配对的面部特征和对应的舵机角度数据用于模型训练。

#### Acceptance Criteria

1. THE Data Collection System SHALL capture synchronized MediaPipe facial landmarks and servo angle states
2. THE Data Collection System SHALL store data in a structured format with timestamps
3. WHEN recording a training session THEN the system SHALL capture facial features at minimum 30 FPS
4. THE Data Collection System SHALL support manual annotation of expression labels (happy, sad, angry, surprised, neutral)
5. THE Data Collection System SHALL export data in formats compatible with PyTorch DataLoader
6. WHEN parsing recorded data files THEN the system SHALL validate data integrity against the defined schema
7. WHEN serializing training data to disk THEN the system SHALL use a documented JSON schema for metadata
8. THE Data Collection System SHALL record both raw video and extracted MediaPipe features for debugging

### Requirement 4: 视觉处理与模型架构

**User Story:** 作为AI研究员，我希望设计一个结合MediaPipe和LNN-S4的混合架构，以便在树莓派5上实现稳健的实时视频到表情转换。

#### Acceptance Criteria

1. THE Vision Pipeline SHALL use MediaPipe Face Mesh to extract 478 facial landmarks from video frames
2. THE Vision Pipeline SHALL extract key facial features including eye aspect ratio, mouth openness, eyebrow positions, and head pose
3. THE LNN-S4 Model SHALL accept MediaPipe extracted features as input instead of raw video frames
4. THE LNN-S4 Model SHALL output 21 servo angle values in the range 0-180
5. THE Combined Pipeline SHALL achieve end-to-end inference latency below 33ms on Raspberry Pi 5
6. THE LNN-S4 Model SHALL utilize S4 layers for temporal sequence modeling of facial feature trajectories
7. THE LNN-S4 Model SHALL have a maximum model size of 20MB for edge deployment
8. WHEN MediaPipe fails to detect a face THEN the system SHALL maintain the last valid servo positions
9. WHEN processing consecutive video frames THEN the model SHALL maintain temporal coherence in output angles
10. THE Feature Extraction Module SHALL normalize landmark coordinates relative to face bounding box

### Requirement 5: 模型训练流程

**User Story:** 作为机器学习工程师，我希望有一个完整的模型训练流程，以便从数据准备到模型导出都有清晰的步骤。

#### Acceptance Criteria

1. THE Training Pipeline SHALL include data preprocessing for MediaPipe feature normalization
2. THE Training Pipeline SHALL implement data augmentation including temporal jittering and feature noise injection
3. THE Training Pipeline SHALL use smooth L1 loss for servo angle regression
4. THE Training Pipeline SHALL support checkpoint saving and resumable training
5. WHEN training completes THEN the system SHALL export the model in ONNX format for edge deployment
6. THE Training Pipeline SHALL include validation metrics for angle prediction accuracy (MAE, RMSE)
7. THE Training Pipeline SHALL support transfer learning from pre-trained S4 weights
8. THE Training Pipeline SHALL implement early stopping based on validation loss

### Requirement 6: 实时推理系统

**User Story:** 作为系统工程师，我希望有一个实时推理系统运行在树莓派5上，以便将摄像头输入转换为舵机控制命令。

#### Acceptance Criteria

1. THE Inference System SHALL run MediaPipe Face Mesh and LNN-S4 model in a pipelined manner at minimum 30 FPS
2. THE Inference System SHALL apply temporal smoothing using exponential moving average to prevent servo jitter
3. WHEN MediaPipe fails to detect a face for more than 500ms THEN the system SHALL transition servos to neutral position
4. THE Inference System SHALL provide a configuration interface for adjusting sensitivity and smoothing parameters
5. WHEN the system starts THEN it SHALL perform MediaPipe initialization, model loading, and camera initialization within 10 seconds
6. THE Inference System SHALL log inference latency statistics for performance monitoring
7. THE Inference System SHALL support runtime switching between different trained models
8. THE Inference System SHALL implement a fallback mode using direct MediaPipe-to-servo mapping when LNN-S4 model is unavailable

### Requirement 7: 测试与验证

**User Story:** 作为质量工程师，我希望有完整的测试方案，以便验证系统各组件的正确性和性能。

#### Acceptance Criteria

1. THE Test Suite SHALL include unit tests for communication protocol encoding and decoding
2. THE Test Suite SHALL include integration tests for RPi5-Pico communication
3. THE Test Suite SHALL include performance benchmarks for model inference latency
4. WHEN running end-to-end tests THEN the system SHALL verify servo response matches expected angles within 2-degree tolerance
5. THE Test Suite SHALL include stress tests for sustained operation over 1 hour

