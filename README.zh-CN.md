# Expression Control System

中文版 | [English](README.md)

基于视觉输入的机器人表情控制系统，使用 MediaPipe 和 LNN-S4 神经网络实现实时面部表情识别与舵机控制。

## 项目简介

本项目通过摄像头采集视频流，使用 MediaPipe 提取面部特征，再通过 Liquid Neural Network with S4 layers (LNN-S4) 模型进行时序建模，最终输出 21 个舵机的角度控制数据，实现机器人面部表情与外界视觉刺激的实时同步。

### 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Raspberry Pi 5                                   │
│  ┌──────────┐    ┌─────────────┐    ┌─────────┐    ┌──────────────┐    │
│  │  Camera  │───▶│  MediaPipe  │───▶│ Feature │───▶│   LNN-S4     │    │
│  │  Module  │    │  Face Mesh  │    │ Extract │    │    Model     │    │
│  └──────────┘    └─────────────┘    └─────────┘    └──────┬───────┘    │
│                                                           │             │
│                                      ┌────────────────────▼──────────┐ │
│                                      │   Servo Command Generator     │ │
│                                      │   (Temporal Smoothing + EMA)  │ │
│                                      └────────────────────┬──────────┘ │
└───────────────────────────────────────────────────────────┼─────────────┘
                                                            │ USB Serial
                                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MouthMaster Pico (主控)                             │
│  ┌──────────────┐    ┌─────────────────┐    ┌─────────────────────┐    │
│  │ USB Command  │───▶│  Command Parser │───▶│  11 Mouth Servos    │    │
│  │   Receiver   │    │  & Distributor  │    │  (JL,JR,LUL,LUR...) │    │
│  └──────────────┘    └────────┬────────┘    └─────────────────────┘    │
│                               │                                         │
│                    ┌──────────┴──────────┐                             │
│                    │    GPIO Signals     │                             │
│                    │  (SDA/SCL, BSDA/BSCL)│                            │
│                    └──────────┬──────────┘                             │
└───────────────────────────────┼─────────────────────────────────────────┘
                    ┌───────────┴───────────┐
                    ▼                       ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│       Eyes Pico             │  │       Brows Pico            │
│  ┌─────────────────────┐   │  │  ┌─────────────────────┐    │
│  │   6 Eye Servos      │   │  │  │   4 Brow Servos     │    │
│  │ (LR,UD,TL,BL,TR,BR) │   │  │  │   (LO,LI,RI,RO)     │    │
│  └─────────────────────┘   │  │  └─────────────────────┘    │
└─────────────────────────────┘  └─────────────────────────────┘
```

## 硬件要求

- **Raspberry Pi 5**: 主控制单元，负责视觉处理和模型推理
- **摄像头**: Raspberry Pi Camera Module 3 或兼容的 CSI/USB 摄像头
- **Raspberry Pi Pico × 3**:
  - MouthMaster Pico: 控制 11 个嘴部舵机，负责与 RPi5 通信
  - Eyes Pico: 控制 6 个眼部舵机
  - Brows Pico: 控制 4 个眉毛舵机
- **舵机 × 21**: 标准 PWM 舵机（0-180°）
- **电源**: 5V 10A 电源供应

## 软件依赖

### 核心依赖

```bash
# 视觉处理
mediapipe>=0.10.0
opencv-python>=4.8.0

# 深度学习
torch>=2.0.0
ncps>=0.0.7

# 模型部署
onnxruntime>=1.15.0

# 硬件通信
pyserial>=3.5

# 基础工具
numpy>=1.24.0
```

### 开发依赖

```bash
hypothesis>=6.82.0
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
isort>=5.12.0
mypy>=1.5.0
```

## 安装

### 1. 克隆仓库

```bash
git clone <repository-url>
cd expression_control
```

### 2. 安装依赖

```bash
# 生产环境
pip install -r requirements.txt

# 开发环境
pip install -r requirements-dev.txt
```

### 3. 安装包（可选）

```bash
pip install -e .
```

## 快速开始

### 1. 数据采集

使用数据采集工具录制训练数据：

```bash
expression-collect --duration 60 --output data/training_session_1.json
```

### 2. 模型训练

训练 LNN-S4 模型：

```bash
expression-train \
  --train-data data/train.json \
  --val-data data/val.json \
  --config configs/default.yaml \
  --output models/expression_model.pt
```

### 3. 模型导出

将 PyTorch 模型导出为 ONNX 格式：

```bash
expression-export \
  --model models/expression_model.pt \
  --output models/expression_model.onnx
```

### 4. 实时推理

在 Raspberry Pi 5 上运行实时推理：

```bash
expression-run \
  --model models/expression_model.onnx \
  --camera 0 \
  --serial /dev/ttyACM0 \
  --config configs/inference.yaml
```

## 项目结构

```
expression_control/
├── expression_control/          # 核心包
│   ├── __init__.py
│   ├── collector.py            # 数据采集
│   ├── config.py               # 配置管理
│   ├── data.py                 # 数据模型
│   ├── dataset.py              # PyTorch 数据集
│   ├── extractor.py            # MediaPipe 特征提取
│   ├── features.py             # 特征定义
│   ├── inference.py            # 推理引擎
│   ├── models/                 # 模型实现
│   │   ├── s4.py              # S4 层实现
│   │   └── liquid_s4.py       # Liquid-S4 模型
│   ├── protocol.py             # 通信协议
│   ├── serial_manager.py       # 串口管理
│   ├── smoother.py             # 时序平滑
│   ├── trainer.py              # 训练流程
│   └── cli/                    # 命令行工具
│       ├── train.py
│       ├── export.py
│       ├── run.py
│       └── evaluate.py
├── tests/                       # 测试
├── Brows.py                     # Brows Pico 固件
├── eyes.py                      # Eyes Pico 固件
├── MouthMaster.py               # MouthMaster Pico 固件
├── servo.py                     # 舵机驱动
├── pyproject.toml               # 项目配置
├── requirements.txt             # 生产依赖
├── requirements-dev.txt         # 开发依赖
└── README.md                    # 本文件
```

## 通信协议

### 批量角度命令

格式：`angles:A1,A2,...,A21`

其中 A1-A21 为 21 个舵机的角度值（0-180），顺序如下：

```python
# 嘴部 (11): MouthMaster Pico
JL, JR, LUL, LUR, LLL, LLR, CUL, CUR, CLL, CLR, TON

# 眼部 (6): Eyes Pico
LR, UD, TL, BL, TR, BR

# 眉毛 (4): Brows Pico
LO, LI, RI, RO
```

示例：
```
angles:90,90,80,80,90,90,80,80,80,80,90,70,100,120,60,120,60,100,80,100,80
```

### 传统命令（兼容）

```python
# LED 控制
"on"          # 打开 LED
"off"         # 关闭 LED

# 眼部控制
"eyes_move"   # 眼睛自动移动
"eyes_open"   # 睁眼
"eyes_close"  # 闭眼

# 眉毛控制
"brows_up"    # 眉毛上扬
"brows_down"  # 眉毛下垂
"brows_happy" # 开心表情
"brows_angry" # 生气表情

# 嘴部控制
"mouth_open"  # 张嘴
"mouth_closed"# 闭嘴
"smile"       # 微笑
"frown"       # 皱眉
```

## 模型架构

### LNN-S4 模型

本项目使用 Liquid Neural Network with Structured State Space (S4) layers，结合了：

1. **S4 层**: 用于长时序依赖建模
2. **Liquid Time-Constant (LTC) 层**: 提供平滑的时序动态特性
3. **特征嵌入**: 将 MediaPipe 特征映射到高维空间

模型流程：
```
Input (14-dim) → Embedding (64-dim) → S4 Blocks × 2 → LTC Layer → Output (21-dim)
```

### 特征提取

从 MediaPipe Face Mesh 提取 14 维特征：

- **眼睛** (4): 左右眼开合度、水平/垂直视线方向
- **眉毛** (3): 左右眉高度、眉头皱起程度
- **嘴巴** (4): 张开度、宽度、噘嘴程度、微笑强度
- **头部姿态** (3): 俯仰角、偏航角、翻滚角

## 配置

### 推理配置示例

```yaml
# configs/inference.yaml
model:
  path: "models/expression_model.onnx"
  input_dim: 14
  output_dim: 21

camera:
  device_id: 0
  width: 640
  height: 480
  fps: 30

serial:
  port: "/dev/ttyACM0"
  baudrate: 115200

smoother:
  alpha: 0.3  # EMA 平滑系数

mediapipe:
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5

fallback:
  face_lost_timeout: 0.5  # 人脸丢失超时（秒）
```

## 测试

运行测试套件：

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_protocol.py

# 生成覆盖率报告
pytest --cov=expression_control --cov-report=html
```

## 性能指标

- **推理延迟**: < 33ms (30 FPS)
- **模型大小**: < 20MB
- **特征提取**: MediaPipe Face Mesh @ 30 FPS
- **通信速率**: 115200 baud, 支持 30+ 命令/秒

## 开发指南

### 代码风格

使用 Black 和 isort 格式化代码：

```bash
black expression_control tests
isort expression_control tests
```

### 类型检查

使用 mypy 进行类型检查：

```bash
mypy expression_control
```

## 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头连接
   - 确认设备 ID 正确（通常为 0）
   - 检查权限：`sudo usermod -a -G video $USER`

2. **串口连接失败**
   - 检查 USB 连接
   - 确认端口名称：`ls /dev/ttyACM*`
   - 检查权限：`sudo usermod -a -G dialout $USER`

3. **MediaPipe 检测失败**
   - 确保光线充足
   - 调整 `min_detection_confidence` 参数
   - 检查摄像头焦距

4. **舵机抖动**
   - 增大 `smoother.alpha` 值（更平滑）
   - 检查电源供应是否稳定
   - 调整模型输出平滑参数

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

- [MediaPipe](https://github.com/google/mediapipe) - 面部特征提取
- [ncps](https://github.com/mlech26l/ncps) - Liquid Neural Networks
- [liquid-s4](https://github.com/raminmh/liquid-s4) - S4 层实现

## 联系方式

如有问题或建议，请通过 Issue 联系我们。
