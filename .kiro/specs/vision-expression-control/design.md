# Design Document: Vision-Driven Expression Control System

## Overview

本设计文档描述了一个基于视觉输入的机器人表情控制系统升级方案。系统通过摄像头采集视频流，使用 MediaPipe 提取面部特征，再通过 LNN-S4 模型进行时序建模，最终输出21个舵机的角度控制数据。

### System Architecture

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

## Architecture

### Hardware Architecture

#### 1. Camera Module Selection

推荐使用 **Raspberry Pi Camera Module 3** 或兼容的 CSI 摄像头：
- 分辨率：1080p @ 30fps（实际使用 640x480 降采样）
- 接口：CSI-2（直连 RPi5）
- 延迟：< 50ms（硬件编码）

备选方案：USB 摄像头（如 Logitech C920），但延迟略高。

#### 2. Power Distribution

```
┌─────────────────────────────────────────────────────────────┐
│                    5V 10A Power Supply                       │
│                           │                                  │
│         ┌─────────────────┼─────────────────┐               │
│         ▼                 ▼                 ▼               │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│   │  RPi5     │    │  Pico x3  │    │  Servos   │          │
│   │  5V/3A    │    │  5V/0.5A  │    │  5V/6A    │          │
│   └───────────┘    └───────────┘    └───────────┘          │
└─────────────────────────────────────────────────────────────┘
```

#### 3. Communication Topology

```
RPi5 ──USB Serial──▶ MouthMaster Pico
                          │
                          ├──GPIO 16,17 (SDA,SCL)──▶ Eyes Pico
                          │
                          └──GPIO 18,19 (BSDA,BSCL)──▶ Brows Pico
```

### Software Architecture

#### Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Data Collect│  │  Training   │  │  Inference Engine   │ │
│  │    Tool     │  │  Pipeline   │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Processing Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  MediaPipe  │  │   LNN-S4    │  │  Temporal Smoother  │ │
│  │  Face Mesh  │  │   Model     │  │       (EMA)         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Communication Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Camera    │  │    Serial   │  │  Protocol Encoder   │ │
│  │   Driver    │  │    Port     │  │     /Decoder        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Hardware Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  CSI Camera │  │  USB Port   │  │   Pico Boards       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. MediaPipe Feature Extractor

```python
class FaceFeatureExtractor:
    """从 MediaPipe Face Mesh 提取关键面部特征"""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """初始化 MediaPipe Face Mesh"""
        pass
    
    def extract(self, frame: np.ndarray) -> Optional[FaceFeatures]:
        """
        从视频帧提取面部特征
        
        Args:
            frame: BGR 格式的视频帧 (H, W, 3)
            
        Returns:
            FaceFeatures 对象，包含归一化的面部特征；
            如果未检测到人脸则返回 None
        """
        pass
    
    def get_feature_vector(self, features: FaceFeatures) -> np.ndarray:
        """
        将 FaceFeatures 转换为模型输入向量
        
        Returns:
            形状为 (feature_dim,) 的特征向量
        """
        pass
```

### 2. LNN-S4 Model Architecture

基于 [liquid-s4](https://github.com/raminmh/liquid-s4) 和 [ncps](https://github.com/mlech26l/ncps) 官方库实现。

#### Model Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Liquid-S4 Expression Model                            │
│                                                                          │
│  Input: FaceFeatures (14-dim)                                           │
│         ↓                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Feature Embedding Layer                              │   │
│  │              Linear(14 → 64) + LayerNorm + GELU                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         ↓                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    S4 Block × 2                                   │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │  S4Layer(d_model=64, d_state=32)                           │  │   │
│  │  │  - Structured State Space: x' = Ax + Bu, y = Cx + Du       │  │   │
│  │  │  - HiPPO initialization for long-range dependencies        │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  │         ↓ + Residual                                              │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │  FFN: Linear(64→256) + GELU + Dropout + Linear(256→64)     │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  │         ↓ + Residual + LayerNorm                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         ↓                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Liquid Time-Constant (LTC) Layer                     │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │  LTCCell from ncps library                                 │  │   │
│  │  │  - Neural ODE: τ(x)·dx/dt = -x + f(x, I)                  │  │   │
│  │  │  - Adaptive time constants for smooth transitions          │  │   │
│  │  │  - 32 liquid neurons with sparse connectivity              │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         ↓                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Output Head                                          │   │
│  │              Linear(32 → 21) + Sigmoid × 180                     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         ↓                                                                │
│  Output: Servo Angles (21-dim, range [0, 180])                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Model Implementation

```python
import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP

# S4 Layer implementation (from liquid-s4 repo)
class S4Layer(nn.Module):
    """Structured State Space Sequence Layer"""
    
    def __init__(self, d_model: int, d_state: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # S4 parameters (HiPPO initialization)
        self.A = nn.Parameter(torch.randn(d_state, d_state) / d_state)
        self.B = nn.Parameter(torch.randn(d_state, d_model) / d_model)
        self.C = nn.Parameter(torch.randn(d_model, d_state) / d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, state=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            state: (batch, d_state) previous state
        Returns:
            y: (batch, seq_len, d_model)
            new_state: (batch, d_state)
        """
        batch, seq_len, _ = x.shape
        
        if state is None:
            state = torch.zeros(batch, self.d_state, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            # State space update: x' = Ax + Bu
            state = torch.tanh(state @ self.A.T + x[:, t] @ self.B.T)
            # Output: y = Cx + Du
            y = state @ self.C.T + x[:, t] * self.D
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)
        return self.dropout(y), state


class LiquidS4Model(nn.Module):
    """Liquid-S4 Model for Expression Control"""
    
    def __init__(self, config: 'LNNS4Config'):
        super().__init__()
        self.config = config
        
        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )
        
        # S4 blocks
        self.s4_layers = nn.ModuleList([
            S4Layer(config.hidden_dim, config.state_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # FFN for each S4 block
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 4, config.hidden_dim)
            )
            for _ in range(config.num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # Liquid Time-Constant layer (from ncps)
        wiring = AutoNCP(config.liquid_units, config.output_dim)
        self.ltc = LTC(config.hidden_dim, wiring, batch_first=True)
        
        # Output projection
        self.output_head = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.Sigmoid()  # Output in [0, 1], scale to [0, 180] later
        )
        
    def forward(self, x, s4_states=None, ltc_state=None):
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim) for single frame
            s4_states: list of (batch, state_dim) for each S4 layer
            ltc_state: (batch, liquid_units) LTC hidden state
            
        Returns:
            angles: (batch, seq_len, 21) or (batch, 21) servo angles [0, 180]
            new_s4_states: updated S4 states
            new_ltc_state: updated LTC state
        """
        # Handle single frame input
        single_frame = x.dim() == 2
        if single_frame:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # Initialize states if needed
        if s4_states is None:
            s4_states = [None] * self.config.num_layers
        
        # Feature embedding
        x = self.embedding(x)  # (batch, seq_len, hidden_dim)
        
        # S4 blocks with residual connections
        new_s4_states = []
        for i, (s4, ffn, norm) in enumerate(zip(self.s4_layers, self.ffn_layers, self.layer_norms)):
            residual = x
            x, new_state = s4(x, s4_states[i])
            x = residual + x
            x = norm(x + ffn(x))
            new_s4_states.append(new_state)
        
        # LTC layer for smooth temporal dynamics
        x, new_ltc_state = self.ltc(x, ltc_state)
        
        # Output head
        angles = self.output_head(x) * 180.0  # Scale to [0, 180]
        
        if single_frame:
            angles = angles.squeeze(1)
        
        return angles, new_s4_states, new_ltc_state
    
    def reset_states(self):
        """Reset all hidden states for new sequence"""
        return None, None


class LNNS4Inference:
    """Inference wrapper for ONNX deployment"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.s4_states = None
        self.ltc_state = None
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Single frame inference
        
        Args:
            features: (14,) face feature vector
            
        Returns:
            angles: (21,) servo angles [0, 180]
        """
        inputs = {
            'features': features.reshape(1, 1, -1).astype(np.float32),
        }
        if self.s4_states is not None:
            inputs['s4_states'] = self.s4_states
        if self.ltc_state is not None:
            inputs['ltc_state'] = self.ltc_state
            
        outputs = self.session.run(None, inputs)
        angles, self.s4_states, self.ltc_state = outputs
        
        return angles.squeeze()
    
    def reset(self):
        """Reset hidden states"""
        self.s4_states = None
        self.ltc_state = None
```

### 3. Servo Command Protocol

```python
class ServoCommandProtocol:
    """舵机命令协议编解码器"""
    
    SERVO_ORDER = [
        # Mouth (11): MouthMaster Pico
        "JL", "JR", "LUL", "LUR", "LLL", "LLR", 
        "CUL", "CUR", "CLL", "CLR", "TON",
        # Eyes (6): Eyes Pico
        "LR", "UD", "TL", "BL", "TR", "BR",
        # Brows (4): Brows Pico
        "LO", "LI", "RI", "RO"
    ]
    
    @staticmethod
    def encode(angles: Dict[str, int]) -> str:
        """
        将舵机角度字典编码为命令字符串
        
        Args:
            angles: 舵机名称到角度的映射
            
        Returns:
            格式为 "angles:A1,A2,...,A21" 的命令字符串
        """
        pass
    
    @staticmethod
    def decode(command: str) -> Optional[Dict[str, int]]:
        """
        将命令字符串解码为舵机角度字典
        
        Args:
            command: 格式为 "angles:A1,A2,...,A21" 的命令字符串
            
        Returns:
            舵机名称到角度的映射；格式错误时返回 None
        """
        pass
```

### 4. Temporal Smoother

```python
class TemporalSmoother:
    """指数移动平均时序平滑器"""
    
    def __init__(self, alpha: float = 0.3, num_servos: int = 21):
        """
        Args:
            alpha: EMA 平滑系数，越小越平滑
            num_servos: 舵机数量
        """
        pass
    
    def smooth(self, angles: np.ndarray) -> np.ndarray:
        """
        应用时序平滑
        
        Args:
            angles: 当前帧的原始角度预测
            
        Returns:
            平滑后的角度
        """
        pass
    
    def reset(self):
        """重置平滑器状态"""
        pass
```

### 5. Serial Communication Manager

```python
class SerialManager:
    """USB 串口通信管理器"""
    
    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 115200):
        """初始化串口连接"""
        pass
    
    def send_angles(self, angles: Dict[str, int]) -> bool:
        """
        发送舵机角度命令
        
        Returns:
            发送成功返回 True
        """
        pass
    
    def send_command(self, command: str) -> bool:
        """发送原始命令（兼容现有命令）"""
        pass
```

## Data Models

### FaceFeatures

```python
@dataclass
class FaceFeatures:
    """MediaPipe 提取的面部特征"""
    
    # 眼睛特征
    left_eye_aspect_ratio: float      # 左眼开合度 [0, 1]
    right_eye_aspect_ratio: float     # 右眼开合度 [0, 1]
    eye_gaze_horizontal: float        # 水平视线方向 [-1, 1]
    eye_gaze_vertical: float          # 垂直视线方向 [-1, 1]
    
    # 眉毛特征
    left_eyebrow_height: float        # 左眉高度 [0, 1]
    right_eyebrow_height: float       # 右眉高度 [0, 1]
    eyebrow_furrow: float             # 眉头皱起程度 [0, 1]
    
    # 嘴巴特征
    mouth_open_ratio: float           # 嘴巴张开度 [0, 1]
    mouth_width_ratio: float          # 嘴巴宽度 [0, 1]
    lip_pucker: float                 # 嘴唇噘起程度 [0, 1]
    smile_intensity: float            # 微笑强度 [0, 1]
    
    # 头部姿态
    head_pitch: float                 # 俯仰角 (degrees)
    head_yaw: float                   # 偏航角 (degrees)
    head_roll: float                  # 翻滚角 (degrees)
    
    # 原始关键点（用于调试）
    landmarks: Optional[np.ndarray] = None  # (478, 3)
    
    timestamp: float = 0.0            # 时间戳
```

### TrainingDataSample

```python
@dataclass
class TrainingDataSample:
    """训练数据样本"""
    
    timestamp: float                  # 时间戳
    face_features: FaceFeatures       # 面部特征
    servo_angles: Dict[str, int]      # 21个舵机角度
    expression_label: Optional[str]   # 表情标签 (happy, sad, angry, surprised, neutral)
    video_frame_path: Optional[str]   # 原始视频帧路径（用于调试）
```

### TrainingDataset Schema (JSON)

```json
{
  "version": "1.0",
  "created_at": "2024-01-01T00:00:00Z",
  "total_samples": 10000,
  "fps": 30,
  "servo_order": ["JL", "JR", "..."],
  "samples": [
    {
      "timestamp": 0.0,
      "features": {
        "left_eye_aspect_ratio": 0.3,
        "right_eye_aspect_ratio": 0.3,
        "...": "..."
      },
      "servo_angles": [90, 90, "..."],
      "expression_label": "neutral"
    }
  ]
}
```

### Model Configuration

```python
@dataclass
class LNNS4Config:
    """LNN-S4 模型配置"""
    
    # 输入输出
    input_dim: int = 14               # FaceFeatures 特征维度
    output_dim: int = 21              # 舵机数量
    
    # S4 层配置
    hidden_dim: int = 64              # 隐藏层维度
    state_dim: int = 32               # S4 状态维度
    num_layers: int = 2               # S4 层数
    
    # Liquid 配置 (ncps)
    liquid_units: int = 32            # Liquid 神经元数量
    
    # 训练配置
    sequence_length: int = 16         # 时序窗口长度
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
```

## Training Pipeline

### Complete Training Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                                 │
│                                                                          │
│  1. Data Collection                                                      │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐                       │
│     │  Camera  │───▶│MediaPipe │───▶│  Record  │                       │
│     │  Stream  │    │ Extract  │    │  to Disk │                       │
│     └──────────┘    └──────────┘    └──────────┘                       │
│                                           │                              │
│  2. Data Preprocessing                    ▼                              │
│     ┌──────────────────────────────────────────────────────────────┐   │
│     │  Load JSON → Normalize Features → Create Sequences → Split   │   │
│     │  (train: 80%, val: 10%, test: 10%)                           │   │
│     └──────────────────────────────────────────────────────────────┘   │
│                                           │                              │
│  3. Training Loop                         ▼                              │
│     ┌──────────────────────────────────────────────────────────────┐   │
│     │  for epoch in epochs:                                         │   │
│     │    for batch in train_loader:                                 │   │
│     │      features, targets = batch                                │   │
│     │      pred = model(features)                                   │   │
│     │      loss = smooth_l1_loss(pred, targets)                     │   │
│     │      loss.backward()                                          │   │
│     │      optimizer.step()                                         │   │
│     │    validate(val_loader)                                       │   │
│     │    early_stopping_check()                                     │   │
│     │    save_checkpoint()                                          │   │
│     └──────────────────────────────────────────────────────────────┘   │
│                                           │                              │
│  4. Export                                ▼                              │
│     ┌──────────────────────────────────────────────────────────────┐   │
│     │  PyTorch → ONNX → Optimize → Quantize (optional)             │   │
│     └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Collection Tool

```python
class DataCollector:
    """数据采集工具"""
    
    def __init__(self, camera_id: int = 0, serial_port: str = "/dev/ttyACM0"):
        self.camera = cv2.VideoCapture(camera_id)
        self.extractor = FaceFeatureExtractor()
        self.serial = SerialManager(serial_port)
        self.samples = []
        
    def record_session(self, duration_seconds: float, expression_label: str = None):
        """
        录制一段训练数据
        
        操作流程：
        1. 人工操作舵机（通过现有命令或手动）
        2. 同时录制面部特征和舵机状态
        """
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            # 提取面部特征
            features = self.extractor.extract(frame)
            if features is None:
                continue
            
            # 获取当前舵机角度（需要从 Pico 读取）
            servo_angles = self.serial.read_current_angles()
            
            # 保存样本
            sample = TrainingDataSample(
                timestamp=time.time() - start_time,
                face_features=features,
                servo_angles=servo_angles,
                expression_label=expression_label,
                video_frame_path=f"frames/{frame_count:06d}.jpg"
            )
            self.samples.append(sample)
            
            # 保存原始帧（用于调试）
            cv2.imwrite(sample.video_frame_path, frame)
            frame_count += 1
            
        return len(self.samples)
    
    def save_dataset(self, output_path: str):
        """保存数据集为 JSON 格式"""
        dataset = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "total_samples": len(self.samples),
            "fps": 30,
            "servo_order": ServoCommandProtocol.SERVO_ORDER,
            "samples": [s.to_dict() for s in self.samples]
        }
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
```

### PyTorch Dataset

```python
class ExpressionDataset(torch.utils.data.Dataset):
    """表情控制训练数据集"""
    
    def __init__(self, json_path: str, sequence_length: int = 16, augment: bool = False):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.samples = data['samples']
        self.sequence_length = sequence_length
        self.augment = augment
        
        # 预处理：创建序列
        self.sequences = self._create_sequences()
        
    def _create_sequences(self):
        """将样本切分为固定长度序列"""
        sequences = []
        for i in range(len(self.samples) - self.sequence_length + 1):
            seq = self.samples[i:i + self.sequence_length]
            sequences.append(seq)
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 提取特征和目标
        features = np.array([s['features'] for s in seq], dtype=np.float32)
        targets = np.array([s['servo_angles'] for s in seq], dtype=np.float32)
        
        # 数据增强
        if self.augment:
            features = self._augment_features(features)
        
        return torch.from_numpy(features), torch.from_numpy(targets)
    
    def _augment_features(self, features):
        """数据增强：时序抖动 + 特征噪声"""
        # 时序抖动：随机跳过帧
        if np.random.random() < 0.3:
            skip = np.random.randint(1, 3)
            features = features[::skip]
            # 插值回原长度
            features = np.interp(
                np.linspace(0, len(features)-1, self.sequence_length),
                np.arange(len(features)),
                features.T
            ).T
        
        # 特征噪声
        noise = np.random.normal(0, 0.02, features.shape)
        features = features + noise
        
        return features.astype(np.float32)
```

### Training Script

```python
class Trainer:
    """模型训练器"""
    
    def __init__(self, config: LNNS4Config, train_path: str, val_path: str):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据加载
        self.train_dataset = ExpressionDataset(train_path, config.sequence_length, augment=True)
        self.val_dataset = ExpressionDataset(val_path, config.sequence_length, augment=False)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size)
        
        # 模型
        self.model = LiquidS4Model(config).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # 损失函数
        self.criterion = nn.SmoothL1Loss()
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for features, targets in self.train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred, _, _ = self.model(features)
            loss = self.criterion(pred, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                pred, _, _ = self.model(features)
                loss = self.criterion(pred, targets)
                
                total_loss += loss.item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 计算指标
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        
        return total_loss / len(self.val_loader), mae, rmse
    
    def train(self, checkpoint_dir: str = "checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            val_loss, mae, rmse = self.validate()
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, MAE: {mae:.2f}°, RMSE: {rmse:.2f}°")
            
            # Checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(f"{checkpoint_dir}/best_model.pt")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.best_val_loss
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
    
    def export_onnx(self, output_path: str):
        """导出 ONNX 模型"""
        self.model.eval()
        
        # Dummy input
        dummy_input = torch.randn(1, 1, self.config.input_dim).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=['features'],
            output_names=['angles', 's4_states', 'ltc_state'],
            dynamic_axes={
                'features': {0: 'batch', 1: 'seq_len'},
                'angles': {0: 'batch', 1: 'seq_len'}
            },
            opset_version=14
        )
        print(f"Model exported to {output_path}")
```

### Training CLI

```bash
# 数据采集
python -m expression_control.collect --duration 300 --label happy --output data/happy_session.json

# 合并数据集
python -m expression_control.merge_datasets data/*.json --output data/full_dataset.json

# 划分数据集
python -m expression_control.split_dataset data/full_dataset.json --train 0.8 --val 0.1 --test 0.1

# 训练
python -m expression_control.train \
    --train data/train.json \
    --val data/val.json \
    --epochs 100 \
    --batch-size 32 \
    --checkpoint-dir checkpoints/

# 导出 ONNX
python -m expression_control.export --checkpoint checkpoints/best_model.pt --output models/expression_model.onnx

# 评估
python -m expression_control.evaluate --model models/expression_model.onnx --test data/test.json
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Based on the acceptance criteria analysis, the following correctness properties have been identified:

### Property 1: Protocol Round-Trip Consistency

*For any* valid dictionary of 21 servo angles (each in range 0-180), encoding to command string and then decoding back SHALL produce an identical dictionary of angles.

**Validates: Requirements 2.3, 2.7, 2.8**

### Property 2: Angle Range Validation

*For any* angle value outside the range [0, 180], the protocol encoder SHALL reject the input and raise a validation error.

**Validates: Requirements 2.6**

### Property 3: Training Data Round-Trip Consistency

*For any* valid TrainingDataSample, serializing to JSON and then deserializing SHALL produce an equivalent TrainingDataSample with identical feature values and servo angles.

**Validates: Requirements 3.6, 3.7**

### Property 4: Feature Extraction Completeness

*For any* video frame containing a detectable face, the FaceFeatureExtractor SHALL produce a FaceFeatures object containing all 14 required feature fields (eye ratios, eyebrow heights, mouth features, head pose).

**Validates: Requirements 4.2**

### Property 5: Model Output Validity

*For any* valid input feature vector, the LNN-S4 model SHALL output exactly 21 values, each within the range [0, 180].

**Validates: Requirements 4.4**

### Property 6: Face Detection Fallback

*For any* sequence of frames where face detection fails, the inference system SHALL output the same servo angles as the last successful detection.

**Validates: Requirements 4.8**

### Property 7: Temporal Coherence

*For any* two consecutive feature vectors with Euclidean distance less than threshold ε, the model output angles SHALL differ by no more than δ degrees per servo (where δ is configurable).

**Validates: Requirements 4.9**

### Property 8: Feature Normalization Invariance

*For any* face detected at different positions and scales within the frame, the extracted feature values SHALL remain within the normalized range [0, 1] for ratios and [-180, 180] for angles.

**Validates: Requirements 4.10**

### Property 9: Checkpoint Round-Trip Consistency

*For any* trained model state, saving to checkpoint and loading back SHALL produce a model that generates identical outputs for the same inputs.

**Validates: Requirements 5.4**

### Property 10: Metric Computation Correctness

*For any* set of predicted and ground truth angle arrays, the computed MAE SHALL equal the mean of absolute differences, and RMSE SHALL equal the square root of mean squared differences.

**Validates: Requirements 5.6**

### Property 11: EMA Smoothing Bounds

*For any* sequence of input angles, the EMA smoother output SHALL always be bounded between the minimum and maximum of the input history within the smoothing window.

**Validates: Requirements 6.2**

### Property 12: Fallback Mode Validity

*For any* valid FaceFeatures input, the direct MediaPipe-to-servo fallback mapping SHALL produce 21 servo angles, each within the range [0, 180].

**Validates: Requirements 6.8**

### Property 13: Angle Tolerance Verification

*For any* commanded servo angle and measured response, the difference SHALL be within 2 degrees tolerance.

**Validates: Requirements 7.4**

## Error Handling

### Camera Errors

| Error Condition | Detection | Recovery Action |
|----------------|-----------|-----------------|
| Camera disconnected | OpenCV read() returns False | Log error, retry connection every 1s, maintain last servo positions |
| Frame corruption | MediaPipe returns None landmarks | Skip frame, use last valid features |
| Low light / blur | Face detection confidence < 0.5 | Use last valid features, log warning |

### Communication Errors

| Error Condition | Detection | Recovery Action |
|----------------|-----------|-----------------|
| Serial port unavailable | pyserial exception | Retry connection every 2s, queue commands |
| Command timeout | No response within 100ms | Retry command once, then skip |
| Invalid command format | Parse error on Pico | Discard command, log error |

### Model Errors

| Error Condition | Detection | Recovery Action |
|----------------|-----------|-----------------|
| Model file missing | FileNotFoundError | Fall back to direct MediaPipe mapping |
| ONNX runtime error | onnxruntime exception | Fall back to direct MediaPipe mapping |
| Output out of range | angle < 0 or angle > 180 | Clamp to valid range, log warning |

### Graceful Degradation

```
Full Pipeline:     Camera → MediaPipe → LNN-S4 → Smoother → Serial → Pico
                                ↓ (model error)
Fallback Mode:     Camera → MediaPipe → Direct Mapping → Smoother → Serial → Pico
                                ↓ (face detection error)
Safe Mode:         Maintain last valid positions → Neutral after 500ms timeout
```

## Testing Strategy

### Dual Testing Approach

本项目采用单元测试和属性测试相结合的方法：
- **单元测试**：验证特定示例和边界情况
- **属性测试**：验证在所有有效输入上都应成立的通用属性

### Property-Based Testing Framework

使用 **Hypothesis** (Python) 作为属性测试框架：

```python
# 安装
pip install hypothesis

# 配置：每个属性测试运行至少 100 次迭代
from hypothesis import settings, given
import hypothesis.strategies as st

@settings(max_examples=100)
@given(...)
def test_property_xxx():
    pass
```

### Test Categories

#### 1. Protocol Tests (Property-Based)

```python
# **Feature: vision-expression-control, Property 1: Protocol Round-Trip Consistency**
@given(angles=st.dictionaries(
    keys=st.sampled_from(ServoCommandProtocol.SERVO_ORDER),
    values=st.integers(min_value=0, max_value=180),
    min_size=21, max_size=21
))
def test_protocol_round_trip(angles):
    encoded = ServoCommandProtocol.encode(angles)
    decoded = ServoCommandProtocol.decode(encoded)
    assert decoded == angles

# **Feature: vision-expression-control, Property 2: Angle Range Validation**
@given(invalid_angle=st.integers().filter(lambda x: x < 0 or x > 180))
def test_angle_range_validation(invalid_angle):
    angles = {name: 90 for name in ServoCommandProtocol.SERVO_ORDER}
    angles["JL"] = invalid_angle
    with pytest.raises(ValueError):
        ServoCommandProtocol.encode(angles)
```

#### 2. Data Serialization Tests (Property-Based)

```python
# **Feature: vision-expression-control, Property 3: Training Data Round-Trip Consistency**
@given(sample=training_data_sample_strategy())
def test_training_data_round_trip(sample):
    json_str = sample.to_json()
    loaded = TrainingDataSample.from_json(json_str)
    assert loaded.servo_angles == sample.servo_angles
    assert np.allclose(loaded.face_features.to_array(), sample.face_features.to_array())
```

#### 3. Model Output Tests (Property-Based)

```python
# **Feature: vision-expression-control, Property 5: Model Output Validity**
@given(features=st.lists(st.floats(min_value=-1, max_value=1), min_size=14, max_size=14))
def test_model_output_validity(features):
    model = LNNS4Model("model.onnx")
    output, _ = model.predict(np.array(features))
    assert output.shape == (21,)
    assert np.all(output >= 0) and np.all(output <= 180)
```

#### 4. Temporal Smoothing Tests (Property-Based)

```python
# **Feature: vision-expression-control, Property 11: EMA Smoothing Bounds**
@given(angle_sequence=st.lists(st.floats(min_value=0, max_value=180), min_size=10, max_size=100))
def test_ema_smoothing_bounds(angle_sequence):
    smoother = TemporalSmoother(alpha=0.3, num_servos=1)
    outputs = [smoother.smooth(np.array([a]))[0] for a in angle_sequence]
    for i, out in enumerate(outputs):
        window = angle_sequence[:i+1]
        assert min(window) <= out <= max(window)
```

#### 5. Unit Tests (Example-Based)

```python
def test_feature_extractor_with_sample_face():
    """Test feature extraction with a known face image"""
    extractor = FaceFeatureExtractor()
    frame = cv2.imread("test_data/sample_face.jpg")
    features = extractor.extract(frame)
    assert features is not None
    assert 0 <= features.left_eye_aspect_ratio <= 1
    assert 0 <= features.mouth_open_ratio <= 1

def test_serial_manager_connection():
    """Test serial port connection (integration)"""
    manager = SerialManager("/dev/ttyACM0")
    assert manager.is_connected()
    result = manager.send_command("on")
    assert result == True
```

### Test Data Generation Strategies

```python
# Hypothesis strategies for generating test data
def face_features_strategy():
    return st.builds(
        FaceFeatures,
        left_eye_aspect_ratio=st.floats(0, 1),
        right_eye_aspect_ratio=st.floats(0, 1),
        eye_gaze_horizontal=st.floats(-1, 1),
        eye_gaze_vertical=st.floats(-1, 1),
        left_eyebrow_height=st.floats(0, 1),
        right_eyebrow_height=st.floats(0, 1),
        eyebrow_furrow=st.floats(0, 1),
        mouth_open_ratio=st.floats(0, 1),
        mouth_width_ratio=st.floats(0, 1),
        lip_pucker=st.floats(0, 1),
        smile_intensity=st.floats(0, 1),
        head_pitch=st.floats(-90, 90),
        head_yaw=st.floats(-90, 90),
        head_roll=st.floats(-90, 90),
    )

def servo_angles_strategy():
    return st.dictionaries(
        keys=st.sampled_from(ServoCommandProtocol.SERVO_ORDER),
        values=st.integers(0, 180),
        min_size=21, max_size=21
    )
```

### Performance Benchmarks

```python
def benchmark_inference_latency():
    """Benchmark end-to-end inference latency"""
    model = LNNS4Model("model.onnx")
    extractor = FaceFeatureExtractor()
    
    latencies = []
    for _ in range(100):
        frame = capture_frame()
        start = time.perf_counter()
        features = extractor.extract(frame)
        if features:
            angles, _ = model.predict(features.to_array())
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
    
    print(f"Mean latency: {np.mean(latencies):.2f}ms")
    print(f"P95 latency: {np.percentile(latencies, 95):.2f}ms")
    assert np.mean(latencies) < 33  # 30 FPS requirement
```

