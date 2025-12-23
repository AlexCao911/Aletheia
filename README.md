# Aletheia

[中文版](README.zh-CN.md) | English

> *"Aletheia" (ἀλήθεια) - Ancient Greek for "truth" or "disclosure". In philosophy, it represents the unconcealment of reality, the moment when what is hidden becomes visible.*

**Aletheia** is a vision-driven robotic expression control system that bridges the gap between human emotion and robotic embodiment. By leveraging MediaPipe for facial feature extraction and Liquid Neural Networks with S4 layers (LNN-S4) for temporal modeling, Aletheia enables robots to mirror human expressions in real-time with remarkable fidelity.

The system orchestrates 21 servo motors across three subsystems (eyes, eyebrows, and mouth) to create nuanced, lifelike facial expressions. Through continuous learning from visual input, Aletheia doesn't just replicate movements—it captures the essence of human expressiveness, making human-robot interaction more intuitive and emotionally resonant.

## Expression Control System

## Overview

This project captures video streams through a camera, extracts facial features using MediaPipe, performs temporal modeling with Liquid Neural Network with S4 layers (LNN-S4), and outputs angle control data for 21 servos to achieve real-time synchronization between robot facial expressions and external visual stimuli.

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
│                      MouthMaster Pico (Main Controller)                  │
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

## Hardware Requirements

- **Raspberry Pi 5**: Main control unit for vision processing and model inference
- **Camera**: Raspberry Pi Camera Module 3 or compatible CSI/USB camera
- **Raspberry Pi Pico × 3**:
  - MouthMaster Pico: Controls 11 mouth servos, handles RPi5 communication
  - Eyes Pico: Controls 6 eye servos
  - Brows Pico: Controls 4 eyebrow servos
- **Servos × 21**: Standard PWM servos (0-180°)
- **Power Supply**: 5V 10A power supply

## Software Dependencies

### Core Dependencies

```bash
# Vision Processing
mediapipe>=0.10.0
opencv-python>=4.8.0

# Deep Learning
torch>=2.0.0
ncps>=0.0.7

# Model Deployment
onnxruntime>=1.15.0

# Hardware Communication
pyserial>=3.5

# Core Utilities
numpy>=1.24.0
```

### Development Dependencies

```bash
hypothesis>=6.82.0
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
isort>=5.12.0
mypy>=1.5.0
```

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd expression_control
```

### 2. Install Dependencies

```bash
# Production environment
pip install -r requirements.txt

# Development environment
pip install -r requirements-dev.txt
```

### 3. Install Package (Optional)

```bash
pip install -e .
```

## Quick Start

### 1. Data Collection

Record training data using the data collection tool:

```bash
expression-collect --duration 60 --output data/training_session_1.json
```

### 2. Model Training

Train the LNN-S4 model:

```bash
expression-train \
  --train-data data/train.json \
  --val-data data/val.json \
  --config configs/default.yaml \
  --output models/expression_model.pt
```

### 3. Model Export

Export PyTorch model to ONNX format:

```bash
expression-export \
  --model models/expression_model.pt \
  --output models/expression_model.onnx
```

### 4. Real-time Inference

Run real-time inference on Raspberry Pi 5:

```bash
expression-run \
  --model models/expression_model.onnx \
  --camera 0 \
  --serial /dev/ttyACM0 \
  --config configs/inference.yaml
```

## Project Structure

```
expression_control/
├── expression_control/          # Core package
│   ├── __init__.py
│   ├── collector.py            # Data collection
│   ├── config.py               # Configuration management
│   ├── data.py                 # Data models
│   ├── dataset.py              # PyTorch dataset
│   ├── extractor.py            # MediaPipe feature extraction
│   ├── features.py             # Feature definitions
│   ├── inference.py            # Inference engine
│   ├── models/                 # Model implementations
│   │   ├── s4.py              # S4 layer implementation
│   │   └── liquid_s4.py       # Liquid-S4 model
│   ├── protocol.py             # Communication protocol
│   ├── serial_manager.py       # Serial port management
│   ├── smoother.py             # Temporal smoothing
│   ├── trainer.py              # Training pipeline
│   └── cli/                    # Command-line tools
│       ├── train.py
│       ├── export.py
│       ├── run.py
│       └── evaluate.py
├── tests/                       # Tests
├── Brows.py                     # Brows Pico firmware
├── eyes.py                      # Eyes Pico firmware
├── MouthMaster.py               # MouthMaster Pico firmware
├── servo.py                     # Servo driver
├── pyproject.toml               # Project configuration
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
└── README.md                    # This file
```

## Communication Protocol

### Batch Angle Command

Format: `angles:A1,A2,...,A21`

Where A1-A21 are angle values (0-180) for 21 servos in the following order:

```python
# Mouth (11): MouthMaster Pico
JL, JR, LUL, LUR, LLL, LLR, CUL, CUR, CLL, CLR, TON

# Eyes (6): Eyes Pico
LR, UD, TL, BL, TR, BR

# Brows (4): Brows Pico
LO, LI, RI, RO
```

Example:
```
angles:90,90,80,80,90,90,80,80,80,80,90,70,100,120,60,120,60,100,80,100,80
```

### Legacy Commands (Compatible)

```python
# LED Control
"on"          # Turn on LED
"off"         # Turn off LED

# Eye Control
"eyes_move"   # Auto eye movement
"eyes_open"   # Open eyes
"eyes_close"  # Close eyes

# Brow Control
"brows_up"    # Raise eyebrows
"brows_down"  # Lower eyebrows
"brows_happy" # Happy expression
"brows_angry" # Angry expression

# Mouth Control
"mouth_open"  # Open mouth
"mouth_closed"# Close mouth
"smile"       # Smile
"frown"       # Frown
```

## Model Architecture

### LNN-S4 Model

This project uses Liquid Neural Network with Structured State Space (S4) layers, combining:

1. **S4 Layers**: For long-range temporal dependency modeling
2. **Liquid Time-Constant (LTC) Layer**: Provides smooth temporal dynamics
3. **Feature Embedding**: Maps MediaPipe features to high-dimensional space

Model pipeline:
```
Input (14-dim) → Embedding (64-dim) → S4 Blocks × 2 → LTC Layer → Output (21-dim)
```

### Feature Extraction

Extract 14-dimensional features from MediaPipe Face Mesh:

- **Eyes** (4): Left/right eye aspect ratio, horizontal/vertical gaze direction
- **Eyebrows** (3): Left/right eyebrow height, eyebrow furrow intensity
- **Mouth** (4): Openness, width, lip pucker, smile intensity
- **Head Pose** (3): Pitch, yaw, roll angles

## Configuration

### Inference Configuration Example

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
  alpha: 0.3  # EMA smoothing coefficient

mediapipe:
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5

fallback:
  face_lost_timeout: 0.5  # Face lost timeout (seconds)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_protocol.py

# Generate coverage report
pytest --cov=expression_control --cov-report=html
```

## Performance Metrics

- **Inference Latency**: < 33ms (30 FPS)
- **Model Size**: < 20MB
- **Feature Extraction**: MediaPipe Face Mesh @ 30 FPS
- **Communication Rate**: 115200 baud, supports 30+ commands/sec

## Development Guide

### Code Style

Format code using Black and isort:

```bash
black expression_control tests
isort expression_control tests
```

### Type Checking

Run type checking with mypy:

```bash
mypy expression_control
```

## Troubleshooting

### Common Issues

1. **Camera Cannot Open**
   - Check camera connection
   - Verify device ID is correct (usually 0)
   - Check permissions: `sudo usermod -a -G video $USER`

2. **Serial Connection Failed**
   - Check USB connection
   - Verify port name: `ls /dev/ttyACM*`
   - Check permissions: `sudo usermod -a -G dialout $USER`

3. **MediaPipe Detection Failed**
   - Ensure adequate lighting
   - Adjust `min_detection_confidence` parameter
   - Check camera focus

4. **Servo Jitter**
   - Increase `smoother.alpha` value (more smoothing)
   - Check power supply stability
   - Adjust model output smoothing parameters

## License

MIT License

## Contributing

Issues and Pull Requests are welcome!

## Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) - Facial feature extraction
- [ncps](https://github.com/mlech26l/ncps) - Liquid Neural Networks
- [liquid-s4](https://github.com/raminmh/liquid-s4) - S4 layer implementation

## Contact

For questions or suggestions, please contact us through Issues.
