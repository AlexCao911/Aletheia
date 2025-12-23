# Implementation Plan

## Phase 1: Project Setup and Core Infrastructure

- [x] 1. Set up project structure and dependencies
  - [x] 1.1 Create Python package structure for expression_control module
    - Create directories: `expression_control/`, `expression_control/models/`, `expression_control/utils/`, `tests/`
    - Create `__init__.py` files and `setup.py`/`pyproject.toml`
    - _Requirements: 1.1, 1.7_
  - [x] 1.2 Set up dependencies and requirements
    - Add dependencies: `mediapipe`, `ncps`, `torch`, `onnxruntime`, `pyserial`, `opencv-python`, `hypothesis`
    - Create `requirements.txt` and `requirements-dev.txt`
    - _Requirements: 4.1, 4.6_

## Phase 2: Communication Protocol

- [x] 2. Implement servo command protocol
  - [x] 2.1 Implement ServoCommandProtocol encoder and decoder
    - Create `expression_control/protocol.py`
    - Implement `encode()` method for angles dict to command string
    - Implement `decode()` method for command string to angles dict
    - Implement validation for angle range [0, 180]
    - _Requirements: 2.3, 2.6, 2.7, 2.8_
  - [x] 2.2 Write property test for protocol round-trip
    - **Property 1: Protocol Round-Trip Consistency**
    - **Validates: Requirements 2.3, 2.7, 2.8**
  - [x] 2.3 Write property test for angle range validation
    - **Property 2: Angle Range Validation**
    - **Validates: Requirements 2.6**
  - [x] 2.4 Implement SerialManager for USB communication
    - Create `expression_control/serial_manager.py`
    - Implement connection management, send_angles(), send_command()
    - _Requirements: 2.4, 2.5_

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 3: MouthMaster Pico Firmware Update

- [x] 4. Update MouthMaster Pico firmware for batch angle commands
  - [x] 4.1 Add angle command parser to MouthMaster.py
    - Parse "angles:A1,A2,...,A21" format
    - Extract 21 angle values and distribute to servos
    - _Requirements: 2.3, 2.4_
  - [x] 4.2 Implement GPIO signal translation for Eyes and Brows
    - Map angle ranges to GPIO signal patterns (SDA/SCL, BSDA/BSCL)
    - Maintain backward compatibility with existing commands
    - _Requirements: 1.3, 1.5_
  - [x] 4.3 Add angle readback capability for data collection
    - Implement command to report current servo angles
    - _Requirements: 3.1_

## Phase 4: MediaPipe Feature Extraction

- [x] 5. Implement MediaPipe face feature extractor
  - [x] 5.1 Create FaceFeatures dataclass
    - Create `expression_control/features.py`
    - Implement all 14 feature fields with proper types
    - Implement `to_array()` and `from_array()` methods
    - _Requirements: 4.2_
  - [x] 5.2 Write property test for feature extraction completeness
    - **Property 4: Feature Extraction Completeness**
    - **Validates: Requirements 4.2**
  - [x] 5.3 Implement FaceFeatureExtractor class
    - Create `expression_control/extractor.py`
    - Initialize MediaPipe Face Mesh with 478 landmarks
    - Extract eye aspect ratios, mouth features, eyebrow positions, head pose
    - Normalize coordinates relative to face bounding box
    - _Requirements: 4.1, 4.2, 4.10_
  - [x] 5.4 Write property test for feature normalization
    - **Property 8: Feature Normalization Invariance**
    - **Validates: Requirements 4.10**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 5: Data Collection System

- [x] 7. Implement data collection and serialization
  - [x] 7.1 Create TrainingDataSample dataclass with JSON serialization
    - Create `expression_control/data.py`
    - Implement `to_dict()`, `from_dict()`, `to_json()`, `from_json()` methods
    - Define JSON schema for dataset format
    - _Requirements: 3.2, 3.7_
  - [x] 7.2 Write property test for training data round-trip
    - **Property 3: Training Data Round-Trip Consistency**
    - **Validates: Requirements 3.6, 3.7**
  - [x] 7.3 Implement DataCollector class
    - Create `expression_control/collector.py`
    - Implement `record_session()` with synchronized capture
    - Implement `save_dataset()` with JSON export
    - Support expression label annotation
    - _Requirements: 3.1, 3.4, 3.8_
  - [x] 7.4 Implement ExpressionDataset for PyTorch
    - Create `expression_control/dataset.py`
    - Implement sequence creation from samples
    - Implement data augmentation (temporal jittering, noise)
    - _Requirements: 3.5, 5.1, 5.2_

## Phase 6: LNN-S4 Model Implementation

- [x] 8. Implement Liquid-S4 model architecture
  - [x] 8.1 Implement S4Layer class
    - Create `expression_control/models/s4.py`
    - Implement structured state space layer with HiPPO initialization
    - Implement forward pass with state management
    - _Requirements: 4.6_
  - [x] 8.2 Implement LiquidS4Model class
    - Create `expression_control/models/liquid_s4.py`
    - Integrate S4 layers with ncps LTC layer
    - Implement embedding, S4 blocks, output head
    - _Requirements: 4.3, 4.4, 4.6_
  - [x] 8.3 Write property test for model output validity
    - **Property 5: Model Output Validity**
    - **Validates: Requirements 4.4**
  - [x] 8.4 Implement LNNS4Config dataclass
    - Define all model hyperparameters
    - _Requirements: 4.7_

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 7: Training Pipeline

- [x] 10. Implement training infrastructure
  - [x] 10.1 Implement Trainer class
    - Create `expression_control/trainer.py`
    - Implement train_epoch(), validate(), train() methods
    - Implement checkpoint saving and loading
    - Implement early stopping
    - _Requirements: 5.3, 5.4, 5.6, 5.8_
  - [x] 10.2 Write property test for checkpoint round-trip
    - **Property 9: Checkpoint Round-Trip Consistency**
    - **Validates: Requirements 5.4**
  - [x] 10.3 Write property test for metric computation
    - **Property 10: Metric Computation Correctness**
    - **Validates: Requirements 5.6**
  - [x] 10.4 Implement ONNX export functionality
    - Add export_onnx() method to Trainer
    - Verify exported model produces same outputs
    - _Requirements: 5.5_
  - [x] 10.5 Create training CLI scripts
    - Create `expression_control/cli/train.py`
    - Create `expression_control/cli/export.py`
    - Create `expression_control/cli/evaluate.py`
    - _Requirements: 5.1, 5.5_

## Phase 8: Temporal Smoothing

- [-] 11. Implement temporal smoothing
  - [x] 11.1 Implement TemporalSmoother class with EMA
    - Create `expression_control/smoother.py`
    - Implement exponential moving average smoothing
    - Implement reset() for new sequences
    - _Requirements: 6.2_
  - [x] 11.2 Write property test for EMA smoothing bounds
    - **Property 11: EMA Smoothing Bounds**
    - **Validates: Requirements 6.2**

## Phase 9: Inference System

- [x] 12. Implement real-time inference engine
  - [x] 12.1 Implement LNNS4Inference class for ONNX runtime
    - Create `expression_control/inference.py`
    - Load ONNX model with onnxruntime
    - Implement predict() with state management
    - _Requirements: 4.3, 4.4_
  - [x] 12.2 Implement InferenceEngine with full pipeline
    - Integrate camera, MediaPipe, model, smoother, serial
    - Implement face detection timeout (500ms → neutral)
    - Implement fallback mode (direct MediaPipe mapping)
    - _Requirements: 6.1, 6.3, 6.8_
  - [x] 12.3 Write property test for face detection fallback
    - **Property 6: Face Detection Fallback**
    - **Validates: Requirements 4.8**
  - [ ]* 12.4 Write property test for fallback mode validity
    - **Property 12: Fallback Mode Validity**
    - **Validates: Requirements 6.8**
  - [x] 12.5 Implement configuration interface
    - Create config file for sensitivity, smoothing parameters
    - Support runtime model switching
    - _Requirements: 6.4, 6.7_
  - [x] 12.6 Create inference CLI script
    - Create `expression_control/cli/run.py`
    - Implement main loop with performance logging
    - _Requirements: 6.5, 6.6_

- [x] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 10: Integration and Testing

- [ ] 14. Integration testing
  - [ ]* 14.1 Write integration tests for RPi5-Pico communication
    - Test command sending and response
    - Test sustained communication at 30 Hz
    - _Requirements: 7.2_
  - [ ]* 14.2 Write property test for temporal coherence
    - **Property 7: Temporal Coherence**
    - **Validates: Requirements 4.9**
  - [ ]* 14.3 Write property test for angle tolerance
    - **Property 13: Angle Tolerance Verification**
    - **Validates: Requirements 7.4**
  - [ ]* 14.4 Write performance benchmarks
    - Benchmark MediaPipe extraction latency
    - Benchmark model inference latency
    - Benchmark end-to-end pipeline latency
    - _Requirements: 7.3_

- [ ] 15. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

