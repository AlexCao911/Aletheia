# Robot Face Expression Control

This is a web-based robot face expression control interface using MediaPipe for real-time facial tracking.

## Features

- 21-DOF (Degrees of Freedom) robot face control
- Real-time facial expression tracking using MediaPipe
- Manual control sliders for all facial features
- Expression presets library
- 3D visualization using Three.js and React Three Fiber

## Run Locally

**Prerequisites:** Node.js (v20.19+ or v22.12+)

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:3000`

## Usage

- **Vision Mode**: Enable camera to mirror your facial expressions in real-time
- **Presets**: Click preset buttons to apply predefined expressions
- **Manual Control**: Use sliders to fine-tune individual facial features
- **Gestures**: Quick action buttons for micro-expressions
