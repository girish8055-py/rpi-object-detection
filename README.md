# Raspberry Pi 5 Object Detection & Flight Controller Telemetry System

Ultra-lightweight real-time object detection system for **Raspberry Pi 5** powered by **Pure ONNX Runtime (`onnxruntime`)** and **OpenCV**. Features auto-discovering Flight Controller (MAVLink / GPS) telemetry, live HTTP web video streaming, and real-time JSON detection logging.

---

## Features

- ⚡ **Zero-PyTorch Overhead**: Uses pure `onnxruntime` + OpenCV NMS for high-speed inference without PyTorch/Ultralytics memory overhead or thermal shutdowns.
- 🚁 **Flight Controller Auto-Connect**: Seamlessly connects to Pixhawk / ArduPilot / PX4 over USB (`/dev/ttyACM0`) or GPIO UART (`/dev/ttyAMA0`). Defaults to `null` GPS when disconnected without crashing.
- 🌐 **Live Web Stream**: Stream live video with bounding boxes to any browser at `http://<RPi_IP>:8080`.
- 📊 **Rich JSON Telemetry**: Outputs frame-by-frame JSON logs containing real UTC ISO timestamps, bounding box coordinates, class IDs/names, confidence scores, and RPi 5 CPU temperature metrics.
- 🎯 **3 Target Classes**: Detects `drone`, `person`, and `vehicle`.

---

## Quick Start on Raspberry Pi 5 (Git Clone Setup)

### 1. Clone & Setup Environment

```bash
git clone <YOUR_GIT_REPO_URL> rpi-object-detection
cd rpi-object-detection

# Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate

# Install lightweight requirements (< 15 seconds)
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Detection

#### Option A: Live Camera Stream with Web Browser Viewing
```bash
python src/main.py --source 0
```
Open your browser at: `http://<RPi5_IP>:8080`

#### Option B: Headless Terminal Execution with JSON Console Output
```bash
python src/main.py --source 0 --no-show --print-json
```

---

## Docker Deployment on RPi 5

```bash
# Build lightweight Docker image
docker build -t rpi5-onnx-detection .

# Run with USB camera and Flight Controller
docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  --device=/dev/ttyACM0:/dev/ttyACM0 \
  -v /sys/class/thermal:/sys/class/thermal:ro \
  -v $(pwd)/output:/app/output \
  rpi5-onnx-detection --model models/best.onnx --source 0 --fc-port /dev/ttyACM0
```
