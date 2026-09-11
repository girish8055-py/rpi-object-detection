# Ultra-lightweight base image for Raspberry Pi 5 (ARM64 / x86_64)
FROM python:3.10-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install minimal system runtime dependencies (OpenCV, V4L2 USB camera, thermal sensors)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    v4l-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install lightweight dependencies (ONNX Runtime, OpenCV, PyMAVLink)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create output volume mount directory
RUN mkdir -p /app/output /app/models /app/src

# Copy source code and ONNX models
COPY src/ /app/src/
COPY models/ /app/models/

# Default command: Run pure ONNX engine on USB camera (/dev/video0)
ENTRYPOINT ["python", "src/main.py"]
CMD ["--model", "models/best.onnx", "--classes", "models/classes.json", "--source", "0", "--json-output", "output/detections.jsonl"]
