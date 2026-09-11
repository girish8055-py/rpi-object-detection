import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import numpy as np
import cv2
import psutil

# Import telemetry & ONNX engine
try:
    from src.gps_telemetry import TelemetryManager
    from src.onnx_engine import YOLOOnnxEngine
except ImportError:
    from gps_telemetry import TelemetryManager
    from onnx_engine import YOLOOnnxEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RPi5-PureONNX-Detection")

# Global frame buffer for optional Web Streamer
latest_encoded_frame = None
frame_lock = threading.Lock()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for background web streaming."""
    daemon_threads = True


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """Streams live detection video frames over HTTP."""
    def do_GET(self):
        global latest_encoded_frame
        if self.path == '/' or self.path == '/video':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=jpgboundary')
            self.end_headers()
            
            while True:
                with frame_lock:
                    if latest_encoded_frame is None:
                        time.sleep(0.05)
                        continue
                    buf = latest_encoded_frame
                
                try:
                    self.wfile.write(b"--jpgboundary\r\n")
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(buf)))
                    self.end_headers()
                    self.wfile.write(buf)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
                except Exception:
                    break
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def start_mjpeg_webserver(port=8080):
    """Starts background HTTP server for optional web streaming."""
    server = ThreadedHTTPServer(('0.0.0.0', port), MJPEGStreamHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server


def get_rpi_cpu_temperature():
    """Reads CPU temperature on Raspberry Pi OS / Linux."""
    try:
        thermal_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(thermal_path):
            with open(thermal_path, "r") as f:
                temp_milli_c = int(f.read().strip())
                return round(temp_milli_c / 1000.0, 1)
        
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if "cpu_thermal" in temps and temps["cpu_thermal"]:
                return round(temps["cpu_thermal"][0].current, 1)
            elif "coretemp" in temps and temps["coretemp"]:
                return round(temps["coretemp"][0].current, 1)
    except Exception:
        pass
    return None


def generate_synthetic_frame(width=640, height=480, frame_count=0):
    """Generates synthetic test frame if USB camera is unavailable."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    color_val = int(127 + 127 * np.sin(frame_count * 0.05))
    cv2.rectangle(frame, (0, 0), (width, height), (30, 30, color_val // 2), -1)
    
    cx = int(width / 2 + 150 * np.cos(frame_count * 0.05))
    cy = int(height / 2 + 100 * np.sin(frame_count * 0.05))
    
    cv2.circle(frame, (cx, cy), 40, (0, 255, 255), -1)
    cv2.putText(frame, "RPi 5 Live Camera Feed", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def build_frame_json_payload(frame_id, detections, fps, telemetry_manager, device_id="rpi5-node-01"):
    """Constructs telemetry JSON object with real-time UTC timestamp, RPi 5 CPU metrics, and FC/GPS data."""
    detection_timestamp_utc = datetime.now(timezone.utc).isoformat()
    
    cpu_temp = get_rpi_cpu_temperature()
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = psutil.virtual_memory().percent

    gps_data = telemetry_manager.get_telemetry()

    formatted_detections = []
    for idx, det in enumerate(detections):
        formatted_detections.append({
            "object_id": idx + 1,
            "class_id": det["class_id"],
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "bbox": det["bbox"]
        })

    payload = {
        "timestamp": detection_timestamp_utc,
        "device_id": device_id,
        "frame_id": frame_id,
        "telemetry": {
            "fps": round(fps, 2),
            "cpu_temp_c": cpu_temp,
            "cpu_usage_pct": cpu_usage,
            "ram_usage_pct": ram_usage,
            "fc_connected": gps_data["fc_connected"],
            "latitude": gps_data["latitude"],
            "longitude": gps_data["longitude"],
            "altitude_m": gps_data["altitude_m"]
        },
        "total_detections": len(formatted_detections),
        "detections": formatted_detections
    }

    return payload


def main():
    global latest_encoded_frame

    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Object Detection Live Display")
    parser.add_argument("--model", type=str, default="models/best.onnx", help="Path to ONNX model file")
    parser.add_argument("--classes", type=str, default="models/classes.json", help="Path to classes.json file")
    parser.add_argument("--source", type=str, default="0", help="Video source: USB camera index (0), video file path, or 'synthetic'")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--device-id", type=str, default="rpi5-camera-01", help="Device identifier string")
    parser.add_argument("--fc-port", type=str, default="", help="Flight Controller serial/MAVLink port")
    parser.add_argument("--json-output", type=str, default="output/detections.jsonl", help="File path to save JSON output")
    parser.add_argument("--save-snapshots", action="store_true", help="Save annotated snapshot images whenever objects are detected")
    parser.add_argument("--webstream", action="store_true", help="Enable optional background web stream")
    parser.add_argument("--port", type=int, default=8080, help="Port for web stream server")
    parser.add_argument("--print-json", action="store_true", help="Print JSON detection payload to console")
    parser.add_argument("--no-show", action="store_true", help="Disable desktop video window")
    args = parser.parse_args()

    # Ensure output directories exist
    if args.json_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
    
    snapshot_dir = "output/snapshots"
    if args.save_snapshots:
        os.makedirs(snapshot_dir, exist_ok=True)

    # Initialize Telemetry Manager
    telemetry_manager = TelemetryManager(connection_str=args.fc_port if args.fc_port else None)

    # Initialize Pure ONNX Engine
    engine = YOLOOnnxEngine(
        model_path=args.model,
        classes_path=args.classes,
        conf_thres=args.conf
    )

    # Optional background web stream
    if args.webstream:
        start_mjpeg_webserver(port=args.port)

    # Setup video source
    use_synthetic = (args.source.lower() == "synthetic")
    cap = None

    if not use_synthetic:
        source_val = int(args.source) if args.source.isdigit() else args.source
        logger.info(f"Opening video camera source: {source_val}")
        cap = cv2.VideoCapture(source_val)

        if not cap.isOpened():
            logger.warning(f"Unable to open camera source '{args.source}'. Falling back to synthetic test stream.")
            use_synthetic = True

    # Initialize desktop display window if GUI active
    window_name = "Raspberry Pi 5 Object Detection Feed"
    show_window = not args.no_show
    if show_window:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            logger.info("📺 Live video display window opened!")
        except Exception as e:
            logger.warning(f"Could not open desktop GUI window (running headless?): {e}")
            show_window = False

    frame_count = 0
    fps = 0.0

    logger.info("Starting detection loop. Press 'q' or Ctrl+C to exit.")
    json_file = open(args.json_output, "a") if args.json_output else None

    try:
        while True:
            loop_start = time.time()
            frame_count += 1

            if use_synthetic:
                frame = generate_synthetic_frame(640, 480, frame_count)
            else:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream or camera disconnected.")
                    break

            # Run Pure ONNX Inference
            detections, annotated_frame = engine.predict(frame, conf_threshold=args.conf)
            
            # Calculate FPS
            loop_time = time.time() - loop_start
            fps = 0.9 * fps + 0.1 * (1.0 / loop_time if loop_time > 0 else 30.0)

            # Draw status overlay
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Update webstream buffer if enabled
            if args.webstream:
                ret_jpg, jpeg_buf = cv2.imencode('.jpg', annotated_frame)
                if ret_jpg:
                    with frame_lock:
                        latest_encoded_frame = jpeg_buf.tobytes()

            # Save snapshot image if object detected
            if args.save_snapshots and len(detections) > 0 and frame_count % 10 == 0:
                snap_path = os.path.join(snapshot_dir, f"det_frame_{frame_count:06d}.jpg")
                cv2.imwrite(snap_path, annotated_frame)

            # Build telemetry payload
            payload = build_frame_json_payload(
                frame_id=frame_count,
                detections=detections,
                fps=fps,
                telemetry_manager=telemetry_manager,
                device_id=args.device_id
            )

            # Write JSON log
            json_line = json.dumps(payload)
            if json_file:
                json_file.write(json_line + "\n")
                json_file.flush()

            if args.print_json or frame_count % 30 == 0:
                logger.info(f"Frame {frame_count} | Detections: {payload['total_detections']} | FC Connected: {payload['telemetry']['fc_connected']} | FPS: {fps:.1f}")
                if args.print_json:
                    print(json.dumps(payload, indent=2))

            # Render live video frame directly on desktop screen
            if show_window:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    logger.info("Quit key pressed.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        telemetry_manager.stop()
        if cap is not None:
            cap.release()
        if json_file is not None:
            json_file.close()
        cv2.destroyAllWindows()
        logger.info(f"Cleaned up video windows and telemetry logs.")


if __name__ == "__main__":
    main()
