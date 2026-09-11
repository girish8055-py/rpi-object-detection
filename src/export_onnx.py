import argparse
import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Export-ONNX")

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics package missing. Run: pip install -r requirements.txt")
    sys.exit(1)


def inspect_and_export(model_path, output_dir=None, imgsz=640, half=False):
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: '{model_path}'")
        sys.exit(1)

    logger.info(f"Loading PyTorch model: '{model_path}'")
    model = YOLO(model_path)

    # Extract class dictionary
    class_names = model.names
    logger.info(f"Found {len(class_names)} classes in model:")
    print("=" * 50)
    print(json.dumps(class_names, indent=2))
    print("=" * 50)

    # Save class names mapping json
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(model_path))

    classes_json_path = os.path.join(output_dir, "classes.json")
    with open(classes_json_path, "w") as f:
        json.dump(class_names, f, indent=2)
    logger.info(f"Saved class mapping to '{classes_json_path}'")

    # Export to ONNX format
    logger.info(f"Exporting model to ONNX format (imgsz={imgsz}, half={half})...")
    onnx_file_path = model.export(format="onnx", imgsz=imgsz, half=half, simplify=True)
    logger.info(f"Successfully exported ONNX model to: '{onnx_file_path}'")

    # Target path inside models directory
    target_onnx_path = os.path.join(output_dir, "best.onnx")
    if os.path.abspath(onnx_file_path) != os.path.abspath(target_onnx_path):
        if os.path.exists(target_onnx_path):
            os.remove(target_onnx_path)
        os.rename(onnx_file_path, target_onnx_path)
        logger.info(f"Renamed/saved ONNX model to: '{target_onnx_path}'")

    return target_onnx_path, class_names


def main():
    parser = argparse.ArgumentParser(description="Inspect PyTorch model classes and export to ONNX for RPi 5")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to input PyTorch model (.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--half", action="store_true", help="Export with FP16 half-precision")
    args = parser.parse_args()

    inspect_and_export(args.model, imgsz=args.imgsz, half=args.half)


if __name__ == "__main__":
    main()
