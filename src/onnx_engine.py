import os
import json
import logging
import numpy as np
import cv2
import onnxruntime as ort

logger = logging.getLogger("ONNX-Engine")


class YOLOOnnxEngine:
    """Lightweight zero-PyTorch YOLO ONNX Inference Engine for Raspberry Pi 5.
    
    Uses pure onnxruntime + OpenCV. Extremely fast, low RAM footprint, zero thermal throttling.
    """

    def __init__(self, model_path="models/best.onnx", classes_path="models/classes.json", conf_thres=0.4, iou_thres=0.45):
        if not os.path.exists(model_path):
            alt_onnx = model_path.rsplit(".", 1)[0] + ".onnx"
            if os.path.exists(alt_onnx):
                model_path = alt_onnx
            else:
                raise FileNotFoundError(f"ONNX model file not found at '{model_path}'. Ensure best.onnx is present in models/.")

        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Initialize ONNX Runtime Session (optimized CPU execution provider for ARM/x86)
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4  # RPi 5 quad-core ARM Cortex-A76
        
        logger.info(f"Loading ONNX Runtime Session for: '{self.model_path}'...")
        self.session = ort.InferenceSession(self.model_path, opts, providers=['CPUExecutionProvider'])

        # Input & Output tensor info
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]
        self.img_height = input_shape[2] if len(input_shape) > 2 and isinstance(input_shape[2], int) else 640
        self.img_width = input_shape[3] if len(input_shape) > 3 and isinstance(input_shape[3], int) else 640

        # Load class names dictionary
        self.names = self._load_classes(classes_path)
        logger.info(f"ONNX Engine initialized successfully. Loaded {len(self.names)} classes: {self.names}")

    def _load_classes(self, classes_path):
        if os.path.exists(classes_path):
            try:
                with open(classes_path, "r") as f:
                    raw = json.load(f)
                    return {int(k): v for k, v in raw.items()}
            except Exception as e:
                logger.warning(f"Unable to parse classes.json: {e}")
        return {0: "drone", 1: "person", 2: "vehicle"}

    def _preprocess(self, frame):
        """Resizes frame with letterboxing, normalizes to [0, 1], and formats to (1, 3, H, W)."""
        h_orig, w_orig = frame.shape[:2]
        
        # Scale factor
        r = min(self.img_height / h_orig, self.img_width / w_orig)
        nw, nh = int(round(w_orig * r)), int(round(h_orig * r))
        
        dw = (self.img_width - nw) / 2
        dh = (self.img_height - nh) / 2

        if (w_orig, h_orig) != (nw, nh):
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame.copy()

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Convert BGR to RGB, scale to 0.0 - 1.0
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)  # Shape: (1, 3, 640, 640)
        
        return blob, r, (dw, dh)

    def predict(self, frame, conf_threshold=None):
        """Runs ONNX inference on an input frame and returns parsed detections."""
        conf_thres = conf_threshold if conf_threshold is not None else self.conf_thres
        h_orig, w_orig = frame.shape[:2]

        blob, r, (dw, dh) = self._preprocess(frame)

        # Execute ONNX Session
        outputs = self.session.run(None, {self.input_name: blob})
        output = outputs[0]  # Shape e.g. (1, 7, 8400) or (1, 84, 8400)

        if len(output.shape) == 3:
            output = output[0]  # Shape (N_attrs, N_anchors) e.g. (7, 8400)

        # Transpose if output is (channels, anchors) -> (anchors, channels)
        if output.shape[0] < output.shape[1]:
            output = output.T  # Shape (8400, 7)

        # Bounding box centers, widths, heights & class probabilities
        boxes_cxcywh = output[:, :4]
        scores = output[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence threshold
        mask = confidences >= conf_thres
        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(confidences) == 0:
            return [], frame

        # Convert cx, cy, w, h to top-left x, y, width, height for NMS
        boxes_xywh = []
        for cx, cy, w, h in boxes_cxcywh:
            x = cx - w / 2
            y = cy - h / 2
            boxes_xywh.append([x, y, w, h])

        # Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences.tolist(), conf_thres, self.iou_thres)

        detections = []
        annotated_frame = frame.copy()

        if len(indices) > 0:
            indices = indices.flatten()
            for idx in indices:
                box_xywh = boxes_xywh[idx]
                conf = float(confidences[idx])
                cls_id = int(class_ids[idx])
                cls_name = self.names.get(cls_id, f"class_{cls_id}")

                # Scale back to original frame coordinates
                x_box, y_box, w_box, h_box = box_xywh
                xmin = max(0, int((x_box - dw) / r))
                ymin = max(0, int((y_box - dh) / r))
                xmax = min(w_orig, int((x_box + w_box - dw) / r))
                ymax = min(h_orig, int((y_box + h_box - dh) / r))

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
                })

                # Draw bounding box on frame
                cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated_frame, label, (xmin, max(20, ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return detections, annotated_frame
