"""
Run this script to debug your YOLOv8 model output.
Usage: python debug_model.py your_image.jpg
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "best.pt"

def debug(image_path):
    print("\n" + "="*50)
    print("🔍 POTHOLE MODEL DEBUGGER")
    print("="*50)

    # Load model
    print(f"\n[1] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"    ✅ Model loaded")
    print(f"    📋 Classes : {model.names}")
    print(f"    📐 Task    : {model.task}")

    # Load image
    print(f"\n[2] Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("    ❌ Could not load image. Check the path.")
        return
    h, w = img.shape[:2]
    print(f"    ✅ Image loaded — size: {w}x{h}")

    # Try multiple confidence levels
    print("\n[3] Running predictions at different confidence levels...\n")
    for conf in [0.01, 0.05, 0.10, 0.15, 0.25, 0.50]:
        results = model.predict(img, conf=conf, verbose=False)
        result  = results[0]
        n_boxes = len(result.boxes) if result.boxes is not None else 0
        print(f"    conf={conf:.2f}  →  {n_boxes} detection(s)")

    # Show ALL raw boxes at conf=0.01 (lowest)
    print("\n[4] Raw detections at conf=0.01:")
    results = model.predict(img, conf=0.01, imgsz=640, verbose=False)
    result  = results[0]

    is_obb = hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0
    is_det = result.boxes is not None and len(result.boxes) > 0

    if not is_obb and not is_det:
        print("    ❌ No detections at all — even at conf=0.01")
        print("    Your model task is OBB — trying with imgsz=640 resize (already done above)")
        print("    If still 0: your model may need retraining or different test images.")
    elif is_obb:
        print(f"    ✅ OBB model — Found {len(result.obb)} detection(s):\n")
        confs = result.obb.conf.cpu().numpy()
        xyxy  = result.obb.xyxy.cpu().numpy()
        cls   = result.obb.cls.cpu().numpy()
        for i in range(len(confs)):
            x1,y1,x2,y2 = map(int, xyxy[i])
            area = (x2-x1)*(y2-y1)
            cls_name = model.names.get(int(cls[i]), "unknown")
            print(f"    #{i+1}: class={cls_name}  conf={float(confs[i]):.4f}  "
                  f"bbox=({x1},{y1},{x2},{y2})  area={area:,}px²")
        annotated = result.plot()
        cv2.imwrite("debug_output.jpg", annotated)
        print(f"\n    💾 Annotated image saved → debug_output.jpg")
    else:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()
        boxes_cls  = result.boxes.cls.cpu().numpy()
        print(f"    Found {len(boxes_xyxy)} raw detection(s):\n")
        for i in range(len(boxes_xyxy)):
            x1,y1,x2,y2 = map(int, boxes_xyxy[i])
            conf_val = float(boxes_conf[i])
            cls_id   = int(boxes_cls[i])
            cls_name = model.names.get(cls_id, "unknown")
            area     = (x2-x1)*(y2-y1)
            print(f"    #{i+1}: class={cls_name}({cls_id})  conf={conf_val:.4f}  "
                  f"bbox=({x1},{y1},{x2},{y2})  area={area:,}px²")

        # Save annotated image
        annotated = result.plot()
        out_path  = "debug_output.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"\n    💾 Annotated image saved → {out_path}")
        print("    Open debug_output.jpg to see what the model detected.")

    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_model.py <path_to_image>")
        print("Example: python debug_model.py road.jpg")
    else:
        debug(sys.argv[1])
