import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.common import CACHE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--name", default="ball")
    a = ap.parse_args()
    from ultralytics import YOLO
    os.makedirs(os.path.join(CACHE, "weights"), exist_ok=True)
    os.chdir(os.path.join(CACHE, "weights"))
    m = YOLO(a.model)
    m.train(data=os.path.join(CACHE, "ds", "data.yaml"), epochs=a.epochs, imgsz=a.imgsz,
            batch=a.batch, workers=16, project=os.path.join(CACHE, "runs"), name=a.name,
            exist_ok=True, seed=0, deterministic=False, cos_lr=True, close_mosaic=4,
            scale=0.25, translate=0.1, degrees=0.0, mosaic=1.0, mixup=0.0, fliplr=0.5,
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, patience=100, plots=False, amp=True)


if __name__ == "__main__":
    main()
