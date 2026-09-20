# -*- coding: utf-8 -*-
"""ตัวหาตำแหน่งป้ายด้วย YOLO (ทางเลือกเมื่อ OpenCV ล้วนเอาไม่อยู่)

ใช้เมื่อไหร่:
  - ฉากหลังรก/แสงยากมากจน detector แบบ OpenCV จับป้ายพลาดบ่อย
  - มีสติกเกอร์/ตัวหนังสืออื่นบนรถเยอะจน candidate ปลอมรบกวน
  หมายเหตุ: YOLO ช่วยเรื่อง "หาป้ายเจอ" เท่านั้น — ความแม่นของ "การอ่านเลข"
  ยังขึ้นกับ OCR และความละเอียดของป้ายในภาพเหมือนเดิม

การติดตั้ง:
    pip install ultralytics

โมเดลที่ใช้ (เลือกอย่างใดอย่างหนึ่ง):
  1) โมเดลตรวจป้ายทะเบียนสำเร็จรูป (แนะนำ) — หาได้จาก Hugging Face /
     Roboflow Universe ค้นคำว่า "license plate yolov8" แล้วเซฟเป็น .pt
     เช่น keremberke/yolov8n-license-plate
  2) เทรนเองจากภาพหน้างานจริง ~300-500 ภาพ (annotate กรอบป้าย)
     yolo detect train data=plate.yaml model=yolov8n.pt epochs=60 imgsz=640
     ได้ runs/detect/train/weights/best.pt มาใช้

วิธีใช้:
    python main.py --image truck.jpg --detector yolo --yolo-model best.pt
"""
import cv2

from . import config as C

_model_cache = {}


def _load_model(model_path):
    if model_path in _model_cache:
        return _model_cache[model_path]
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "ยังไม่ได้ติดตั้ง ultralytics — รัน: pip install ultralytics") from e
    model = YOLO(model_path)
    _model_cache[model_path] = model
    return model


def detect_plates_yolo(bgr, model_path, conf=0.25, debug_dir=None):
    """หาตำแหน่งป้ายด้วยโมเดล YOLO — คืนรูปแบบเดียวกับ detect_plates()

    Returns
    -------
    list ของ dict: {bbox, plate_bgr, score}
        score = confidence ของ YOLO คูณ 10 (ให้เทียบเคียงสเกล detect_score เดิม
        และชนะ candidate จาก OpenCV เสมอเมื่อใช้ร่วมกัน)
    """
    model = _load_model(model_path)
    H, W = bgr.shape[:2]
    results = model.predict(bgr, conf=conf, verbose=False)

    out = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            score = float(box.conf[0])
            # เผื่อขอบเหมือน detector หลัก
            w, h = x2 - x1, y2 - y1
            mx, my = int(w * C.CROP_MARGIN), int(h * C.CROP_MARGIN)
            X, Y = max(0, x1 - mx), max(0, y1 - my)
            X2, Y2 = min(W, x2 + mx), min(H, y2 + my)
            if X2 - X < 10 or Y2 - Y < 5:
                continue
            out.append({
                "bbox": (X, Y, X2 - X, Y2 - Y),
                "plate_bgr": bgr[Y:Y2, X:X2].copy(),
                "score": round(score * 10, 2),
            })
    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:C.MAX_CANDIDATES]
