# -*- coding: utf-8 -*-
"""หาตำแหน่งป้ายทะเบียนด้วยโมเดล ONNX สำเร็จรูป (open-image-models)

แทน plate_detector.py (morphology + contour) ที่หาป้ายเจอไม่ครบเมื่อภาพยาก
โมเดลเป็น YOLOv9-t เทรนกับป้ายทะเบียนโดยเฉพาะ รัน ONNX บน CPU ~10-30 ms/ภาพ

ติดตั้ง:  pip install open-image-models
โมเดลโหลดอัตโนมัติครั้งแรกแล้ว cache ไว้ (~8 MB)

คืนค่ารูปแบบเดียวกับ detect_plates() เดิม: [{"bbox": (x,y,w,h), "plate_bgr": crop, "score": conf}]
"""
import os

import cv2

from . import config as C

DEFAULT_MODEL = os.environ.get("PLATE_DET_MODEL", "yolo-v9-t-384-license-plate-end2end")

_detector = None


def _get(model=DEFAULT_MODEL):
    global _detector
    if _detector is None:
        try:
            from open_image_models import LicensePlateDetector
        except ImportError as e:
            raise RuntimeError(
                "ยังไม่ได้ติดตั้ง open-image-models — รัน: pip install open-image-models") from e
        _detector = LicensePlateDetector(detection_model=model)
    return _detector


def detect_plates_onnx(bgr, debug_dir=None, model=DEFAULT_MODEL):
    det = _get(model)
    H, W = bgr.shape[:2]
    out = []
    for r in det.predict(bgr):
        bb = r.bounding_box
        x1, y1 = int(bb.x1), int(bb.y1)
        x2, y2 = int(bb.x2), int(bb.y2)
        # เผื่อขอบเล็กน้อย ช่วยให้ OCR อ่านตัวริมสุดได้ครบ
        m = int((x2 - x1) * C.CROP_MARGIN)
        x1, y1 = max(0, x1 - m), max(0, y1 - m // 2)
        x2, y2 = min(W, x2 + m), min(H, y2 + m // 2)
        if x2 - x1 < 12 or y2 - y1 < 8:
            continue
        out.append({
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "plate_bgr": bgr[y1:y2, x1:x2].copy(),
            "score": float(getattr(r, "confidence", 1.0)),
        })
    out.sort(key=lambda c: c["score"], reverse=True)
    out = out[:C.MAX_CANDIDATES]
    if debug_dir and out:
        os.makedirs(debug_dir, exist_ok=True)
        for i, c in enumerate(out):
            cv2.imwrite(os.path.join(debug_dir, f"onnx_cand{i}.png"), c["plate_bgr"])
    return out
