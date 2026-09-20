# -*- coding: utf-8 -*-
"""ตัวอ่านป้ายด้วยโมเดล ONNX เฉพาะทาง (fast-plate-ocr)

ต่างจาก reader_template.py ตรงที่ตัวนี้เป็นโมเดลที่เทรนกับ "ป้ายทะเบียน" โดยตรง
(CCT / mobile-ViT) แต่รันเป็น ONNX บน CPU ระดับ ~5-20 ms ต่อป้าย — เร็วกว่า EasyOCR
หลายสิบเท่าและแม่นกว่า template matching มาก เพราะทนฟอนต์/มุม/เบลอได้จริง

ติดตั้ง:  pip install fast-plate-ocr
โมเดลโหลดอัตโนมัติครั้งแรกแล้ว cache ไว้ (~5-25 MB แล้วแต่รุ่น)

รุ่นที่เลือกได้ (จากเร็วสุด -> แม่นสุด):
    cct-xs-v2-global-model      เล็ก/เร็ว
    cct-s-v2-global-model       ค่าเริ่มต้นที่ใช้ที่นี่ — สมดุลดีสุด
    global-plates-mobile-vit-v2-model

หมายเหตุ: อ่านได้เฉพาะตัวเลข/อักษรละติน (ป้ายรถบรรทุกไทย NN-NNNN เป็นตัวเลขล้วน
จึงตรงงาน) — ไม่อ่านชื่อจังหวัดภาษาไทย ถ้าต้องการจังหวัดให้ใช้ engine easyocr
"""
import os
import re

import numpy as np

DEFAULT_MODEL = os.environ.get("FAST_PLATE_MODEL", "cct-s-v2-global-model")
# ป้ายรถบรรทุกไทยเป็นตัวเลขล้วน — ตัดอักษรที่โมเดลอาจเดาปนมาทิ้ง
_KEEP = re.compile(r"[^0-9]")
# อักษรละตินที่หน้าตาเหมือนตัวเลข (โมเดล global เจอป้ายหลายประเทศจึงมีโอกาสสลับ)
_LOOKALIKE = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                            "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"})


class FastPlateReader:
    """อ่านป้ายด้วย fast-plate-ocr (ONNX) — คืนผลรูปแบบเดียวกับ PlateReader ตัวอื่น"""

    def __init__(self, model=DEFAULT_MODEL, digits_only=True):
        try:
            from fast_plate_ocr import LicensePlateRecognizer
        except ImportError as e:
            raise RuntimeError(
                "ยังไม่ได้ติดตั้ง fast-plate-ocr — รัน: pip install fast-plate-ocr") from e
        self.model_name = model
        self.digits_only = digits_only
        self._m = LicensePlateRecognizer(hub_ocr_model=model, device="auto")

    def _clean(self, text):
        """แปลงตัวอักษรที่หน้าตาเหมือนเลข แล้วเหลือเฉพาะตัวเลข"""
        t = (text or "").upper().strip()
        if not self.digits_only:
            return t
        return _KEEP.sub("", t.translate(_LOOKALIKE))

    def read(self, plate_bgr):
        """@param plate_bgr ภาพป้ายที่ crop มา (BGR 3 ช่อง — โมเดล resize/normalize เอง)
        @returns list [(ข้อความ, ความมั่นใจ 0-1)]"""
        img = plate_bgr
        if img is None or img.size == 0:
            return []
        if img.ndim == 2:                       # เผื่อได้ grayscale มา
            img = np.repeat(img[:, :, None], 3, axis=2)
        try:
            preds = self._m.run(img, return_confidence=True)
        except Exception:
            return []
        if not preds:
            return []
        p = preds[0]

        text = getattr(p, "plate", "") or ""
        probs = getattr(p, "char_probs", None)
        if probs is None:
            conf = 0.5
        else:
            probs = np.asarray(probs, dtype=np.float32).reshape(-1)
            # ใช้ค่าต่ำสุดรายตัวอักษร — จุดที่อ่อนที่สุดคือจุดที่มักอ่านผิด
            conf = float(probs.min()) if probs.size else 0.5

        cleaned = self._clean(text)
        if len(cleaned) < 3:
            return []
        return [(cleaned, round(max(0.0, min(1.0, conf)), 3))]
