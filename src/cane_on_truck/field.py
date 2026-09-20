# -*- coding: utf-8 -*-
"""
cane_detector.field — โหมดภาพมุมอิสระ (layered pipeline)

Pipeline:
  L1  หาตัวรถด้วย YOLOv4-tiny — ถ้าไม่เจอ (เช่น หางพ่วงล้วน ไม่เห็นหัวรถ)
      จะสลับไปโหมดสำรอง: วิเคราะห์บริเวณกลางเฟรมด้วยเกณฑ์ที่เข้มงวดขึ้น
  L2  โซนสัมภาระ = แถบบน 40% ของ bbox รถ (หรือ pseudo-box กลางเฟรม)
  L3  หลักฐาน 5 ตัว: edge density, orientation entropy, bright/dark
      permeability, cane color
  L4  fusion — ตอบมั่นใจเฉพาะหลักฐานชัด นอกนั้น UNCERTAIN ให้คนตรวจ

annotate() จะระบายสีทองตำแหน่งที่ระบบเห็นว่าเป็นกองอ้อย (cane_mask)
เมื่อผลลัพธ์เป็น CANE เพื่อให้ตรวจสอบย้อนกลับได้ว่าระบบมองอะไรอยู่

ต้องมีไฟล์โมเดล (~24MB):
  models/yolov4-tiny.cfg
    https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
  models/yolov4-tiny.weights
    https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .core import ImageInput, _load

FIELD_LABELS_TH = {
    "CANE": "มีอ้อย",
    "EMPTY": "ไม่พบอ้อย",
    "UNCERTAIN": "ไม่แน่ใจ – ส่งตรวจ",
    "UNKNOWN": "วิเคราะห์ไม่ได้ – ส่งตรวจ",
}
_COLORS = {"CANE": (60, 170, 60), "EMPTY": (60, 60, 220),
           "UNCERTAIN": (30, 160, 235), "UNKNOWN": (120, 120, 120)}
_VEHICLE_CLS = {2, 5, 7}  # COCO: car, bus, truck


@dataclass
class FieldConfig:
    conf_th: float = 0.20
    roi_top_frac: float = 0.40
    roi_x_inset: float = 0.04
    std_width: int = 480
    # เกณฑ์ fusion ปกติ (เมื่อพบรถ)
    edge_empty: float = 0.09
    ent_empty: float = 0.85
    bright_empty: float = 0.25
    edge_cane: float = 0.18
    ent_cane: float = 0.90
    perm_bright: float = 0.30
    perm_dark: float = 0.30
    color_cane: float = 0.50
    ent_color: float = 0.88
    # โหมดสำรอง (ไม่พบรถ): pseudo-box กลางเฟรม + เกณฑ์เข้มขึ้น
    fb_box: Tuple[float, float, float, float] = (0.04, 0.10, 0.92, 0.78)  # x,y,w,h สัดส่วนเฟรม
    fb_edge_cane: float = 0.20
    fb_ent_cane: float = 0.92
    fb_perm: float = 0.35
    # cane highlight (annotate)
    hl_win: int = 31           # หน้าต่างวัดความหนาแน่นขอบ (px)
    hl_density: float = 0.12   # เกณฑ์ความหนาแน่นขอบขั้นต่ำ

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FieldResult:
    label: str                    # CANE | EMPTY | UNCERTAIN | UNKNOWN
    label_th: str
    cues: Optional[dict]
    bbox: Optional[Tuple[int, int, int, int]]   # bbox รถ (None เมื่อใช้โหมดสำรอง)
    source: str = "vehicle"       # vehicle | frame_fallback | none

    @property
    def has_cane(self) -> bool:
        return self.label == "CANE"

    @property
    def needs_review(self) -> bool:
        return self.label in ("UNCERTAIN", "UNKNOWN")

    def to_dict(self) -> dict:
        return {"label": self.label, "label_th": self.label_th, "cues": self.cues,
                "bbox": list(self.bbox) if self.bbox else None, "source": self.source}


class FieldCaneDetector:
    def __init__(self, model_dir: str = "models", config: Optional[FieldConfig] = None):
        self.cfg = config or FieldConfig()
        d = Path(model_dir)
        cfg_p, w_p = d / "yolov4-tiny.cfg", d / "yolov4-tiny.weights"
        if not cfg_p.exists() or not w_p.exists():
            raise FileNotFoundError(
                f"ไม่พบไฟล์โมเดลใน {d.resolve()} — ดู docstring ของ cane_detector.field")
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_p), str(w_p))
        self.ln = self.net.getUnconnectedOutLayersNames()

    # ------------------------------------------------------------ L1: หารถ
    def find_vehicle(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        boxes, confs = [], []
        for out in self.net.forward(self.ln):
            for d in out:
                cid = int(np.argmax(d[5:]))
                conf = float(d[5 + cid])
                if cid in _VEHICLE_CLS and conf > self.cfg.conf_th:
                    cx, cy, bw, bh = d[0] * w, d[1] * h, d[2] * w, d[3] * h
                    boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                    confs.append(conf)
        if not boxes:
            return None
        idxs = np.array(cv2.dnn.NMSBoxes(boxes, confs, self.cfg.conf_th, 0.4)).flatten()
        x, y, bw, bh = max((boxes[i] for i in idxs), key=lambda b: b[2] * b[3])
        x, y = max(0, x), max(0, y)
        return x, y, min(bw, w - x), min(bh, h - y)

    def _pseudo_box(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = img.shape[:2]
        fx, fy, fw, fh = self.cfg.fb_box
        return int(fx * w), int(fy * h), int(fw * w), int(fh * h)

    # ---------------------------------------------------- L2+L3: วัดหลักฐาน
    def _roi_rect(self, box) -> Tuple[int, int, int, int]:
        c = self.cfg
        x, y, bw, bh = box
        return (x + int(c.roi_x_inset * bw), y,
                x + int((1 - c.roi_x_inset) * bw), y + int(c.roi_top_frac * bh))

    def load_cues(self, img: np.ndarray, box) -> Optional[dict]:
        c = self.cfg
        x0, y0, x1, y1 = self._roi_rect(box)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        roi = cv2.resize(roi, (c.std_width, max(24, int(roi.shape[0] * c.std_width / roi.shape[1]))))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        ang = np.mod(np.arctan2(gy, gx), np.pi)[edges > 0]
        if ang.size < 50:
            ent = 0.0
        else:
            p = np.histogram(ang, bins=12, range=(0, np.pi))[0].astype(float)
            p = p / p.sum()
            p = p[p > 0]
            ent = float(-(p * np.log(p)).sum() / np.log(12))
        H, S, V = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
        return dict(edge=float((edges > 0).mean()), ent=ent,
                    bright=float((V > 170).mean()), dark=float((V < 80).mean()),
                    color=float(((H >= 8) & (H <= 40) & (S > 25) & (V > 50)).mean()))

    # ------------------------------------------------------------ L4: fusion
    def _classify(self, cu: Optional[dict]) -> str:
        c = self.cfg
        if cu is None:
            return "UNKNOWN"
        if cu["edge"] < c.edge_empty:
            return "EMPTY"
        if cu["ent"] < c.ent_empty and cu["bright"] >= c.bright_empty:
            return "EMPTY"
        if cu["edge"] >= c.edge_cane and cu["ent"] >= c.ent_cane:
            return "CANE"
        if cu["bright"] >= c.perm_bright and cu["dark"] >= c.perm_dark and cu["ent"] >= c.ent_cane:
            return "CANE"
        if cu["color"] >= c.color_cane and cu["ent"] >= c.ent_color:
            return "CANE"
        return "UNCERTAIN"

    def _classify_fallback(self, cu: Optional[dict]) -> str:
        """โหมดสำรอง — เข้มงวดขึ้น กันการเดาผิดบนภาพที่ไม่มีรถจริง ๆ"""
        c = self.cfg
        if cu is None:
            return "UNKNOWN"
        if cu["edge"] < c.edge_empty:
            return "EMPTY"                       # ไม่มีอะไรกองอยู่เลย
        if cu["ent"] < 0.80 and cu["bright"] >= 0.30:
            return "EMPTY"                       # โครงสร้างระเบียบ + สว่างลอด = คอกเปล่า
        if cu["edge"] >= c.fb_edge_cane and cu["ent"] >= c.fb_ent_cane:
            return "CANE"
        if cu["bright"] >= c.fb_perm and cu["dark"] >= c.fb_perm and cu["ent"] >= c.fb_ent_cane:
            return "CANE"
        return "UNCERTAIN"

    # ----------------------------------------------------------------- API
    def detect(self, image: ImageInput) -> FieldResult:
        img = _load(image)
        box = self.find_vehicle(img)
        if box is not None:
            cues = self.load_cues(img, box)
            label = self._classify(cues)
            source = "vehicle"
        else:
            box = None
            pbox = self._pseudo_box(img)
            cues = self.load_cues(img, pbox)
            label = self._classify_fallback(cues)
            source = "frame_fallback"
        if cues:
            cues = {k: round(v, 3) for k, v in cues.items()}
        return FieldResult(label, FIELD_LABELS_TH[label], cues, box, source)

    # -------------------------------------------- cane highlight (ตำแหน่งอ้อย)
    def cane_mask(self, img: np.ndarray, box) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """คืน mask (0/255) ของบริเวณที่ texture หนาแน่นในโซนสัมภาระ + offset (x,y)
        ใช้ระบุ 'ตำแหน่งกองอ้อย' บนภาพต้นฉบับ"""
        c = self.cfg
        x0, y0, x1, y1 = self._roi_rect(box)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            return None, (x0, y0)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        e = (cv2.Canny(gray, 60, 150) > 0).astype(np.float32)
        dens = cv2.boxFilter(e, -1, (c.hl_win, c.hl_win))
        m = (dens > c.hl_density).astype(np.uint8) * 255
        k = np.ones((11, 11), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        return m, (x0, y0)

    def annotate(self, image: ImageInput, result: Optional[FieldResult] = None) -> np.ndarray:
        img = _load(image).copy()
        res = result or self.detect(img)
        color = _COLORS[res.label]
        box = res.bbox if res.bbox else self._pseudo_box(img)

        # ระบายสีทองตำแหน่งอ้อยเมื่อพบอ้อย
        if res.label == "CANE":
            m, (ox, oy) = self.cane_mask(img, box)
            if m is not None:
                sub = img[oy:oy + m.shape[0], ox:ox + m.shape[1]]
                sel = m > 0
                gold = np.array([40, 200, 255], dtype=np.float32)
                sub[sel] = (0.45 * sub[sel] + 0.55 * gold).astype(np.uint8)

        # กรอบรถ: เส้นทึบเมื่อพบรถ / เส้นประเมื่อใช้โหมดสำรอง
        x, y, bw, bh = box
        if res.source == "vehicle":
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 220, 0), 3)
        else:
            for xx in range(x, x + bw, 24):
                cv2.line(img, (xx, y), (min(xx + 12, x + bw), y), (255, 220, 0), 3)
                cv2.line(img, (xx, y + bh), (min(xx + 12, x + bw), y + bh), (255, 220, 0), 3)
            for yy in range(y, y + bh, 24):
                cv2.line(img, (x, yy), (x, min(yy + 12, y + bh)), (255, 220, 0), 3)
                cv2.line(img, (x + bw, yy), (x + bw, min(yy + 12, y + bh)), (255, 220, 0), 3)

        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color,
                      max(6, img.shape[1] // 200))
        txt = res.label if not res.cues else \
            f"{res.label}  edge={res.cues['edge']} ent={res.cues['ent']}"
        cv2.putText(img, txt, (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        return img
