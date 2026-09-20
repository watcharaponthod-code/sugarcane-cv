# -*- coding: utf-8 -*-
"""
cane_detector.core — ตัวจำแนกรถบรรทุกอ้อยจากภาพด้านข้าง (rule-based)

Public API:
    CaneDetector(config=None).detect(image) -> DetectionResult
    detect(image)                           -> DetectionResult (default config)

`image` รับได้ 3 แบบ: path (str/Path), bytes (จาก upload/HTTP), หรือ numpy array (BGR)

เกณฑ์คาลิเบรตจากชุดทดสอบ 19 คัน — ใช้กล้องจริงควรจูนใหม่ผ่าน DetectorConfig
(บันทึก/โหลดเป็น JSON ต่อ site ได้: cfg.to_json("sithep.json"))
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]

LABELS_TH = {
    "CANE": "มีอ้อย",
    "COVERED": "คลุมผ้าใบ – ตรวจสอบ",
    "EMPTY": "ไม่พบอ้อย",
}
_COLORS = {"CANE": (60, 170, 60), "COVERED": (30, 160, 235), "EMPTY": (60, 60, 220)}


@dataclass
class DetectorConfig:
    """Threshold ทั้งหมดของ detector — ปรับต่อ site/กล้องได้"""
    # ท้องฟ้า (HSV)
    sky_h_lo: int = 85
    sky_h_hi: int = 135
    sky_s_lo: int = 20
    sky_v_lo: int = 100
    # โทนสีอ้อยแห้ง (HSV)
    cane_h_lo: int = 8
    cane_h_hi: int = 32
    cane_s_lo: int = 25
    cane_v_lo: int = 70
    # แถบสัมภาระเหนือขอบคอก (สัดส่วนของภาพ y0,y1,x0,x1)
    band_y0: float = 0.02
    band_y1: float = 0.42
    band_x0: float = 0.30
    band_x1: float = 0.97
    # ช่วงแนวนอนของกระบะสำหรับ silhouette
    bed_x0: float = 0.35
    bed_x1: float = 0.95
    sil_ylim: float = 0.60      # สแกน silhouette ลึกสุด (สัดส่วนความสูงภาพ)
    min_run_px: int = 8         # non-sky ต้องต่อเนื่องอย่างน้อยกี่พิกเซล
    # เกณฑ์ตัดสิน
    cane_tex_th: float = 0.20
    top_cane_th: float = 0.25
    top_cover_th: float = 0.26

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DetectorConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class DetectionResult:
    label: str        # CANE | COVERED | EMPTY
    label_th: str
    cane_tex: float   # สัดส่วนพิกเซล "สีอ้อย + texture" ในแถบสัมภาระ
    top_med: float    # มัธยฐานความสูง silhouette (ต่ำ = ของกองสูง)

    @property
    def has_cane(self) -> bool:
        return self.label == "CANE"

    @property
    def needs_review(self) -> bool:
        return self.label == "COVERED"

    def to_dict(self) -> dict:
        return asdict(self)


def _load(image: ImageInput) -> np.ndarray:
    """แปลง input ทุกแบบเป็น BGR ndarray"""
    if isinstance(image, np.ndarray):
        img = image
    elif isinstance(image, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(image))
    if img is None or img.size == 0:
        raise ValueError("cannot decode image")
    return img


class CaneDetector:
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()

    # ---------------------------------------------------------------- detect
    def detect(self, image: ImageInput) -> DetectionResult:
        img = _load(image)
        c = self.cfg
        h, w = img.shape[:2]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        sky = (H > c.sky_h_lo) & (H < c.sky_h_hi) & (S > c.sky_s_lo) & (V > c.sky_v_lo)

        # --- silhouette เหนือกระบะ (vectorized) ---
        x0, x1 = int(c.bed_x0 * w), int(c.bed_x1 * w)
        ylim = int(c.sil_ylim * h)
        nonsky = (~sky[:ylim, x0:x1]).astype(np.uint8)
        solidcol = cv2.erode(nonsky, np.ones((c.min_run_px, 1), np.uint8))
        first = np.argmax(solidcol, axis=0)
        hit = solidcol[first, np.arange(solidcol.shape[1])] > 0
        tops = np.where(hit, first / h, c.sil_ylim)
        top_med = float(np.median(tops))

        # --- cane texture ในแถบสัมภาระ ---
        y0, y1 = int(c.band_y0 * h), int(c.band_y1 * h)
        bx0, bx1 = int(c.band_x0 * w), int(c.band_x1 * w)
        band = img[y0:y1, bx0:bx1]
        bh, bs, bv = cv2.split(cv2.cvtColor(band, cv2.COLOR_BGR2HSV))
        cane = (bh >= c.cane_h_lo) & (bh <= c.cane_h_hi) & (bs > c.cane_s_lo) & (bv > c.cane_v_lo)
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        edge_nb = cv2.dilate(cv2.Canny(gray, 60, 150), np.ones((5, 5), np.uint8))
        cane_tex = float((cane & (edge_nb > 0)).mean())

        # --- ตัดสิน ---
        if cane_tex >= c.cane_tex_th and top_med <= c.top_cane_th:
            label = "CANE"
        elif top_med <= c.top_cover_th:
            label = "COVERED"
        else:
            label = "EMPTY"
        return DetectionResult(label, LABELS_TH[label], round(cane_tex, 3), round(top_med, 3))

    def detect_batch(self, images: List[ImageInput]) -> List[DetectionResult]:
        return [self.detect(im) for im in images]

    # -------------------------------------------------------------- annotate
    def annotate(self, image: ImageInput, result: Optional[DetectionResult] = None) -> np.ndarray:
        """คืนภาพพร้อมกรอบสี + label (ตัวอักษรอังกฤษล้วน — ไม่ต้องพึ่ง font ไทย)"""
        img = _load(image).copy()
        res = result or self.detect(img)
        color = _COLORS[res.label]
        th = max(6, img.shape[1] // 200)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, th)
        cv2.putText(img, f"{res.label}  ctex={res.cane_tex} top={res.top_med}",
                    (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        return img


# ------------------------------------------------------- module-level helper
_default: Optional[CaneDetector] = None


def detect(image: ImageInput) -> DetectionResult:
    """Shortcut ใช้ config มาตรฐาน: from cane_detector import detect"""
    global _default
    if _default is None:
        _default = CaneDetector()
    return _default.detect(image)
