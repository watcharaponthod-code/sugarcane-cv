# -*- coding: utf-8 -*-
"""ตัวอ่านป้าย — แพ็กเกจนี้มีตัวอ่านเดียว: template matching (CV ล้วน 100%)
ไม่มีการ import easyocr / pytesseract / torch ใดๆ ทั้งสิ้น"""
import cv2

from . import config as C
from .plate_detector import deskew
from .reader_template import TemplateReader


def _prep_gray(plate_bgr):
    """เตรียมภาพป้าย: ปรับขนาด -> grayscale -> ปรับ contrast -> deskew"""
    h, w = plate_bgr.shape[:2]
    if h < C.OCR_PLATE_HEIGHT:
        scale = C.OCR_PLATE_HEIGHT / float(h)
        plate_bgr = cv2.resize(plate_bgr, (int(w * scale), C.OCR_PLATE_HEIGHT),
                               interpolation=cv2.INTER_CUBIC)
    elif h > C.OCR_MAX_HEIGHT:
        scale = C.OCR_MAX_HEIGHT / float(h)
        plate_bgr = cv2.resize(plate_bgr, (int(w * scale), C.OCR_MAX_HEIGHT),
                               interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = deskew(gray)
    return gray


class PlateReader:
    """สร้างครั้งเดียวแล้วเรียก .read() ซ้ำได้"""

    def __init__(self, engine="template"):
        # แพ็กเกจ Pure CV รองรับเฉพาะ template matching
        self._template = TemplateReader()
        self.engine = "template"

    def read(self, plate_bgr):
        """อ่านป้าย 1 ใบ -> list ของ (ข้อความ, คะแนน 0-1)"""
        gray = _prep_gray(plate_bgr)
        gray_orig = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        scale = gray.shape[0] / float(max(1, gray_orig.shape[0]))
        return self._template.read(gray, gray_orig=gray_orig, scale=scale)
