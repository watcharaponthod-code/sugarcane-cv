# -*- coding: utf-8 -*-
"""Thai LPR — ระบบอ่านป้ายทะเบียนรถไทยด้วย OpenCV"""
from .pipeline import ThaiLPR, draw_text_thai
from .plate_detector import detect_plates
from .validator import parse_plate, match_province, PROVINCES

__all__ = ["ThaiLPR", "draw_text_thai", "detect_plates",
           "parse_plate", "match_province", "PROVINCES"]
__version__ = "1.0.0"
