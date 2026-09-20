# -*- coding: utf-8 -*-
"""Thai LPR Pure CV — อ่านป้ายรถบรรทุกด้วย Computer Vision ล้วน 100%"""
from .pipeline import ThaiLPR, draw_text_thai
from .plate_detector import detect_plates
from .validator import parse_plate, match_province, PROVINCES
from .aggregator import PlateVote

__all__ = ["ThaiLPR", "draw_text_thai", "detect_plates",
           "parse_plate", "match_province", "PROVINCES", "PlateVote"]
__version__ = "1.0-pure"
