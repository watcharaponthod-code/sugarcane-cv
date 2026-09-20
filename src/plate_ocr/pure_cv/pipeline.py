# -*- coding: utf-8 -*-
"""Pipeline หลัก: หาป้าย -> อ่าน -> ตรวจรูปแบบ -> คืนผล + วาดกำกับบนภาพ"""
import os

import cv2
import numpy as np

from . import config as C
from .plate_detector import detect_plates
from .plate_reader import PlateReader
from .validator import parse_plate

# ฟอนต์ไทยสำหรับวาดข้อความบนภาพ (ลองตามลำดับ)
_FONT_PATHS = [
    "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf",
    "/usr/share/fonts/opentype/tlwg/Loma.otf",
    "/usr/share/fonts/truetype/tlwg/Loma-Bold.ttf",
    "/usr/share/fonts/truetype/tlwg/Loma.ttf",
    "C:/Windows/Fonts/tahomabd.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Ayuthaya.ttf",
]


def _find_thai_font():
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return p
    return None


def draw_text_thai(img_bgr, text, xy, font_size=32,
                   color=(0, 255, 0), bg=(0, 0, 0)):
    """วาดข้อความไทยบนภาพ OpenCV (ผ่าน PIL) — ถ้าไม่มีฟอนต์ไทย fallback เป็น cv2"""
    font_path = _find_thai_font()
    if font_path is None:
        ascii_only = "".join(ch for ch in text if ord(ch) < 128) or "plate"
        cv2.putText(img_bgr, ascii_only, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 2, cv2.LINE_AA)
        return img_bgr

    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, font_size)
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    x, y = xy
    # กล่องพื้นหลังให้อ่านง่าย
    box = draw.textbbox((x, y), text, font=font)
    draw.rectangle([box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2], fill=bg)
    draw.text((x, y), text, font=font,
              fill=(color[2], color[1], color[0]))     # BGR -> RGB
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


class ThaiLPR:
    """ระบบอ่านทะเบียนรถไทยครบวงจร

    ตัวอย่างการใช้:
        lpr = ThaiLPR()
        results = lpr.process_image(cv2.imread("truck.jpg"))
        for r in results:
            print(r["number"], r["province"], r["confidence"])
    """

    def __init__(self, engine=C.OCR_ENGINE, debug_dir=None):
        """ระบบ CV ล้วน 100% — หาตำแหน่งด้วย OpenCV, อ่านด้วย template matching"""
        self.reader = PlateReader(engine)
        self.debug_dir = debug_dir
        print(f"[ThaiLPR] Pure CV mode (ไม่มีโมเดล ML)")

    def _detect(self, bgr):
        return detect_plates(bgr, self.debug_dir)

    def process_image(self, bgr):
        """คืน list ของผลอ่าน เรียงตามความมั่นใจมาก->น้อย

        แต่ละผลเป็น dict:
            number, province, plate_type, confidence, raw, bbox, plate_bgr
        """
        candidates = self._detect(bgr)

        results = self._read_candidates(candidates)

        # อ่านไม่ได้สักใบ -> ภาพนี้อาจเป็น "ป้ายที่ crop มาแล้ว" ลอง OCR ทั้งภาพ
        # (เฉพาะภาพที่สัดส่วนเหมือนป้ายจริง กันภาพถ่ายทั้งฉากหลุดเข้ามา)
        if not results and C.FALLBACK_FULL_IMAGE:
            H, W = bgr.shape[:2]
            ar = W / float(H)
            if 1.3 <= ar <= 6.5:
                results = self._read_candidates([{
                    "bbox": (0, 0, W, H),
                    "plate_bgr": bgr,
                    "score": 0,
                }])

        def _rank(r):
            """คะแนนรวม: หลักฐาน (ขีด/จังหวัด) สำคัญสุด > จำนวนวิธีที่เห็น
            > ตำแหน่ง (ป้ายจริงอยู่ 'ต่ำและกลาง' ตัวรถ ต่างจากสติกเกอร์บนกระจก)
            > ความมั่นใจ OCR"""
            evidence = int(bool(r["province"])) + int(r.get("has_dash", False))
            pos = 0.0
            if C.POSITION_PRIOR:
                x, y, w, h = r["bbox"]
                H, W = bgr.shape[:2]
                y_low = (y + h / 2) / float(H)             # ยิ่งต่ำยิ่งดี
                centered = 1 - abs((x + w / 2) - W / 2) / (W / 2.0)
                pos = 0.7 * y_low + 0.3 * centered
            return (evidence * 10 + r["detect_score"] * 2
                    + pos * 1.5 + r["confidence"])

        results.sort(key=_rank, reverse=True)

        # (โค้ดต่อจากนี้คือ NMS รอบสุดท้าย)

        # NMS รอบสุดท้าย: กรอบซ้อนกันมาก = ป้ายเดียวกัน เก็บอันที่ดีที่สุด
        final = []
        for r in results:
            dup = False
            for f in final:
                if self._box_overlap(r["bbox"], f["bbox"]) > 0.5:
                    dup = True
                    break
            if not dup:
                final.append(r)
        # โหมดลานชั่ง: รถผ่านทีละคัน -> ตอบป้ายที่ดีที่สุด "ใบเดียว"
        if C.SINGLE_PLATE_MODE:
            return final[:1]
        return final

    def _read_candidates(self, candidates):
        """OCR + ตรวจรูปแบบ ทีละ candidate คืนเฉพาะผลที่ผ่านเกณฑ์
        เจอผลที่หลักฐานแน่น (มีขีด/จังหวัด) แล้วหยุดทันที — ประหยัดเวลามาก
        เพราะงานหน้าลานชั่งมีรถทีละคันอยู่แล้ว (ปิดได้ที่ config)"""
        results = []
        for cand in candidates:
            lines = self.reader.read(cand["plate_bgr"])
            parsed = parse_plate(lines)
            if parsed is None:
                continue
            if parsed["number"] is None:
                continue                      # ต้องได้เลขทะเบียนเป็นอย่างน้อย
            if parsed["confidence"] < C.OCR_MIN_CONF:
                continue
            # ผล "อ่อน" = ไม่มีทั้งขีด(-) และจังหวัด -> ต้องมั่นใจสูงพอ
            has_evidence = parsed["province"] or parsed.get("has_dash")
            if not has_evidence and \
                    parsed["confidence"] < C.WEAK_RESULT_MIN_CONF:
                continue
            parsed["bbox"] = cand["bbox"]
            parsed["plate_bgr"] = cand["plate_bgr"]
            parsed["detect_score"] = cand["score"]
            results.append(parsed)
            if C.EARLY_STOP_ON_STRONG and has_evidence \
                    and parsed["confidence"] >= 0.40:
                break
            if len(results) >= 2:            # ได้สองผลแล้วพอ กันเสียเวลา
                break
        return results

    @staticmethod
    def _box_overlap(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        return inter / min(aw * ah, bw * bh)

    def annotate(self, bgr, results):
        """วาดกรอบ + เลขทะเบียน + จังหวัด ลงบนสำเนาภาพ"""
        vis = bgr.copy()
        for r in results:
            x, y, w, h = r["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = r["number"] or "?"
            if r["province"]:
                label += f"  {r['province']}"
            label += f"  ({r['confidence']:.0%})"
            ty = max(0, y - 42)
            vis = draw_text_thai(vis, label, (x, ty), font_size=34)
        return vis
