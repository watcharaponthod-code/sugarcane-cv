# -*- coding: utf-8 -*-
"""โหมดเข้มงวด — "ห้ามอ่านผิดแม้แต่คันเดียว"

เป้าหมายของไฟล์นี้ไม่ใช่ "อ่านออกทุกคัน" แต่คือ **ถ้าตอบ ต้องถูก**
ทุกเคสที่หลักฐานไม่แน่นพอจะตอบว่า `accepted=False` พร้อมเหตุผล เพื่อส่งให้คนยืนยันแทน
(ความแม่นที่ถูกบันทึกจริง = 100% ส่วนที่หลุดไปเป็นภาระของคน ไม่ใช่ข้อมูลผิด)

ด่านที่ต้องผ่าน:
  1. FORMAT      ต้องเข้ารูปแบบทะเบียนรถบรรทุกไทย NN-NNNN เป๊ะ
  2. CONFIDENCE  ความมั่นใจรายตัวอักษรต่ำสุด >= min_conf
  3. FRAMES      ทุกเฟรมที่อ่านได้ต้องให้เลขตรงกัน (ไม่ใช่แค่เสียงข้างมาก)
  4. CORROBORATION  แยกเป็น 2 ระดับตามความมั่นใจ:
       - conf >= auto_conf  -> รับเลย ไม่ต้องให้ใครตรวจทาน
       - conf ระหว่างกลาง    -> ให้ OCR คนละตระกูลอ่านซ้ำ ต้องได้เลขเดียวกัน

*** ทำไมถึงไม่ให้ตัวที่สองตรวจทานทุกครั้ง ***
วัดจริงบนชุดทดสอบ 60 ภาพ: engine หลัก (fast@onnx+opencv) อ่านถูก 55 ผิด 0
ส่วน easyocr อ่านถูก 50 ผิด 4 — ตัวตรวจทานแม่นน้อยกว่าตัวหลัก
ถ้าให้มันมีสิทธิ์ veto ทุกครั้ง มันจะปัดของที่ถูกอยู่แล้วทิ้ง (เช่น หลักอ่าน 83-6237 conf 1.0
แต่ตัวตรวจทานอ่านเป็น 93-6237 แล้วระบบไม่รับ ทั้งที่ตัวหลักถูก)
การตรวจทานจึงถูกใช้เป็น "หลักฐานเสริม" เฉพาะตอนที่ตัวหลักเองยังไม่มั่นใจพอเท่านั้น
และตัวตรวจทานไม่มีสิทธิ์ "แก้" คำตอบของตัวหลัก มีแค่สิทธิ์ "ยังยืนยันให้ไม่ได้"

การกระจาย confidence ของการอ่านที่ถูกต้อง (จากการวัดจริง):
  >= 0.99 : 46 ครั้ง   >= 0.98 : 2   >= 0.95 : 3   < 0.95 : 4
"""
import re

from lpr import ThaiLPR
from lpr.plate_reader import PlateReader
from lpr.validator import parse_plate

RE_TRUCK_EXACT = re.compile(r"^\d{2}-\d{4}$")

REASONS = {
    "no_plate": "หาป้าย/อ่านไม่ได้เลย",
    "format": "รูปแบบไม่ใช่ NN-NNNN",
    "low_conf": "ความมั่นใจต่ำกว่าเกณฑ์",
    "frames_disagree": "อ่านได้ไม่ตรงกันระหว่างเฟรม",
    "too_few_frames": "อ่านได้จากเฟรมน้อยเกินไป",
    "cross_disagree": "ยังไม่มั่นใจพอ และ OCR ตัวที่สองอ่านได้ไม่ตรงกัน",
    "cross_unavailable": "ยังไม่มั่นใจพอ และไม่มี OCR ตัวที่สองไว้ยืนยัน",
}


def digits(s):
    return re.sub(r"[^0-9]", "", s or "")


class StrictPlateReader:
    """อ่านทะเบียนแบบเข้มงวด — ตอบเมื่อผ่านทุกด่านเท่านั้น

    Parameters
    ----------
    min_conf : ความมั่นใจต่ำสุดที่ยอมรับเลย (ต่ำกว่านี้ = ไม่รับไม่ว่ากรณีใด)
    auto_conf : ถึงระดับนี้ถือว่าหลักฐานแน่นพอในตัวเอง ไม่ต้องให้ใครตรวจทาน
    min_frames : ต้องอ่านได้อย่างน้อยกี่เฟรม (และทุกเฟรมที่อ่านได้ต้องตรงกัน)
    cross : engine ตัวที่สองไว้ยืนยันเฉพาะเคสที่มั่นใจไม่พอ (None = ปิด)
    """

    def __init__(self, engine="fast", detector="onnx+opencv",
                 cross="easyocr", min_conf=0.90, auto_conf=0.98, min_frames=2):
        self.primary = ThaiLPR(engine=engine, detector=detector)
        self.engine = engine
        self.detector = detector
        self.cross_name = cross or None
        self.min_conf = min_conf
        self.auto_conf = max(min_conf, auto_conf)
        self.min_frames = min_frames
        self._cross = None

    def cross_reader(self):
        """โหลด OCR ตัวที่สองตอนใช้ครั้งแรก — อ่านเฉพาะ 'ภาพป้ายที่ crop แล้ว'
        จึงเร็วกว่าการให้มันไล่หาป้ายเองทั้งภาพมาก"""
        if not self.cross_name:
            return None
        if self._cross is None:
            # ป้ายรถบรรทุกเป็นตัวเลขล้วน -> จำกัดชุดตัวอักษรของ easyocr
            # ทั้งเร็วขึ้นและตัดโอกาสอ่านเป็นอักษรมั่วทิ้ง
            from lpr import config as _C
            if self.cross_name == "easyocr" and getattr(_C, "OCR_ALLOWLIST", None) is None:
                _C.OCR_ALLOWLIST = "0123456789-"
            self._cross = PlateReader(self.cross_name)
        return self._cross

    # ------------------------------------------------------------------ main
    def read(self, images):
        """@param images list ของภาพ BGR (หลายเฟรมของรถคันเดียวกัน)
        @returns dict — ดู build_result()"""
        per_frame = [self.primary.process_image(im) for im in images]
        reads = [r[0] for r in per_frame if r]
        if not reads:
            return self._reject("no_plate", frames=len(images))

        best = max(reads, key=lambda r: r["confidence"])
        number = best["number"]

        # ---- ด่าน 1: รูปแบบ ----
        if not number or not RE_TRUCK_EXACT.match(number):
            return self._reject("format", frames=len(images), seen=number, best=best)

        # ---- ด่าน 2: ความมั่นใจ ----
        conf = min(float(r["confidence"]) for r in reads if digits(r["number"]) == digits(number))
        if conf < self.min_conf:
            return self._reject("low_conf", frames=len(images), seen=number,
                                best=best, confidence=conf)

        # ---- ด่าน 3: ทุกเฟรมที่อ่านได้ต้องตรงกัน ----
        uniq = {digits(r["number"]) for r in reads}
        if len(uniq) > 1:
            return self._reject("frames_disagree", frames=len(images), best=best,
                                seen=" / ".join(sorted(uniq)), confidence=conf)
        need = min(self.min_frames, len(images))
        if len(reads) < need:
            return self._reject("too_few_frames", frames=len(images), seen=number,
                                best=best, confidence=conf, votes=len(reads))

        # ---- ด่าน 4a: มั่นใจสูงพอในตัวเอง -> รับเลย ----
        # ตัวหลักที่ conf ระดับนี้ไม่เคยอ่านผิดในชุดทดสอบ การให้ OCR ที่แม่นน้อยกว่า
        # มา veto จึงมีแต่จะทำให้ปัดของถูกทิ้ง
        if conf >= self.auto_conf:
            return self._accept(number, best, conf, len(reads), len(images),
                                None, verified="confidence+frames")

        # ---- ด่าน 4b: ยังไม่มั่นใจพอ -> ขอหลักฐานเสริมจาก OCR คนละตระกูล ----
        cross = self.cross_reader()
        if cross is None:
            return self._reject("cross_unavailable", frames=len(images), seen=number,
                                best=best, confidence=conf, votes=len(reads))
        cross_lines = cross.read(best["plate_bgr"])
        cross_parsed = parse_plate(cross_lines)
        cross_num = cross_parsed["number"] if cross_parsed else None
        if digits(cross_num) != digits(number):
            return self._reject("cross_disagree", frames=len(images), seen=number,
                                best=best, confidence=conf, votes=len(reads),
                                cross=cross_num)

        return self._accept(number, best, conf, len(reads), len(images), cross_num,
                            verified="cross-check")

    # ------------------------------------------------------------- ผลลัพธ์
    def _base(self, best=None, frames=0):
        return {
            "number": None, "province": None, "plate_type": None,
            "confidence": 0.0, "votes": 0, "frames": frames,
            "bbox": list(map(int, best["bbox"])) if best and best.get("bbox") else None,
            "raw": (best or {}).get("raw", ""),
            "engine": f"{self.engine}@{self.detector}",
            "strict": True, "accepted": False, "reason": None, "reason_th": None,
            "candidate": None, "cross": None, "verified_by": None,
            "auto_conf": self.auto_conf, "min_conf": self.min_conf,
        }

    def _reject(self, reason, frames=0, seen=None, best=None, confidence=0.0,
                votes=0, cross=None):
        d = self._base(best, frames)
        d.update(reason=reason, reason_th=REASONS.get(reason, reason),
                 candidate=seen, confidence=round(float(confidence), 3),
                 votes=votes, cross=cross)
        return d

    def _accept(self, number, best, conf, votes, frames, cross, verified="cross-check"):
        d = self._base(best, frames)
        d.update(number=number, province=best.get("province"),
                 plate_type=best.get("plate_type"), confidence=round(float(conf), 3),
                 votes=votes, accepted=True, candidate=number, cross=cross,
                 verified_by=verified)
        return d
