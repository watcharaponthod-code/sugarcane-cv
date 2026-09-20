# -*- coding: utf-8 -*-
"""อ่านตัวอักษรบนป้ายทะเบียน (OCR)

เลือก engine ได้ 2 แบบ:
  easyocr   : แม่นกว่าสำหรับภาษาไทย (deep learning, ต้องติดตั้ง PyTorch)
  tesseract : เบา รันได้ทุกเครื่อง — ติดตั้ง tesseract-ocr + ข้อมูลภาษาไทย (tha)
  auto      : ใช้ easyocr ถ้ามีในเครื่อง ไม่มีก็ fallback เป็น tesseract อัตโนมัติ
"""
import cv2
import numpy as np

from . import config as C
from .plate_detector import deskew


def _prep_gray(plate_bgr):
    """เตรียมภาพป้าย: ปรับขนาด -> grayscale -> ปรับ contrast -> deskew
    จำกัดขนาดทั้งขั้นต่ำ (OCR อ่านออก) และขั้นสูง (กันกินแรมเกิน)"""
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
    """ตัวอ่านป้าย — สร้างครั้งเดียวแล้วเรียก .read() ซ้ำได้ (โหลดโมเดลครั้งเดียว)"""

    def __init__(self, engine="auto"):
        self.engine = None
        self._easy = None

        if engine == "template":
            from .reader_template import TemplateReader
            self._template = TemplateReader()
            self.engine = "template"
            return

        if engine == "fast":
            from .reader_fast import FastPlateReader
            self._fast = FastPlateReader()
            self.engine = "fast"
            return

        if engine in ("auto", "easyocr"):
            try:
                import easyocr
                self._easy = easyocr.Reader(["th", "en"], gpu=False,
                                            verbose=False)
                self.engine = "easyocr"
            except ImportError:
                if engine == "easyocr":
                    raise RuntimeError(
                        "ยังไม่ได้ติดตั้ง easyocr — รัน: pip install easyocr")

        if self.engine is None:
            try:
                import pytesseract
                self._tess = pytesseract
                # ทดสอบว่าเรียก binary ได้จริง
                self._tess.get_tesseract_version()
            except Exception as e:
                raise RuntimeError(
                    "ไม่พบ OCR engine — ติดตั้ง easyocr (pip install easyocr) "
                    "หรือ tesseract-ocr + tesseract-ocr-tha อย่างใดอย่างหนึ่ง"
                ) from e
            self.engine = "tesseract"

    # ------------------------------------------------------------ easyocr
    # ชุดพารามิเตอร์ CRAFT เรียงตามผลทดสอบจริง: mag_ratio=2 อ่านป้ายชัด/เล็ก
    # ได้ดีสุด จึงยิงก่อน — ส่วนใหญ่จบใน pass เดียว
    _EASY_PASSES = (
        dict(mag_ratio=2.0),
        dict(),
        dict(mag_ratio=2.0, text_threshold=0.5, low_text=0.3),
    )

    def _read_easyocr(self, gray):
        """คืน list ของ (ข้อความ, ความมั่นใจ) — ยิงหลาย pass แล้วรวมผล
        หยุดเร็วเมื่อได้ข้อความ 'หน้าตาเหมือนทะเบียน' และข้าม candidate
        ที่ pass แรกว่างเปล่า (ไม่มีตัวหนังสือให้เสียเวลา)"""
        all_lines = []
        for i, params in enumerate(self._EASY_PASSES):
            extra = {"allowlist": C.OCR_ALLOWLIST} if getattr(C, "OCR_ALLOWLIST", None) else {}
            res = self._easy.readtext(gray, detail=1, paragraph=False,
                                      canvas_size=C.EASYOCR_CANVAS, **params, **extra)
            if not res and i == 0:
                return []                       # ไม่มีข้อความเลย -> จบ
            items = []
            for box, text, conf in res:
                ys = [p[1] for p in box]
                items.append((sum(ys) / len(ys), text.strip(), float(conf)))
            items.sort(key=lambda t: t[0])      # เรียงตามแนวตั้ง
            lines = [(t, c) for _, t, c in items if t]
            all_lines.extend(lines)
            # early stop: ต่อเมื่อผลที่ได้ "validate เป็นทะเบียนได้จริง" แล้ว
            # (แค่ตัวอักษรเยอะยังไม่พอ — บางทีอ่านเพี้ยนจน conf ต่ำ
            #  ต้องปล่อยให้ pass ถัดไปที่พารามิเตอร์ต่างกันได้ลองอ่านด้วย)
            from .validator import parse_plate
            parsed = parse_plate(lines)
            if parsed and parsed["number"] \
                    and parsed["confidence"] >= C.OCR_MIN_CONF:
                break
        return all_lines

    # ---------------------------------------------------------- tesseract
    def _tess_line(self, img, langs, psm=7):
        """อ่าน 1 บรรทัด คืน (ข้อความ, conf เฉลี่ย) หรือ None"""
        # ขอบขาวช่วยให้ tesseract อ่านแม่นขึ้น
        img = cv2.copyMakeBorder(img, 12, 12, 16, 16,
                                 cv2.BORDER_CONSTANT, value=255)
        data = self._tess.image_to_data(
            img, lang=langs, config=f"--psm {psm}",
            output_type=self._tess.Output.DICT)
        words, confs = [], []
        for txt, cf in zip(data["text"], data["conf"]):
            cf = float(cf)
            if txt.strip() and cf > 0:
                words.append(txt.strip())
                confs.append(cf / 100.0)
        if not words:
            return None
        return (" ".join(words), sum(confs) / len(confs))

    @staticmethod
    def _binarize(img):
        th = cv2.threshold(img, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return cv2.medianBlur(th, 3)

    def _read_tesseract(self, gray):
        """Tesseract บนป้ายรถไม่เสถียร จึงยิงหลาย variant
        (binary/gray x psm 7/13 + อ่านทั้งป้าย psm 6) แล้วส่งผลทุกชุด
        ให้ validator เลือกชุดที่เข้ารูปแบบทะเบียนไทยดีที่สุด"""
        H, W = gray.shape[:2]
        single_line = (W / float(H)) > 3.5   # ป้ายบรรทัดเดียว (strip แคบยาว)

        lines = []

        def try_all(strip, langs):
            variants = (self._binarize(strip), strip)
            for var in variants:
                for psm in (7, 13):
                    r = self._tess_line(var, langs, psm)
                    if r:
                        lines.append(r)

        if single_line:
            try_all(gray, "tha+eng")
        else:
            try_all(gray[0:int(H * 0.65), :], "tha+eng")   # บรรทัดเลขทะเบียน
            # ป้ายรุ่นเก่า 3 บรรทัด (THAILAND / เลข / จังหวัด) เลขอยู่แถบกลาง
            try_all(gray[int(H * 0.22):int(H * 0.78), :], "tha+eng")
            try_all(gray[int(H * 0.55):, :], "tha")        # บรรทัดจังหวัด
            # เผื่อการแบ่งบรรทัดพลาด: อ่านทั้งป้ายแบบหลายบรรทัดด้วย
            for var in (self._binarize(gray), gray):
                r = self._tess_line(var, "tha+eng", psm=6)
                if r:
                    lines.append(r)
        return lines

    # -------------------------------------------------------------- public
    def read(self, plate_bgr):
        """อ่านป้าย 1 ใบ

        Returns
        -------
        list ของ (ข้อความดิบ, ความมั่นใจ 0-1) เรียงจากบรรทัดบนลงล่าง
        """
        if self.engine == "fast":
            return self._fast.read(plate_bgr)   # โมเดลต้องการภาพสีดิบ ทำ preprocess ในตัวเอง
        gray = _prep_gray(plate_bgr)
        if self.engine == "template":
            gray_orig = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
            scale = gray.shape[0] / float(max(1, gray_orig.shape[0]))
            return self._template.read(gray, gray_orig=gray_orig, scale=scale)
        if self.engine == "easyocr":
            return self._read_easyocr(gray)
        return self._read_tesseract(gray)
