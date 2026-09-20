# -*- coding: utf-8 -*-
"""ตัวอ่านตัวเลขแบบ Computer Vision ล้วน 100% — ไม่มีโมเดล ML เลยแม้แต่ตัวเดียว

หลักการ (เทคนิคเดียวกับระบบ ANPR ยุคก่อน deep learning):
  1. binarize ป้าย -> หา connected components ขนาดเท่าตัวอักษร
  2. เลือก "แถวตัวเลข" (แถวที่ตัวอักษรสูงสุด) เรียงซ้าย->ขวา
  3. normalize แต่ละตัวเป็นกล่องมาตรฐาน แล้วเทียบกับ template ตัวเลข 0-9
     ด้วย normalized correlation — ขีด (-) ตรวจจากรูปทรง (แบนกว้าง)

ใช้ได้จริงกับป้ายรถบรรทุก (NN-NNNN = ตัวเลขล้วน) ภายใต้เงื่อนไข:
  - ป้ายในภาพกว้างพอ (>= ~150px) และแสงสม่ำเสมอ
  - ฟอนต์ป้ายทะเบียนไทยเป็นมาตรฐานกรมขนส่งฯ ทั้งประเทศ template จึงคงที่

โปรทิป production: แทน template จากฟอนต์ระบบ ด้วยการ crop ตัวเลขจริง
จากภาพป้ายที่หน้างาน 1 ชุด (0-9 อย่างละตัว) — ยังเป็น CV ล้วน ไม่ใช่การเทรนโมเดล
แต่แม่นขึ้นมากเพราะตรงฟอนต์จริงเป๊ะ (วางไฟล์ png ไว้ที่ lpr/templates/)
"""
import os
import glob

import cv2
import numpy as np

# ฟอนต์ระบบที่ใช้สร้าง template ตั้งต้น (หลายฟอนต์ = ทนความต่างของฟอนต์จริง)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/tahomabd.ttf",
]
_BOX_W, _BOX_H = 32, 48          # ขนาดกล่องมาตรฐานตอนเทียบ
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _norm_box(binary_char):
    """ตัดขอบขาวออก แล้วย่อ/ขยายเป็นกล่องมาตรฐาน (คงสัดส่วน กึ่งกลาง)"""
    ys, xs = np.where(binary_char > 0)
    if len(xs) == 0:
        return None
    ch = binary_char[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ch.shape
    scale = min((_BOX_W - 4) / w, (_BOX_H - 4) / h)
    ch = cv2.resize(ch, (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA)
    out = np.zeros((_BOX_H, _BOX_W), np.uint8)
    y0 = (_BOX_H - ch.shape[0]) // 2
    x0 = (_BOX_W - ch.shape[1]) // 2
    out[y0:y0 + ch.shape[0], x0:x0 + ch.shape[1]] = ch
    return out


class TemplateReader:
    """อ่านตัวเลขบนป้ายด้วย template matching ล้วนๆ"""

    def __init__(self):
        self.templates = {}          # '0'-'9' -> [ภาพ template หลายฟอนต์]
        self._build_templates()

    # ------------------------------------------------------------ templates
    def _build_templates(self):
        # 1) template จากภาพจริงที่ผู้ใช้เตรียมไว้ (แม่นสุด) — lpr/templates/3.png
        for p in glob.glob(os.path.join(_TEMPLATE_DIR, "*.png")):
            digit = os.path.splitext(os.path.basename(p))[0]
            if digit.isdigit() and len(digit) == 1:
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV
                                   | cv2.THRESH_OTSU)[1]
                if th.mean() > 127:          # เผื่อภาพเป็นตัวขาวพื้นดำอยู่แล้ว
                    th = 255 - th
                box = _norm_box(th)
                if box is not None:
                    self.templates.setdefault(digit, []).append(box)

        # 2) template จากฟอนต์ระบบ (ตั้งต้น)
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return
        for path in _FONT_CANDIDATES:
            if not os.path.exists(path):
                continue
            font = ImageFont.truetype(path, 96)
            for d in "0123456789":
                im = Image.new("L", (140, 140), 0)
                dr = ImageDraw.Draw(im)
                dr.text((20, 10), d, font=font, fill=255)
                box = _norm_box(np.array(im))
                if box is not None:
                    self.templates.setdefault(d, []).append(box)

    # ------------------------------------------------------------ segment
    @staticmethod
    def _segment_number_row(gray):
        """คืน list ของ (x, binary_char, w, h) ของแถวตัวเลข เรียงซ้าย->ขวา
        + ตำแหน่ง gap ที่น่าจะเป็นขีด"""
        H, W = gray.shape
        th = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                              np.ones((2, 2), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(th)
        comps = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if h < H * 0.25 or h > H * 0.95 or w > W * 0.35 or area < 12:
                continue
            comps.append((x, y, w, h, i))
        if len(comps) < 3:
            return [], None, th

        # เลือกแถว: จัดกลุ่มตาม y-กึ่งกลาง เอากลุ่มที่ตัวอักษร "สูงสุด"
        comps.sort(key=lambda c: c[1] + c[3] / 2)
        rows = []
        for c in comps:
            cy = c[1] + c[3] / 2
            for row in rows:
                if abs(row["cy"] - cy) < max(c[3], row["h"]) * 0.5:
                    row["items"].append(c)
                    row["cy"] = np.mean([i[1] + i[3] / 2
                                         for i in row["items"]])
                    row["h"] = max(row["h"], c[3])
                    break
            else:
                rows.append({"cy": cy, "h": c[3], "items": [c]})
        rows = [r for r in rows if len(r["items"]) >= 3]
        if not rows:
            return [], None, th
        best = max(rows, key=lambda r: r["h"])
        items = sorted(best["items"], key=lambda c: c[0])
        med_h = np.median([c[3] for c in items])

        chars, dash_x = [], None
        for x, y, w, h, i in items:
            mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
            if w > h * 1.4 and h < med_h * 0.55:      # แบนกว้าง = ขีด
                dash_x = x
                continue
            if h < med_h * 0.55:                       # จุดสกรู/ฝุ่น
                continue
            chars.append((x, y, w, h, mask))
        return chars, dash_x, th

    # ------------------------------------------------------------ structure
    @staticmethod
    def _tighten(gray_char):
        """หดกรอบให้พอดีตัวอักษร (กันขอบป้าย/เพื่อนบ้านติดมาตอน map พิกัด)"""
        otsu, ink = cv2.threshold(gray_char, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
        if n < 2:
            return gray_char
        i = 1 + int(np.argmax(stats[1:, 4]))          # ก้อนหมึกใหญ่สุด
        x, y, w, h = stats[i][:4]
        if w < 3 or h < 6:
            return gray_char
        x0, y0 = max(0, x - 1), max(0, y - 1)
        return gray_char[y0:y + h + 1, x0:x + w + 1]

    @staticmethod
    def _edge_gaps(gray_char):
        """วัด 'ช่องเปิด' สี่ตำแหน่ง (ขวาบน/ขวาล่าง/ซ้ายบน/ซ้ายล่าง)
        = ระยะเฉลี่ยจากขอบภาพถึงหมึกแรก (สัดส่วนของความกว้าง)"""
        g = gray_char
        otsu, _ = cv2.threshold(g, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        ink = (g < otsu).astype(np.uint8)
        H, W = ink.shape
        if H < 8 or W < 4:
            return None

        def gap(rows, from_right):
            vals = []
            for y in rows:
                xs = np.where(ink[y] > 0)[0]
                if len(xs) == 0:
                    vals.append(1.0)
                elif from_right:
                    vals.append((W - 1 - xs.max()) / W)
                else:
                    vals.append(xs.min() / W)
            return float(np.mean(vals)) if vals else 1.0

        top = range(int(H * 0.18), max(int(H * 0.18) + 1, int(H * 0.40)))
        bot = range(int(H * 0.60), max(int(H * 0.60) + 1, int(H * 0.84)))
        return {
            "rt": gap(top, True),  "rb": gap(bot, True),
            "lt": gap(top, False), "lb": gap(bot, False),
        }

    def _disambiguate(self, digit, gray_char, strong):
        """แก้ความสับสน 8/6/9 ด้วยโครงสร้าง (ทำงานเมื่อ template ตอบ '8')

        หลักการ: ความเบลอทำให้ช่องเปิดของ 6 (ขวาบน) และ 9 (ซ้ายล่าง)
        ถูก 'ปิด' จนหน้าตาเหมือน 8 — แต่ร่องรอยช่องเปิดยังวัดได้จาก
        ภาพความละเอียดต้นฉบับ (ก่อน upscale) เพราะรอยเชื่อมปลอมตื้นกว่าเส้นจริง
        เกณฑ์: ช่องฝั่งนั้นลึกพอ และลึกกว่าฝั่งตรงข้ามชัดเจน (เลข 8 แท้สมมาตร)"""
        if digit != "8" or gray_char is None:
            return digit
        gray_char = self._tighten(gray_char)
        gaps = self._edge_gaps(gray_char)
        if gaps is None:
            return digit
        deep, diff = (0.25, 0.10) if strong else (0.12, 0.06)
        if gaps["rt"] > deep and gaps["rt"] - gaps["rb"] > diff:
            return "6"           # เปิดขวาบน ปิดขวาล่าง = เลข 6
        if gaps["lb"] > deep and gaps["lb"] - gaps["lt"] > diff:
            return "9"           # เปิดซ้ายล่าง ปิดซ้ายบน = เลข 9
        return digit

    # ------------------------------------------------------------ match
    def _match_digit(self, binary_char):
        box = _norm_box(binary_char)
        if box is None:
            return None, 0.0
        best_d, best_s = None, -1.0
        boxf = box.astype(np.float32)
        for d, tmpls in self.templates.items():
            for t in tmpls:
                s = cv2.matchTemplate(boxf, t.astype(np.float32),
                                      cv2.TM_CCOEFF_NORMED)[0][0]
                if s > best_s:
                    best_d, best_s = d, float(s)
        return best_d, best_s

    # ------------------------------------------------------------ public
    def read(self, gray, gray_orig=None, scale=1.0):
        """อ่านแถวตัวเลขจากภาพป้าย

        Parameters
        ----------
        gray      : ภาพป้ายที่เตรียมแล้ว (upscale/CLAHE) ใช้แยกตัวอักษร
        gray_orig : ภาพป้ายความละเอียดต้นฉบับ (ก่อน upscale) — ใช้ตรวจ
                    โครงสร้าง 6/8/9 เพราะรายละเอียดช่องเปิดยังไม่ถูกเกลี่ย
        scale     : gray.height / gray_orig.height สำหรับ map พิกัดกลับ

        Returns
        -------
        list [(ข้อความ, คะแนนเฉลี่ย 0-1)] เข้ากันได้กับ PlateReader.read()
        """
        chars, dash_x, _ = self._segment_number_row(gray)
        if len(chars) < 3:
            return []
        out, scores = [], []
        for x, y, w, h, mask in chars:
            if dash_x is not None and out and x > dash_x and "-" not in out:
                out.append("-")
            d, s = self._match_digit(mask)
            if d is None:
                continue
            # ตรวจโครงสร้างซ้ำ (แก้ 6/9 ที่เบลอจนเหมือน 8):
            # ใช้ภาพต้นฉบับ (รายละเอียดยังไม่ถูกเกลี่ย) + tighten กรอบ เกณฑ์เข้ม
            # — เกณฑ์ผ่อนบนภาพ upscale ใช้เฉพาะเมื่อไม่มีภาพต้นฉบับเท่านั้น
            if d == "8":
                oc = None
                if gray_orig is not None and scale > 0:
                    ox, oy = int(x / scale), int(y / scale)
                    ow, oh = max(4, int(round(w / scale)) + 2), \
                        max(8, int(round(h / scale)) + 2)
                    oc = gray_orig[max(0, oy - 1):oy + oh,
                                   max(0, ox - 1):ox + ow]
                if oc is not None and oc.shape[0] >= 8 and oc.shape[1] >= 4:
                    d = self._disambiguate(d, oc, strong=True)
                else:
                    d = self._disambiguate(d, gray[y:y + h, x:x + w],
                                           strong=False)
            out.append(d)
            scores.append(s)
        if not scores:
            return []
        text = "".join(out)
        # แปลงคะแนน correlation (-1..1) เป็น 0..1 คร่าวๆ
        conf = float(np.clip((np.mean(scores) + 1) / 2, 0, 1))
        return [(text, round(conf, 3))]
