# -*- coding: utf-8 -*-
"""ตรวจสอบและจัดรูปแบบผล OCR ให้เป็นทะเบียนไทยที่ถูกต้อง

รูปแบบที่รองรับ:
  รถตาม พ.ร.บ.ขนส่งทางบก (รถบรรทุก/รถบัส) : NN-NNNN      เช่น 70-1234
  รถยนต์ทั่วไป                              : [เลข]กก NNNN  เช่น 1กข 1234, กข 1234
จังหวัดใช้ fuzzy matching กับรายชื่อ 77 จังหวัด (OCR มักอ่านเพี้ยนเล็กน้อย)
"""
import re
import difflib

from . import config as C

PROVINCES = [
    "กระบี่", "กรุงเทพมหานคร", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ",
    "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก",
    "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์",
    "นนทบุรี", "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี",
    "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา",
    "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
    "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน", "ยโสธร",
    "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี", "ลพบุรี", "ลำปาง",
    "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี",
    "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย",
    "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์",
    "อุทัยธานี", "อุบลราชธานี",
]

# NN-NNNN (จับได้ทั้งมี/ไม่มีขีด — แต่ "มีขีดจริง" จะถูกใช้เป็นหลักฐานความน่าเชื่อ)
RE_TRUCK = re.compile(r"(\d{2})\s*(-?)\s*(\d{4})")
# [เลขนำ 0-1 ตัว] พยัญชนะไทย 1-2 ตัว [เลขตาม 1-4 ตัว]
RE_CAR = re.compile(r"(\d)?\s*([ก-ฮ]{1,2})\s*(\d{1,4})")


def _clean(text):
    """เก็บเฉพาะอักขระที่เป็นไปได้บนบรรทัดเลขทะเบียน"""
    return re.sub(r"[^0-9ก-ฮ\s\-]", "", text).strip()


def match_province(text):
    """จับคู่ข้อความกับชื่อจังหวัด (ทน OCR อ่านเพี้ยน) คืน None ถ้าไม่เข้าเค้า"""
    t = re.sub(r"[^ก-๙]", "", text)
    if len(t) < 3:
        return None
    if t in PROVINCES:
        return t
    hit = difflib.get_close_matches(t, PROVINCES, n=1, cutoff=0.6)
    return hit[0] if hit else None


def parse_plate(lines):
    """แปลงผล OCR ดิบ -> ทะเบียนที่จัดรูปแบบแล้ว

    Parameters
    ----------
    lines : list ของ (ข้อความ, ความมั่นใจ 0-1) จาก PlateReader.read()

    Returns
    -------
    dict {number, province, plate_type, confidence, raw}  หรือ None ถ้าไม่เข้าเค้าเลย
    """
    if not lines:
        return None

    raw = " | ".join(t for t, _ in lines)

    # เก็บ "ทุก" match จากทุกบรรทัด/variant แล้วค่อยเลือกอันดีที่สุด
    # เกณฑ์เลือก: รูปแบบรถบรรทุกมาก่อน (บริบทโรงงาน) > อ่านได้ครบตัวอักษรกว่า > conf สูงกว่า
    num_matches = []   # (priority, evidence/len, conf, number, type, has_dash)
    prov_best = None   # (ratio, province, conf)

    for text, conf in lines:
        cleaned = _clean(text)

        m = RE_TRUCK.search(cleaned)
        if m:
            has_dash = m.group(2) == "-"
            num_matches.append((2, 1 if has_dash else 0, conf,
                                f"{m.group(1)}-{m.group(3)}", "truck",
                                has_dash))

        m = RE_CAR.search(cleaned)
        if m and m.group(3) and len(m.group(3)) >= C.MIN_NUMBER_DIGITS:
            lead = m.group(1) or ""
            total = len(lead) + len(m.group(2)) + len(m.group(3))
            # ทะเบียนจริงสั้นสุดคือ กก 12 = 4 ตัวอักษร (กันอ่านขยะ)
            if total >= 4:
                num_matches.append((1, total, conf,
                                    f"{lead}{m.group(2)} {m.group(3)}",
                                    "car", False))

        # จับคู่จังหวัด — เก็บอันที่คล้ายที่สุด
        t = re.sub(r"[^ก-๙]", "", text)
        if len(t) >= 3:
            hit = difflib.get_close_matches(t, PROVINCES, n=1, cutoff=0.6)
            if hit:
                ratio = difflib.SequenceMatcher(None, t, hit[0]).ratio()
                if prov_best is None or ratio > prov_best[0]:
                    prov_best = (ratio, hit[0], conf)

    number = plate_type = province = None
    num_conf = prov_conf = 0.0
    has_dash = False
    if num_matches:
        num_matches.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        _, _, num_conf, number, plate_type, has_dash = num_matches[0]
    if prov_best:
        _, province, prov_conf = prov_best

    if number is None and province is None:
        return None

    confs = [c for c in (num_conf, prov_conf) if c > 0]
    return {
        "number": number,
        "province": province,
        "plate_type": plate_type,
        "has_dash": has_dash,
        "confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "raw": raw,
    }
