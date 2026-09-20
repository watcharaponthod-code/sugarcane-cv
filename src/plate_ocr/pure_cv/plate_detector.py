# -*- coding: utf-8 -*-
"""หาตำแหน่งป้ายทะเบียนด้วย OpenCV ล้วน (ไม่ใช้ Machine Learning)

ใช้ 2 วิธีร่วมกันแล้วรวมผล (คล้ายแนว multi-cue ของงานตรวจฝุ่น):
  วิธีที่ 1  morphology  : blackhat + sobel — จับ "โซนตัวอักษรเข้มถี่ ๆ บนพื้นสว่าง"
  วิธีที่ 2  edge+contour: canny + สี่เหลี่ยม — จับ "กรอบป้าย" ที่ขอบชัด
กรอบที่ทั้งสองวิธีเห็นตรงกันจะได้คะแนนสูงกว่า
"""
import os
import math

import cv2
import numpy as np

from . import config as C


# ---------------------------------------------------------------- helpers
def _resize_keep(img, width):
    """ปรับภาพให้กว้าง ~width: ภาพใหญ่ย่อลง / ภาพเล็กขยายขึ้น
    (ภาพจากกล้องความละเอียดต่ำ ป้ายจะเล็กเกินจนหาไม่เจอ ต้องขยายก่อน)"""
    h, w = img.shape[:2]
    if w == width:
        return img, 1.0
    scale = width / float(w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (width, int(round(h * scale))),
                         interpolation=interp)
    return resized, scale


def _rect_ok(w, h, img_area):
    """เช็คว่ากรอบนี้ 'หน้าตาเหมือนป้ายทะเบียน' หรือไม่"""
    if h <= 0 or w <= 0:
        return False
    ar = w / float(h)
    area = w * h
    return (C.ASPECT_MIN <= ar <= C.ASPECT_MAX
            and C.AREA_MIN_RATIO * img_area <= area <= C.AREA_MAX_RATIO * img_area
            and w >= C.PLATE_MIN_WIDTH)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _overlap(a, b):
    """ค่าซ้อนทับ = max(IoU, intersection/พื้นที่กรอบเล็ก)
    ครอบคลุมกรณีกรอบเล็ก (เช่นแถบจังหวัด) อยู่ 'ข้างใน' กรอบป้ายใหญ่"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    small = min(aw * ah, bw * bh)
    return max(_iou(a, b), inter / small)


# ---------------------------------------------------------------- วิธีที่ 1
def _candidates_morph(gray, debug=None):
    """morphology: เน้นตัวอักษรเข้มบนพื้นสว่าง (ป้ายขาว/เหลือง)"""
    img_area = gray.shape[0] * gray.shape[1]
    rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, C.BLACKHAT_KERNEL)

    # blackhat = ดึงรายละเอียดเข้มบนพื้นสว่าง (ตัวอักษรบนป้าย)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_k)

    # mask บริเวณสว่าง (ตัวแผ่นป้าย)
    sq_k = cv2.getStructuringElement(cv2.MORPH_RECT, C.LIGHT_KERNEL)
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, sq_k)
    light = cv2.threshold(light, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # gradient แกน X — ตัวอักษรมีขอบแนวตั้งถี่
    gx = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    gx = np.absolute(gx)
    mn, mx = float(gx.min()), float(gx.max())
    if mx - mn < 1e-6:
        return []
    gx = (255 * (gx - mn) / (mx - mn)).astype("uint8")

    gx = cv2.GaussianBlur(gx, (5, 5), 0)
    gx = cv2.morphologyEx(gx, cv2.MORPH_CLOSE, rect_k)
    th = cv2.threshold(gx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    th = cv2.erode(th, None, iterations=2)
    th = cv2.dilate(th, None, iterations=2)
    th = cv2.bitwise_and(th, th, mask=light)
    th = cv2.dilate(th, None, iterations=2)
    th = cv2.erode(th, None, iterations=1)

    if debug is not None:
        cv2.imwrite(os.path.join(debug, "10_blackhat.jpg"), blackhat)
        cv2.imwrite(os.path.join(debug, "11_morph_thresh.jpg"), th)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if _rect_ok(w, h, img_area):
            out.append((x, y, w, h))
    return out


# ---------------------------------------------------------------- วิธีที่ 2
def _candidates_contour(gray, debug=None):
    """edge + contour: หากรอบสี่เหลี่ยมที่สัดส่วนเหมือนป้าย"""
    img_area = gray.shape[0] * gray.shape[1]
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    edges = cv2.dilate(edges, None, iterations=1)

    if debug is not None:
        cv2.imwrite(os.path.join(debug, "20_edges.jpg"), edges)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    out = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if _rect_ok(w, h, img_area):
                out.append((x, y, w, h))
    return out


# ---------------------------------------------------------------- วิธีที่ 3
def _candidates_chars(gray, debug=None):
    """จับกลุ่มตัวอักษร: หา blob ขนาดเท่าตัวอักษรที่เรียงแถวเดียวกัน >= 3 ตัว
    ทนต่อกรณีขอบป้ายกลืนกับตัวรถ (เช่นป้ายขาวบนกันชนขาว) ได้ดีกว่าสองวิธีแรก"""
    img_area = gray.shape[0] * gray.shape[1]
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 15)
    if debug is not None:
        cv2.imwrite(os.path.join(debug, "40_chars_thresh.jpg"), th)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 10 <= h <= 90 and 2 <= w <= 70 and 0.08 <= w / float(h) <= 1.3 \
                and cv2.contourArea(c) >= 20:
            blobs.append((x, y, w, h))

    # จัดกลุ่มแบบ greedy: อยู่แถวเดียวกัน = กึ่งกลางแนวตั้งใกล้กัน,
    # สูงพอๆ กัน, และช่องว่างแนวนอนไม่เกิน ~2.5 เท่าของความสูง
    blobs.sort(key=lambda b: b[0])
    used = [False] * len(blobs)
    out = []
    for i, b in enumerate(blobs):
        if used[i]:
            continue
        cluster = [b]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, b2 in enumerate(blobs):
                if used[j]:
                    continue
                for b1 in cluster:
                    cy1, cy2 = b1[1] + b1[3] / 2, b2[1] + b2[3] / 2
                    hmax = max(b1[3], b2[3])
                    gap = max(b1[0], b2[0]) - min(b1[0] + b1[2],
                                                  b2[0] + b2[2])
                    if (abs(cy1 - cy2) < 0.6 * hmax
                            and max(b1[3], b2[3]) / max(1, min(b1[3], b2[3])) < 1.8
                            and gap < 2.5 * hmax):
                        cluster.append(b2)
                        used[j] = True
                        changed = True
                        break
        if len(cluster) < 3:
            continue
        xs = [b[0] for b in cluster]
        ys = [b[1] for b in cluster]
        xe = [b[0] + b[2] for b in cluster]
        ye = [b[1] + b[3] for b in cluster]
        x, y = min(xs), min(ys)
        w, h = max(xe) - x, max(ye) - y
        # ขยายกรอบเผื่อบรรทัดจังหวัดด้านล่างและขอบป้าย
        mx, my = int(w * 0.10), int(h * 0.60)
        x, y = max(0, x - mx), max(0, y - my)
        w, h = w + 2 * mx, h + 2 * my
        ar = w / float(h)
        area = w * h
        if (1.2 <= ar <= 8.0 and w >= C.PLATE_MIN_WIDTH * 0.7
                and C.AREA_MIN_RATIO * img_area <= area
                <= C.AREA_MAX_RATIO * img_area):
            out.append((x, y, w, h))
    return out


# ---------------------------------------------------------------- รวมผล
def _aspect_prior(box):
    """คะแนนความ 'ทรงเหมือนป้าย' — ป้ายไทยมาตรฐานอัตราส่วน ~2.3
    ยิ่งใกล้ยิ่งได้คะแนนสูง (0 = พอดีเป๊ะ, ติดลบมาก = ทรงเพี้ยน)"""
    ar = box[2] / float(max(1, box[3]))
    return -abs(math.log(ar / 2.3))


def _merge_candidates(*box_lists):
    """รวมกรอบจากหลายวิธีโดย 'ไม่แทนที่กรอบกันเอง':
    - dedupe เฉพาะกรอบที่แทบเหมือนกัน (IoU > 0.6)
    - กรอบซ้อนใน (ป้ายเล็กในพื้นหลังใหญ่) เก็บไว้ทั้งคู่ ให้ OCR ตัดสิน
    - คะแนน = จำนวน 'วิธี' ที่เห็นกรอบใกล้เคียงตำแหน่งนี้ (IoU > 0.4)"""
    kept = []          # กรอบที่เก็บ
    for boxes in box_lists:
        for box in boxes:
            if not any(_iou(box, k) > 0.6 for k in kept):
                kept.append(box)

    scored = []
    for box in kept:
        score = sum(1 for boxes in box_lists
                    if any(_iou(box, b) > 0.4 for b in boxes))
        scored.append([box, score])

    # เรียง: วิธีเห็นตรงกันมากก่อน แล้วค่อยทรงเหมือนป้าย
    scored.sort(key=lambda m: (m[1], _aspect_prior(m[0])), reverse=True)
    return scored[:C.MAX_CANDIDATES]


# ---------------------------------------------------------------- deskew
def deskew(plate_gray):
    """ปรับป้ายที่เอียงเล็กน้อย (±20°) ให้ตรง เพื่อให้ OCR อ่านง่ายขึ้น"""
    th = cv2.threshold(plate_gray, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(th > 0))
    if len(coords) < 50:
        return plate_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.5 or abs(angle) > 20:   # ตรงอยู่แล้ว หรือเอียงผิดปกติ
        return plate_gray
    h, w = plate_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(plate_gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- public API
def detect_plates(bgr, debug_dir=None):
    """หาตำแหน่งป้ายทะเบียนในภาพ (ตรวจหลายสเกลแล้วรวมผล)

    ขนาดภาพมีผลกับ morphology มาก จึงรันหลายสเกล:
      - สเกลธรรมชาติ (ไม่เกิน 1600px)
      - 1280px มาตรฐาน (ถ้าต่างจากข้อแรกพอสมควร)
      - 2 เท่า สำหรับภาพเล็ก (< 800px) ที่ป้ายเล็กจนหาไม่เจอ

    Returns
    -------
    list ของ dict:
        bbox      : (x, y, w, h) พิกัดบน "ภาพต้นฉบับ"
        plate_bgr : ภาพป้ายที่ crop จากภาพต้นฉบับ (เผื่อขอบแล้ว)
        score     : จำนวนวิธี/สเกลที่เห็นกรอบใกล้ตำแหน่งนี้ (มาก = น่าเชื่อ)
    """
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    H, W = bgr.shape[:2]
    widths = [min(W, 1600)]
    if abs(widths[0] - C.PROC_WIDTH) / float(C.PROC_WIDTH) > 0.15:
        widths.append(C.PROC_WIDTH)
    if W < 800:
        widths.append(min(W * 2, C.PROC_WIDTH))
    widths = sorted(set(widths))

    box_lists = []          # ทุก (วิธี x สเกล) map กลับพิกัดต้นฉบับแล้ว
    for tw in widths:
        small, scale = _resize_keep(bgr, tw)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        dbg = debug_dir if (debug_dir and tw == widths[0]) else None
        for finder in (_candidates_morph, _candidates_contour,
                       _candidates_chars):
            boxes = finder(gray, dbg)
            box_lists.append([
                (int(x / scale), int(y / scale),
                 int(w / scale), int(h / scale))
                for x, y, w, h in boxes])

    merged = _merge_candidates(*box_lists)

    results = []
    for (x, y, w, h), score in merged:
        # เผื่อขอบ
        mx = int(w * C.CROP_MARGIN)
        my = int(h * C.CROP_MARGIN)
        X = max(0, x - mx)
        Y = max(0, y - my)
        X2 = min(W, x + w + mx)
        Y2 = min(H, y + h + my)
        if X2 - X < 10 or Y2 - Y < 5:
            continue
        crop = bgr[Y:Y2, X:X2].copy()
        results.append({
            "bbox": (X, Y, X2 - X, Y2 - Y),
            "plate_bgr": crop,
            "score": score,
        })

    if debug_dir:
        vis = bgr.copy()
        for i, r in enumerate(results):
            x, y, w, h = r["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.imwrite(os.path.join(debug_dir, f"30_plate_{i}.jpg"),
                        r["plate_bgr"])
        cv2.imwrite(os.path.join(debug_dir, "31_candidates.jpg"), vis)

    return results
