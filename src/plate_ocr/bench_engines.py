#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""เทียบ OCR engine ของ thai-lpr: template (CV ล้วน) vs easyocr vs tesseract

    python bench_engines.py                      # ชุดทดสอบสังเคราะห์ (สร้างเอง)
    python bench_engines.py --engines template   # เฉพาะบางตัว
    python bench_engines.py --folder photos --truth truth.csv
        truth.csv = ชื่อไฟล์,เลขทะเบียน  (หนึ่งบรรทัดต่อภาพ)

ชุดสังเคราะห์จงใจใส่ฟอนต์ที่ "ไม่ได้อยู่ใน template" + เบลอ + เอียง + มืด/สว่าง
+ JPEG คุณภาพต่ำ + ป้ายเล็ก เพื่อไม่ให้ผลเข้าข้าง template reader
"""
import argparse
import csv
import os
import re
import statistics
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lpr import ThaiLPR

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_fixtures", "bench")

# ฟอนต์ที่ "ไม่ใช่" ตัวที่ TemplateReader ใช้สร้าง template (arialbd/tahomabd)
FONTS = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/tahomabd.ttf",     # ใส่ไว้ 1 ตัวเป็นกรณีฟอนต์ตรง
]
PLATES = ["83-6237", "70-1234", "12-3456", "99-0007", "45-6789", "60-8812"]


def _font(path, size):
    return ImageFont.truetype(path, size)


def make_scene(number, font_path, plate_w, degrade):
    """สร้างภาพรถ + ป้ายทะเบียน แล้วทำให้ภาพแย่ลงตาม degrade"""
    W, H = 900, 700
    img = np.zeros((H, W, 3), np.uint8)
    img[:H // 2] = (200, 180, 150)
    img[H // 2:] = (110, 120, 130)
    cv2.rectangle(img, (150, 140), (750, 580), (228, 228, 233), -1)
    cv2.rectangle(img, (210, 175), (690, 330), (90, 90, 95), -1)
    for i in range(0, W, 37):                       # ลายรบกวนบนตัวรถ
        cv2.line(img, (i, 140), (i + 20, 580), (215, 215, 220), 1)

    ph = int(plate_w * 0.46)
    plate = Image.new("RGB", (plate_w, ph), (250, 250, 248))
    d = ImageDraw.Draw(plate)
    d.rectangle([2, 2, plate_w - 3, ph - 3], outline=(25, 25, 25), width=max(2, plate_w // 90))
    fsize = int(ph * 0.46)
    f = _font(font_path, fsize)
    tw = d.textbbox((0, 0), number, font=f)[2]
    d.text(((plate_w - tw) / 2, ph * 0.06), number, font=f, fill=(12, 12, 12))
    try:
        d.text((plate_w * 0.34, ph * 0.62), "ชัยภูมิ", font=_font(font_path, int(ph * 0.2)), fill=(12, 12, 12))
    except Exception:
        pass
    p = cv2.cvtColor(np.array(plate), cv2.COLOR_RGB2BGR)

    if "rot" in degrade:
        M = cv2.getRotationMatrix2D((plate_w / 2, ph / 2), degrade["rot"], 1.0)
        p = cv2.warpAffine(p, M, (plate_w, ph), borderValue=(250, 250, 248))

    y0, x0 = 430, (W - plate_w) // 2
    img[y0:y0 + ph, x0:x0 + plate_w] = p

    if degrade.get("blur"):
        k = degrade["blur"] * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    if degrade.get("gain"):
        img = np.clip(img.astype(np.float32) * degrade["gain"], 0, 255).astype(np.uint8)
    if degrade.get("noise"):
        img = np.clip(img.astype(np.int16) + np.random.default_rng(7).normal(
            0, degrade["noise"], img.shape), 0, 255).astype(np.uint8)
    if degrade.get("jpeg"):
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, degrade["jpeg"]])[1], 1)
    return img


def build_cases():
    """คืน [(ชื่อเคส, ภาพ, เลขจริง)] — ไล่จากง่ายไปยาก"""
    rng = np.random.default_rng(20260820)
    cases = []
    levels = [
        ("clean_260", 260, {}),
        ("clean_180", 180, {}),
        ("small_120", 120, {}),
        ("blur_260", 260, {"blur": 1}),
        ("blur_180", 180, {"blur": 2}),
        ("rot5_220", 220, {"rot": 5}),
        ("dark_220", 220, {"gain": 0.55}),
        ("bright_220", 220, {"gain": 1.45}),
        ("noise_220", 220, {"noise": 12}),
        ("jpeg40_180", 180, {"jpeg": 40}),
    ]
    fonts = [f for f in FONTS if os.path.exists(f)]
    if not fonts:
        sys.exit("ไม่พบฟอนต์สำหรับสร้างชุดทดสอบ")
    for i, (name, pw, deg) in enumerate(levels):
        for j, plate in enumerate(PLATES):
            font = fonts[(i + j) % len(fonts)]
            cases.append((f"{name}_{plate}", make_scene(plate, font, pw, deg), plate))
    return cases


def norm(s):
    return re.sub(r"[^0-9]", "", s or "")


def run_engine(spec, cases, save=False):
    """spec = "engine" หรือ "engine@detector" เช่น fast@onnx"""
    engine, _, detector = spec.partition("@")
    detector = detector or "opencv"
    try:
        lpr = ThaiLPR(engine=engine, detector=detector)
    except Exception as e:
        return {"engine": spec, "error": str(e)}
    lpr.process_image(np.full((160, 360, 3), 220, np.uint8))     # warmup

    ok = found = 0
    times = []
    misses = []
    for name, img, truth in cases:
        t0 = time.perf_counter()
        res = lpr.process_image(img)
        times.append((time.perf_counter() - t0) * 1000)
        got = res[0]["number"] if res else None
        if got:
            found += 1
        if norm(got) == norm(truth):
            ok += 1
        else:
            misses.append(f"{name}: อ่านได้ {got or '-'}")
        if save:
            os.makedirs(OUT_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(OUT_DIR, f"{name}.jpg"), img)
    return {
        "engine": spec, "total": len(cases), "correct": ok, "found": found,
        "median_ms": round(statistics.median(times), 1),
        "mean_ms": round(statistics.mean(times), 1),
        "misses": misses,
    }


def run_strict(cases, scored, args):
    """วัดโหมดเข้มงวด: ตัวเลขที่สำคัญคือ 'ผิด' ต้องเป็น 0"""
    from strict_reader import StrictPlateReader, digits
    r = StrictPlateReader(min_conf=args.min_conf, auto_conf=args.auto_conf,
                          cross=args.cross or None, min_frames=1)
    r.read([np.full((160, 360, 3), 220, np.uint8)])        # warmup
    r.cross_reader()

    acc_ok = acc_bad = rejected = 0
    times = []
    bad, why = [], {}
    for name, img, truth in cases:
        t0 = time.perf_counter()
        d = r.read([img])
        times.append((time.perf_counter() - t0) * 1000)
        if d["accepted"]:
            if digits(d["number"]) == digits(truth):
                acc_ok += 1
            else:
                acc_bad += 1
                bad.append(f"{name}: ตอบ {d['number']} ควรเป็น {truth}")
        else:
            rejected += 1
            why[d["reason_th"]] = why.get(d["reason_th"], 0) + 1

    n = len(scored)
    print(f"โหมดเข้มงวด (min_conf={args.min_conf}, cross-check={r.cross_name})")
    print(f"  ตอบออกมา   {acc_ok + acc_bad}/{n}  ({(acc_ok + acc_bad) / n * 100:.0f}%)")
    print(f"  ในนั้นถูก   {acc_ok}")
    print(f"  ในนั้นผิด   {acc_bad}      <-- ตัวเลขนี้ต้องเป็น 0")
    print(f"  ส่งคนตรวจ  {rejected}")
    print(f"  มัธยฐาน    {statistics.median(times):.0f} ms/ภาพ")
    for k, v in sorted(why.items(), key=lambda kv: -kv[1]):
        print(f"     เหตุที่ไม่ตอบ: {k} x{v}")
    for b in bad:
        print(f"     ผิด: {b}")
    return []


def load_folder(folder, truth_csv):
    truth = {}
    if truth_csv:
        with open(truth_csv, encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    truth[row[0].strip()] = row[1].strip()
    cases = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        img = cv2.imread(os.path.join(folder, fn))
        if img is not None:
            cases.append((fn, img, truth.get(fn, "")))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="*", default=["fast@onnx", "fast", "template"],
                    help='รูปแบบ engine หรือ engine@detector เช่น fast@onnx easyocr template')
    ap.add_argument("--folder", help="ใช้ภาพจริงจากโฟลเดอร์แทนชุดสังเคราะห์")
    ap.add_argument("--truth", help="CSV: ชื่อไฟล์,เลขทะเบียน")
    ap.add_argument("--save", action="store_true", help="บันทึกภาพชุดทดสอบไว้ดู")
    ap.add_argument("--misses", action="store_true", help="พิมพ์รายการที่อ่านผิด")
    ap.add_argument("--strict", action="store_true",
                    help="วัดโหมดเข้มงวด (strict_reader) — สนใจว่า 'ที่ตอบมา ผิดกี่อัน'")
    ap.add_argument("--min-conf", type=float, default=0.90)
    ap.add_argument("--auto-conf", type=float, default=0.98)
    ap.add_argument("--cross", default="easyocr", help="engine ตัวตรวจทาน (ว่าง = ปิด)")
    args = ap.parse_args()

    cases = load_folder(args.folder, args.truth) if args.folder else build_cases()
    scored = [c for c in cases if c[2]]
    print(f"ชุดทดสอบ {len(cases)} ภาพ (มีเฉลย {len(scored)})\n")

    if args.strict:
        return run_strict(cases, scored, args)

    rows = []
    for e in args.engines:
        r = run_engine(e, cases, save=args.save)
        rows.append(r)
        if "error" in r:
            print(f"{e:14s} ใช้ไม่ได้: {r['error']}")
            continue
        acc = r["correct"] / max(1, len(scored)) * 100
        print(f"{e:14s} ถูก {r['correct']}/{len(scored)} ({acc:.0f}%)  "
              f"เจอป้าย {r['found']}/{r['total']}  "
              f"มัธยฐาน {r['median_ms']} ms  เฉลี่ย {r['mean_ms']} ms")
        if args.misses:
            for m in r["misses"][:20]:
                print(f"             {m}")
    return rows


if __name__ == "__main__":
    main()
