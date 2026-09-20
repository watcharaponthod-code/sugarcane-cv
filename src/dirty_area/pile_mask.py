# -*- coding: utf-8 -*-
"""หากองอ้อยในช่องรับ โดย**ไม่**ใช้กรอบสัดส่วนภาพตายตัว (แทน BAY hard-code ใน stalk_mix.py)

หมายเหตุสำคัญเรื่องแนวทาง: แผนเดิมคือ "หาราวเหลือง 2 คู่ → ช่องกลาง = ระหว่างราวคู่ใน"
ใช้กับชุด 30 ใบนี้ไม่ได้ เพราะ generator ไม่ได้รักษามุมกล้องคงที่ — ราวเหลืองที่เห็นชัดมีเส้นเดียว
ในภาพส่วนใหญ่ และตำแหน่งกองเลื่อนไปมา (บางใบกองอยู่ซ้ายของกึ่งกลางภาพ) ดู pile_mask_report.md
จึงเปลี่ยนเป็น: หา "กอง" ตรง ๆ จากคุณสมบัติพื้นผิว/สี เทียบกับพื้นเหล็กในภาพเดียวกัน แล้วใช้ราว
เป็นแค่ข้อมูลประกอบ (รายงานตำแหน่งที่เจอ ไม่ได้เอามาตัดกรอบ)

เกณฑ์ทุกตัวผูกกับสถิติของภาพนั้นเอง (พื้นเหล็กที่วัดได้) ไม่ใช่สัดส่วนภาพ:
  พื้นอ้างอิง = พิกเซล texture ต่ำสุดครึ่งภาพ → ได้ v_floor, s_floor, tex_floor
  กอง        = (สีจัดกว่าพื้น | มืดกว่าพื้นมาก) & มี texture สูงกว่าพื้น
  ตัดทิ้ง     = ก้อนที่แตะขอบบน (สายพาน/โครงเหล็ก/กระบะรถที่ห้อยลงมาจากขอบบน) และขอบดำ fisheye
  → morphology (kernel ผูกกับขนาดก้อนที่เจอ) → blob ใหญ่สุดที่เหลือ

API:
  pile_mask(bgr, debug=None, dark_tex_mult=1.0) -> (mask | None, info)
      mask = bool ndarray ขนาดเท่าภาพ (True = กองอ้อยบนพื้น) หรือ **None เมื่อไม่มีกองบนพื้น**
      info = dict: reason (ทำไมถึงคืน None), fill/green/dark/ypaint, area_pct, truck, n_blobs, v_floor/s_floor/tex_hi

เกณฑ์ "ยอมรับว่าเป็นกอง" — เลือกจากคุณสมบัติของกอง ไม่ใช่ตำแหน่งในเฟรม:
  ปฏิเสธ (1) ypaint > 0.20 และ green < 0.05
      ypaint = สัดส่วนพิกเซลสีเหลืองสดแบบ "สีทาเหล็ก" (H 15-35, S > 110, V > 0.9*v_floor)
      เหตุผล: false positive ที่พบจริงทั้งหมดคือกรงเหล็ก/ราวทาสีเหลืองที่ขอบเฟรม ซึ่งเป็นเหลืองล้วน
      ไม่มีใบเขียวเลย ส่วนกองอ้อยที่ดูเหลือง ๆ จะมีใบเขียวปนเสมอ จึงต้องเข้าเงื่อนไขทั้งสองข้อพร้อมกัน
  ปฏิเสธ (2) fill < 0.58 และ (green + dark) < 0.20
      fill = สัดส่วนพิกเซลแบบอ้อยภายในก้อน, dark = สัดส่วนที่มืดกว่าครึ่งหนึ่งของพื้น (ลำไหม้/เงาในกอง)
      เหตุผล: กองจริงต้องแน่นและต้องมีหลักฐานอ้อยอย่างน้อยหนึ่งอย่าง (ใบเขียวหรือลำดำ)
      ก้อนโครงเหล็ก/เครื่องจักรมีทั้งความหนาแน่นต่ำและไม่มีหลักฐานอ้อยทั้งสองแบบ

  **Hough ถูกทดลองแล้วและไม่ใช้** — วัดความหนาแน่นเส้นตรงยาว (HoughLinesP บน Canny ในกรอบก้อน)
  บนทั้ง 36 ใบ ได้ 5/2/8/5/2 เส้นบนก้อนที่ไม่ใช่กอง เทียบกับ 0-9 เส้นบนกองจริง → แยกไม่ออก จึงตัดทิ้ง

  ค่าตัวเลขทั้งสี่ตั้งจากชุดอ้างอิง 36 ใบนี้ (fit บนชุดเดียวกับที่วัด) ถือเป็น placeholder
  ต้องตั้งใหม่เมื่อมีภาพจริงหลายลาน — ดู pile_mask_report.md

dark_tex_mult — ความเข้มของ texture gate เฉพาะสาขา "ลำไหม้" (สาขาสีไม่เปลี่ยน) วัดบนชุดอ้างอิง 36 ใบ:
  1.0 (ดีฟอลต์) หากองถูก 31 · คืน None ถูก 5/5 · พื้นที่ mask เฉลี่ย 11.5% ของภาพ
  2.0            หากองถูก 31 · คืน None ถูก 4/5 · พื้นที่ mask เฉลี่ย 10.3% ของภาพ
                 ใบที่เสียคือ `mix20_B_night_dust_1.png` — กลับมาถูกตรวจว่ามีกองทั้งที่อ้อยยังอยู่ในกระบะ
  เหตุผลที่ 2.0 ทำ mask แคบลง: ลำไหม้มีลายเส้น ส่วนแถบเงามืดบนพื้นข้างกองเรียบ การขึ้น gate จึงตัดเงาออก
  ดีฟอลต์เป็น 1.0 เพราะ false positive "มีกองทั้งที่ไม่มี" แพงกว่ามาสก์กว้างเกิน 1 จุดในงาน label
  (3.0 ทดสอบแล้วแย่กว่าทั้งสองทาง: ถูก 29 · None 4/5)

รัน: python pile_mask.py [ไฟล์|โฟลเดอร์ ...]  → pile_mask_out/<stem>.jpg + summary.json
     python pile_mask.py --selftest           → สรุป ถูก/None/พลาด บนชุดอ้างอิง 36 ใบ (บรรทัดเดียว)
"""
import sys, json
from pathlib import Path
import numpy as np, cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
OUT = HERE / "pile_mask_out"
BAY_OLD = (0.42, 0.60, 0.30, 0.88)   # กรอบตายตัวเดิมของ stalk_mix.py — ใช้เทียบอย่างเดียว


def _longest_run(mask):
    """ความยาวแถบต่อเนื่องแนวดิ่งที่ยาวที่สุดของแต่ละคอลัมน์"""
    run = np.zeros(mask.shape[1], np.int32); best = run.copy()
    for row in mask:
        run = np.where(row, run + 1, 0)
        np.maximum(best, run, out=best)
    return best


def find_rails(bgr):
    """ตำแหน่ง x ของราวเหลือง (แถบดิ่งยาว สีเหลืองจัดกว่าฉากรอบ) — ข้อมูลประกอบ ไม่ได้ใช้ตัดกรอบ"""
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # "เหลืองจัดกว่าฉาก": S สูงกว่า percentile 85 ของทั้งภาพ และ V สูงกว่าค่ากลาง → กลางคืนที่พื้นติดส้มทั้งภาพก็ยังคัดได้
    s_hi = max(70.0, float(np.percentile(s, 85)))
    yellow = ((hh >= 12) & (hh <= 40) & (s >= s_hi) & (v >= max(40.0, float(np.median(v)))))
    yellow = cv2.dilate(yellow.astype(np.uint8), np.ones((1, max(3, W // 200)), np.uint8))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((max(9, H // 60), 1), np.uint8))
    runs = cv2.blur(_longest_run(yellow.astype(bool)).astype(np.float32).reshape(1, -1),
                    (max(3, W // 150), 1)).ravel()
    thr, sep = 0.45 * H, max(8, W // 20)
    peaks, cand = [], np.where(runs > thr)[0]
    while len(cand):
        i = int(cand[np.argmax(runs[cand])]); peaks.append(i)
        cand = cand[np.abs(cand - i) > sep]
    return sorted(peaks)


def pile_mask(bgr, debug=None, dark_tex_mult=1.0):
    """คืน bool mask ของกองอ้อยบนพื้น (ไม่รวมอ้อยในกระบะรถ); mask ว่างถ้าหาไม่เจอ"""
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), -1, (9, 9))

    # ขอบดำจาก fisheye / แถบราวหน้ากล้อง: มืดสนิทและไม่มีลาย → ตัดออกก่อนคิดสถิติพื้น
    frame_dark = (v < max(12.0, 0.18 * float(np.median(v)))) & (tex < float(np.percentile(tex, 50)))
    valid = ~frame_dark

    # พื้นเหล็กอ้างอิง = ครึ่งที่ texture ต่ำสุดของพื้นที่ใช้ได้
    tex_med = float(np.percentile(tex[valid], 50))
    floor = valid & (tex <= tex_med)
    v_floor, s_floor = float(np.median(v[floor])), float(np.median(s[floor]))
    tex_hi = float(np.percentile(tex[floor], 90))

    # กระบะรถ: ตัวถังสว่าง-เรียบที่ห้อยลงมาจากขอบบนของเฟรม → ตัดทั้งกล่อง (อ้อยในกระบะไม่ใช่กองบนพื้น)
    v_bright = max(1.35 * v_floor, float(np.percentile(v[floor], 98)))   # สว่างกว่า "พื้นที่สว่างที่สุด" จริง ๆ
    bright = (valid & (v > v_bright) & (tex < tex_hi)).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((max(5, H // 60),) * 2, np.uint8))
    nb, labb, sb, _ = cv2.connectedComponentsWithStats(bright, 8)
    truck, ta = None, 0                            # เอาเฉพาะก้อนสว่างที่ใหญ่ที่สุดที่แตะขอบบน = ตัวรถ
    for i in range(1, nb):
        x, y, w, h, a = sb[i]
        # ตัวรถ: แตะขอบบน + ใหญ่พอ + แต่ต้องไม่กว้าง/ใหญ่จนกลายเป็นพื้นทั้งลาน
        if y <= 0.06 * H and 0.004 * H * W < a < 0.15 * H * W and w < 0.5 * W and a > ta:
            truck, ta = (x, 0, x + w, y + h), a
    d = dict(truck=truck)
    if truck:
        valid = valid.copy(); valid[truck[1]:truck[3], truck[0]:truck[2]] = False

    colorful = s > s_floor + 40.0                 # ลำสด/ใบเขียว: อิ่มสีกว่าพื้นเหล็กชัดเจน
    darker = v < 0.55 * v_floor                   # ลำไหม้: มืดกว่าพื้นราวครึ่ง
    cane = (valid & ((colorful & (tex > tex_hi)) | (darker & (tex > dark_tex_mult * tex_hi)))).astype(np.uint8)

    # กอง = บริเวณที่พิกเซลแบบอ้อย "หนาแน่น" ไม่ใช่แค่มี — ราว/ขอบ/เงาเป็นเส้นบาง ความหนาแน่นในหน้าต่างจึงต่ำ
    # หน้าต่างกว้างกว่าลำอ้อยเดี่ยวหลายเท่า (สเกลความยาว ไม่ใช่ตำแหน่ง); เกณฑ์ 0.5 = ครึ่งหน้าต่างเป็นอ้อย
    win = max(9, (min(H, W) // 20) | 1)
    dens = cv2.boxFilter(cane.astype(np.float32), -1, (win, win))
    core = (dens > 0.5).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((win // 2 | 1,) * 2, np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    d.update(n_blobs=int(n - 1), rejected_top=0)
    best, best_a = None, 0
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if y <= 2:                                # แตะขอบบน = สายพาน/โครงเหล็กเหนือช่อง ไม่ใช่กองบนพื้น
            d["rejected_top"] += 1
            continue
        if max(w / max(h, 1), h / max(w, 1)) > 3.5 or a < 0.35 * w * h:   # เรียว/โปร่ง = ราวหรือโครงเหล็ก ไม่ใช่กอง
            d["rejected_shape"] = d.get("rejected_shape", 0) + 1
            continue
        if a > best_a:
            best, best_a = i, a
    if debug is not None:
        d.update(v_floor=round(v_floor, 1), s_floor=round(s_floor, 1), tex_hi=round(tex_hi, 1))
        debug.update(d)
    if best is None:
        d["reason"] = "no blob passed shape/top filter"
        if debug is not None:
            debug.update(d)
        return None, d

    # คืนขอบกองที่ถูกหน้าต่างกินไป: ขยายแกนกองแล้วตัดด้วยพิกเซลอ้อยจริง
    grown = cv2.dilate((lab == best).astype(np.uint8), np.ones((win, win), np.uint8))
    m = cv2.morphologyEx(cv2.bitwise_and(grown, cane), cv2.MORPH_CLOSE, np.ones((win // 2 | 1,) * 2, np.uint8))
    # ตัดแขนบางที่ยื่นไปตามราว/ขอบพื้น แล้วเลือกก้อนใหญ่สุดที่เหลือ
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win // 2 | 1,) * 2))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(m, 8)
    if n2 < 2:
        d["reason"] = "blob vanished after trim"
        if debug is not None:
            debug.update(d)
        return None, d
    out = lab2 == 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))

    # ---- ยอมรับ/ปฏิเสธ ด้วย "คุณสมบัติของกอง" (ไม่ใช่ตำแหน่ง) ----
    hh = hsv[..., 0]
    fill = float(cane[out].mean())                                     # ความหนาแน่นอ้อยในก้อน
    green = float(((hh >= 35) & (hh <= 85) & (s > 60))[out].mean())     # ใบเขียว
    dark = float((v < 0.55 * v_floor)[out].mean())                      # ลำไหม้/ลำในเงากอง
    ypaint = float(((hh >= 15) & (hh <= 35) & (s > 110) & (v > 0.9 * v_floor))[out].mean())  # สีเหลืองทาเหล็ก
    d.update(fill=round(fill, 3), green=round(green, 3), dark=round(dark, 3), ypaint=round(ypaint, 3))

    if ypaint > 0.20 and green < 0.05:
        d["reason"] = f"steel-yellow structure (ypaint {ypaint:.2f}, green {green:.2f})"
        out = None
    elif fill < 0.58 and (green + dark) < 0.20:
        d["reason"] = f"too little cane evidence (fill {fill:.2f}, green+dark {green + dark:.2f})"
        out = None

    if out is None:
        if debug is not None:
            debug.update(d)
        return None, d

    # ---- ตัดชายพื้นเหล็กรอบกอง: ก้อน ∩ พิกเซลอ้อยที่ขยายเล็กน้อย แล้วอุดรูใน ----
    near = cv2.dilate(cane, np.ones((max(3, win // 6) | 1,) * 2, np.uint8))
    out = cv2.bitwise_and(out.astype(np.uint8), near)
    flood = cv2.copyMakeBorder(out, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood, np.zeros((H + 4, W + 4), np.uint8), (0, 0), 1)   # พื้นหลังที่ต่อกับขอบภาพ
    out = out | (1 - flood[1:-1, 1:-1])                                    # ที่เหลือเป็น 0 = รูในกอง → อุด
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
    if n3 < 2:
        d["reason"] = "blob vanished after skirt trim"
        if debug is not None:
            debug.update(d)
        return None, d
    out = lab3 == 1 + int(np.argmax(st3[1:, cv2.CC_STAT_AREA]))
    d["area_pct"] = round(100.0 * float(out.sum()) / (H * W), 2)
    if debug is not None:
        debug.update(d)
    return out, d


# ---------------- runner ----------------
def _overlay(bgr, mask, dbg, name):
    v = bgr.copy(); H, W = v.shape[:2]
    x0, x1, y0, y1 = BAY_OLD
    cv2.rectangle(v, (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H)), (170, 170, 170), 2)
    for p in dbg.get("rails", []):
        cv2.line(v, (p, 0), (p, H), (255, 0, 255), 1)
    if mask.any():
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, cnts, -1, (0, 255, 255), 3)
    cv2.rectangle(v, (0, 0), (W, 34), (0, 0, 0), -1)
    cv2.putText(v, f"{name}   {dbg.get('label','')}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return cv2.resize(v, (W // 2, H // 2))


def main(srcs):
    OUT.mkdir(exist_ok=True)
    paths = []
    for sp in srcs:
        p = Path(sp)
        paths += sorted(q for q in (sorted(p.glob("*")) if p.is_dir() else [p])
                        if q.is_file() and q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    rows = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        dbg = {"rails": find_rails(bgr)}
        m, info = pile_mask(bgr, dbg)
        if m is None:
            m = np.zeros((H, W), bool)
        x0, x1, y0, y1 = BAY_OLD
        old = np.zeros((H, W), bool); old[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
        inter = int((m & old).sum()); a_new, a_old = int(m.sum()), int(old.sum())
        ys, xs = np.where(m)
        r = dict(file=p.name, ok=bool(m.any()), reason=info.get("reason"), rails=dbg["rails"], n_blobs=dbg.get("n_blobs", 0),
                 rejected_top=dbg.get("rejected_top", 0),
                 cx=round(float(xs.mean()) / W, 3) if a_new else None,
                 cy=round(float(ys.mean()) / H, 3) if a_new else None,
                 new_pct=round(100 * a_new / (H * W), 2), old_pct=round(100 * a_old / (H * W), 2),
                 iou=round(inter / max(a_new + a_old - inter, 1), 3),
                 inside_old=round(inter / max(a_new, 1), 3))
        dbg["label"] = f"new {r['new_pct']}% old {r['old_pct']}% IoU {r['iou']} inside_old {r['inside_old']}"
        cv2.imwrite(str(OUT / (p.stem + ".jpg")), _overlay(bgr, m, dbg, p.name), [cv2.IMWRITE_JPEG_QUALITY, 80])
        rows.append(r)
        print(f"{p.name:36s} blobs {r['n_blobs']:3d} cx {str(r['cx']):>5} cy {str(r['cy']):>5} "
              f"new {r['new_pct']:5.2f}% IoU {r['iou']:.2f} inside_old {r['inside_old']:.2f}")
    json.dump(rows, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    bad = [r["file"] for r in rows if not r["ok"]]
    print(f"\n{len(rows)} ภาพ · คืน None (ไม่มีกองบนพื้น) {len(bad)} ใบ" + (": " + ", ".join(bad) if bad else ""))
    return rows


# ---------------- self-test ----------------
NO_PILE = {"mix0_A_night_nodust_1.png", "mix20_B_night_dust_1.png", "reference_real.png",
           "sugarcane_cctv_dumping.webp", "sugarcane_truck_dumping_fisheye.jpg"}
PILE_LEFT = {"three_bay_fisheye_unloading.webp"}          # กองอยู่ช่องซ้าย ไม่ใช่ช่องกลาง


def selftest():
    """รัน 36 ใบอ้างอิงแล้วสรุป ถูก/None/พลาด — `python pile_mask.py --selftest`"""
    gi = ROOT / "gen_images"
    srcs = [gi / "burnt_mixed"] + [gi / f for f in ("reference_real.png", "sugarcane_cctv_dumping.webp",
            "sugarcane_dump_overhead.webp", "three_bay_fisheye_unloading.webp", "sugarcane_truck_dumping_fisheye.jpg")]
    rows = main([str(x) for x in srcs])
    ok = none_ok = none_bad = miss = 0
    for r in rows:
        want_none = r["file"] in NO_PILE
        if not r["ok"]:
            if want_none:
                none_ok += 1
            else:
                none_bad += 1; print(f"  พลาด(คืน None ทั้งที่มีกอง) {r['file']}  reason={r['reason']}")
        elif want_none:
            miss += 1; print(f"  พลาด(ควรเป็น None) {r['file']}  cx={r['cx']}")
        else:
            lo, hi = (0.10, 0.35) if r["file"] in PILE_LEFT else (0.40, 0.62)
            if lo < (r["cx"] or -1) < hi:
                ok += 1
            else:
                miss += 1; print(f"  พลาด(กองผิดที่) {r['file']}  cx={r['cx']}")
    area = [r["new_pct"] for r in rows if r["ok"]]
    print(f"\nSELFTEST: ถูก {ok} · None ถูกต้อง {none_ok}/{len(NO_PILE)} · พลาด {miss + none_bad} · รวม {len(rows)}")
    print(f"พื้นที่ mask เฉลี่ย {sum(area) / max(len(area), 1):.2f}% ของภาพ (กรอบ BAY เดิม 10.4% คงที่)")
    return miss + none_bad == 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    main(sys.argv[1:] or [str(ROOT / "gen_images" / "burnt_mixed")])
