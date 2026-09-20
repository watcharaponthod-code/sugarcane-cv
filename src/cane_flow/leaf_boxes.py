# -*- coding: utf-8 -*-
"""ตีกรอบใบอ้อย/ยอดยาวทีละใบ ด้วย SAM2 + กฎรูปทรง (ไม่เทรน ไม่ใช้ข้อความนำทาง)

  python leaf_boxes.py                 # 12 ภาพมาตรฐาน
  python leaf_boxes.py --src <ไฟล์|โฟลเดอร์> --limit 4

ออก: leaf_box_out/<ชื่อ>_pair.jpg (ต้นฉบับซ้าย | ตีกรอบขวา) + leaf_box_out/summary.json
SAM2 หา instance ทั้งหมด → แปลงเป็นกรอบเอียง (minAreaRect) → กรองด้วยรูปทรงล้วน ๆ ข้างล่างนี้
"""
import argparse, json, time
from pathlib import Path

import cv2, numpy as np, psutil, torch

# ---- เกณฑ์รูปทรง ปรับตรงนี้ ----
AR_MIN = 4.0          # ยาว/กว้าง ของ minAreaRect — ใบและยอดเป็นแท่งยาว
AREA_MIN = 0.0005     # 0.05% ของภาพ
AREA_MAX = 0.03       # 3% ของภาพ — ใหญ่กว่านี้คือกอง ไม่ใช่ใบ
SOLIDITY_MAX = 0.75   # พื้นที่ mask / พื้นที่กรอบเอียง — ใบโค้งงอไม่เต็มกรอบ
GRID_N = 20           # จุด prompt ต่อด้าน (20x20 = 400 จุด/ภาพ)
MAX_SIDE = 800        # ห้ามขยายภาพ ย่อลงถ้าใหญ่กว่านี้
PILE_MIN_IN = 0.70    # --pile-only: กรอบต้องมีพื้นที่อยู่ในกองอย่างน้อยเท่านี้
# --------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
OUT = HERE / "leaf_box_out"
OUT.mkdir(exist_ok=True)


def pick_images():
    """6 ภาพจริง (fresh 3 / cut 3 เป็นตัวควบคุม) + 6 ภาพผู้ใช้ (mix0 3 / mix50 3)"""
    rf = ROOT / "gen_images" / "roboflow_cane"
    g2 = ROOT / "gen_images" / "burnt_mixed_g2"
    out = []
    for cls, n in (("cane_fresh", 3), ("cane_cut", 3)):
        got = []
        for sp in ("train", "valid", "test"):
            got += sorted((rf / sp / cls).glob("*.jpg"))
        step = max(1, len(got) // (n + 1))
        out += [(p, cls) for p in got[::step][:n]]
    for pat, n in (("mix0_*_g2.png", 3), ("mix50_*_g2.png", 3)):
        out += [(p, pat.split("_")[0]) for p in sorted(g2.glob(pat))[:n]]
    return out


def shape_ok(sub, img_area):
    """สอบรูปทรงจาก mask ในกรอบ — คืน (ผ่านไหม, rect, สถิติ)"""
    m = sub.astype(np.uint8)
    a = float(m.sum())
    if not (AREA_MIN * img_area <= a <= AREA_MAX * img_area):
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (w, h) = rect[1]
    if min(w, h) < 1:
        return None
    ar = max(w, h) / min(w, h)
    sol = a / max(w * h, 1)
    if ar < AR_MIN or sol > SOLIDITY_MAX:
        return None
    return rect, dict(area_frac=round(a / img_area, 5), ar=round(ar, 2), solidity=round(sol, 3))


def pile_from_segs(segs, h, w, min_frac=0.05):
    """ทางสำรอง (A): ขอบเขตกอง = รวม instance ของ SAM2 ที่ใหญ่กว่า min_frac ของเฟรม

    ตัดชิ้นที่แตะขอบภาพทั้งสองด้านตรงข้าม (ซ้าย-ขวา หรือ บน-ล่าง) ออก = โครงสร้างโรงงาน ไม่ใช่กอง
    กฎเดียวเท่านั้นตามที่ตกลง ไม่ปรับอย่างอื่น
    """
    area = h * w
    m = np.zeros((h, w), bool)
    n_used = 0
    for sg in segs:
        sh, sw = sg.sub.shape
        if float(sg.sub.sum()) < min_frac * area:
            continue
        spans_x = sg.x0 <= 2 and sg.x0 + sw >= w - 2
        spans_y = sg.y0 <= 2 and sg.y0 + sh >= h - 2
        if spans_x or spans_y:
            continue
        m[sg.y0:sg.y0 + sh, sg.x0:sg.x0 + sw] |= sg.sub
        n_used += 1
    return m if n_used else None


def cv_segments(im):
    """หา instance แบบไม่ใช้โมเดล: แยกส่วนสว่าง/ขอบชัด แล้วตัดเป็นชิ้นด้วย connected components

    ใช้เมื่อ SAM2 โหลดไม่ได้ · คุณภาพต่ำกว่า SAM2 แน่นอน ใบที่ติดกันจะกลายเป็นชิ้นเดียว
    """
    from sam_stalks import Seg
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 120)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    inv = cv2.bitwise_not(e)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    segs = []
    for i in range(1, n):
        x, y, w_, h_, area = stats[i]
        if area < 40 or w_ * h_ > 0.25 * im.shape[0] * im.shape[1]:
            continue
        segs.append(Seg(int(y), int(x), (lab[y:y + h_, x:x + w_] == i)))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--engine", choices=("sam2", "cv"), default="sam2",
                    help="cv = กฎรูปทรงล้วนบน OpenCV (ใช้เมื่อ SAM2 โหลดไม่ได้)")
    ap.add_argument("--max-new", type=int, default=0,
                    help="ทำใหม่ไม่เกิน N ภาพต่อการรันหนึ่งครั้งแล้วออก (กัน OOM kill กลางคัน)")
    ap.add_argument("--restart", action="store_true", help="ไม่สนผลเดิม เริ่มใหม่ทั้งหมด")
    ap.add_argument("--pile-from-sam", action="store_true",
                    help="ถ้า pile_mask ใช้ไม่ได้กับฉากนี้ ให้รวม instance ใหญ่ของ SAM2 เป็นขอบเขตกองแทน")
    ap.add_argument("--pile-only", action="store_true",
                    help="เก็บเฉพาะกรอบที่อยู่ในกองอ้อย >= PILE_MIN_IN (ใช้ pile_mask.py) · ออกที่ pile_only/")
    a = ap.parse_args()

    global OUT
    if a.pile_only:
        OUT = OUT / "pile_only"
        OUT.mkdir(exist_ok=True)
    if a.src:
        p = Path(a.src)
        imgs = [(q, "?") for q in (sorted(p.glob("*.jpg")) + sorted(p.glob("*.png")))] if p.is_dir() else [(p, "?")]
    else:
        imgs = pick_images()
    if a.limit:
        imgs = imgs[: a.limit]

    engine = a.engine
    if engine == "sam2":
        try:
            from transformers import Sam2Model, Sam2Processor
            from sam_stalks import MODEL_ID, segments
        except ImportError as e:
            raise SystemExit(
                "โหลด SAM2 ไม่ได้: " + str(e)[:120]
                + " | สาเหตุ: transformers ต้องใช้ scipy และ DLL ของ scipy ถูก Application Control ของ Windows บล็อก"
                + " | ทางออกต้องให้ผู้ใช้ตัดสิน (ปลดบล็อกใน policy หรือรันบนเครื่องอื่น)"
                + " | ระหว่างนี้ใช้ --engine cv ได้ (กฎรูปทรงล้วน ไม่ต้องใช้ SAM2)")

    dev = torch.device("cpu" if a.cpu or engine == "cv" or not torch.cuda.is_available() else "cuda")
    if engine == "sam2":
        proc = Sam2Processor.from_pretrained(MODEL_ID)
        model = Sam2Model.from_pretrained(MODEL_ID).to(dev).eval()
    else:
        MODEL_ID = "opencv-rule (ไม่มีโมเดล)"
    me = psutil.Process()
    rss0 = me.memory_info().rss / 1e6
    print(f"SAM2 {MODEL_ID} · {dev.type} · ภาพ {len(imgs)} ใบ · RSS หลังโหลดโมเดล {rss0:.0f} MB")

    # resume: อ่านผลเดิม ข้ามภาพที่ทำแล้ว เขียนลงดิสก์ทุกภาพ — โดน OOM kill กี่รอบก็เดินต่อ
    sumf = OUT / "summary.json"
    done = {}
    if sumf.exists() and not a.restart:
        try:
            done = {r["file"]: r for r in json.loads(sumf.read_text(encoding="utf-8"))["images"]}
        except Exception:
            done = {}
    if done:
        print(f"  ทำไปแล้ว {len(done)} ภาพ ข้ามให้")

    def save(results):
        by_tag_ = {}
        for r in results:
            by_tag_.setdefault(r["tag"], []).append(r["n_boxes"])
        sumf.write_text(json.dumps({
            "model": MODEL_ID, "device": dev.type, "engine": engine,
            "rules": dict(AR_MIN=AR_MIN, AREA_MIN=AREA_MIN, AREA_MAX=AREA_MAX,
                          SOLIDITY_MAX=SOLIDITY_MAX, GRID_N=GRID_N, MAX_SIDE=MAX_SIDE),
            "rss_after_model_mb": round(rss0),
            "peak_rss_mb": max([r["rss_mb"] for r in results] or [0]),
            "peak_gpu_mb": max([r["gpu_mb"] for r in results] or [0]),
            "mean_sec_per_image": round(float(np.mean([r["sec"] for r in results])), 1) if results else None,
            "by_tag_mean_boxes": {t: round(float(np.mean(v)), 1) for t, v in by_tag_.items()},
            "images": results,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    results = list(done.values())
    n_new = 0
    for path, tag in imgs:
        if path.name in done:
            continue
        if a.max_new and n_new >= a.max_new:
            print(f"  ถึงเพดาน --max-new {a.max_new} แล้ว ออกก่อน (รันซ้ำเพื่อทำต่อ)")
            break
        n_new += 1
        im = cv2.imread(str(path))
        if im is None:
            print(f"  ! อ่านไม่ได้ {path.name}")
            continue
        h, w = im.shape[:2]
        if max(h, w) > MAX_SIDE:                       # ย่อเท่านั้น ไม่ขยาย
            s = MAX_SIDE / max(h, w)
            im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = im.shape[:2]
        img_area = float(h * w)
        whole = np.ones((h, w), bool)                  # ไม่จำกัดขอบเขตกอง ปล่อยให้กฎรูปทรงคัดเอง
        gx = np.linspace(0, w - 1, GRID_N).astype(int)
        gy = np.linspace(0, h - 1, GRID_N).astype(int)
        pts = [[int(x), int(y)] for y in gy for x in gx]

        t0 = time.time()
        segs = (segments(model, proc, cv2.cvtColor(im, cv2.COLOR_BGR2RGB), pts, whole, dev)
                if engine == "sam2" else cv_segments(im))
        pile = None
        if a.pile_only:
            try:
                from pile_mask import pile_mask
                pile, _ = pile_mask(im)
            except Exception as err:
                print(f"    (pile_mask ล้ม: {str(err)[:60]})")
                pile = None
            # pile_mask.py เขียนไว้สำหรับฉากลานเท (ราง+มุมบน) ใช้กับ CCTV กระบะเต็มเฟรมไม่ได้ คืน None
            # ทางสำรอง (A): รวม instance ใหญ่ของ SAM2 เป็นขอบเขตกอง — ไม่ได้เขียนตัวหากองใหม่
            if pile is None and a.pile_from_sam:
                pile = pile_from_segs(segs, h, w)
                if pile is not None:
                    ov = im.copy()
                    ov[pile] = (0.45 * np.array([0, 255, 0]) + 0.55 * ov[pile]).astype(np.uint8)
                    cv2.imwrite(str(OUT / f"pile_bounds_{path.stem[:34]}.jpg"),
                                np.hstack([im, ov]), [cv2.IMWRITE_JPEG_QUALITY, 88])
            if pile is None:
                print(f"    (หากองไม่เจอ — ข้ามภาพนี้ ไม่เดา)")
                continue

        boxes = []
        n_out_pile = 0
        for sg in segs:
            r = shape_ok(sg.sub, img_area)
            if r is None:
                continue
            rect, st = r
            (cx, cy), (bw, bh), ang = rect
            rect = ((cx + sg.x0, cy + sg.y0), (bw, bh), ang)
            if pile is not None:
                bm = np.zeros((h, w), np.uint8)
                cv2.fillPoly(bm, [cv2.boxPoints(rect).astype(np.int32)], 1)
                inside = float((bm & pile.astype(np.uint8)).sum()) / max(float(bm.sum()), 1)
                if inside < PILE_MIN_IN:
                    n_out_pile += 1
                    continue
                st = dict(st, in_pile=round(inside, 2))
            boxes.append((rect, st))
        dt = time.time() - t0

        right = im.copy()
        for rect, _ in boxes:
            pts_box = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(right, [pts_box], True, (0, 255, 255), 1, cv2.LINE_AA)
        txt = f"{len(boxes)} boxes"
        cv2.putText(right, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(right, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
        pair = np.hstack([im, np.full((h, 6, 3), 255, np.uint8), right])
        dst = OUT / f"{path.stem[:40]}_pair.jpg"
        cv2.imwrite(str(dst), pair, [cv2.IMWRITE_JPEG_QUALITY, 88])

        rss = me.memory_info().rss / 1e6
        gpu = torch.cuda.max_memory_allocated() / 1e6 if dev.type == "cuda" else 0.0
        results.append(dict(file=path.name, tag=tag, size=[w, h], n_instances=len(segs), n_boxes=len(boxes),
                            n_dropped_outside_pile=n_out_pile, pile_only=bool(a.pile_only),
                            sec=round(dt, 1), rss_mb=round(rss), gpu_mb=round(gpu),
                            boxes=[st for _, st in boxes], out=dst.name))
        print(f"  {path.name[:42]:42s} [{tag:10s}] instance {len(segs):3d} -> กรอบ {len(boxes):3d} · {dt:5.1f}s · RSS {rss:.0f} MB"
              + (f" · GPU {gpu:.0f} MB" if gpu else "")
              + (f" · ตัดนอกกอง {n_out_pile}" if a.pile_only else ""))
        save(results)      # เขียนทุกภาพ ไม่รอจบ

    by_tag = {}
    for r in results:
        by_tag.setdefault(r["tag"], []).append(r["n_boxes"])
    print("\n=== กรอบเฉลี่ยต่อกลุ่ม ===")
    for t, v in by_tag.items():
        print(f"  {t:10s} n={len(v)} · กรอบเฉลี่ย {np.mean(v):.1f} · ช่วง {min(v)}-{max(v)}")

    save(results)
    print(f"\nเขียน {OUT/'summary.json'} · ภาพคู่ {len(results)}/{len(imgs)} ใบใน {OUT}")


if __name__ == "__main__":
    main()
