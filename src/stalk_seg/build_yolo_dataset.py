# -*- coding: utf-8 -*-
"""A-5 แขนง 1 ขั้นที่ 1: สร้างชุดข้อมูล instance-segmentation สำหรับ YOLOv8-seg จากภาพ g1
ชิ้นต่อลำมาจาก SAM2 (ตรรกะยกมาจาก sam_stalks.py ของ db2b2-25 ไม่แก้ไฟล์นั้น)
คลาสต่อชิ้นตัดสินด้วย label พิกเซลของเรา (gen_images/burnt_mixed/labels/*.png):
  ชิ้นที่ >=70% ของพิกเซลที่ตัดสินได้เป็นคลาส 1 -> fresh_stalk (0) · >=70% เป็นคลาส 2 -> burnt_stalk (1) · อื่น -> ทิ้ง
ภาพที่เขียนออกคือ "กรอบ CROP เดิม" (x .36-.66, y .24-.92) ที่ v2 ใช้ เพื่อให้ลำใหญ่พอที่ imgsz 1024
hold-out 6 ใบเดิมไม่เข้าชุดเทรน (เขียนไว้ใน val เพื่อดูตัวเลข แต่ห้ามใช้เลือก hyperparameter)
รัน: python build_yolo_dataset.py [--src DIR] [--out DIR]"""
import argparse, json
from pathlib import Path
import numpy as np, cv2, torch
from transformers import Sam2Model, Sam2Processor
from pile_mask import pile_mask

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SAM_ID = "facebook/sam2-hiera-small"
CROP = (0.36, 0.66, 0.24, 0.92)
GRID_N, NMS_IOU, BATCH = 22, 0.70, 32
PURITY = 0.70
HOLD = {"mix0_A_night_nodust_1", "mix10_B_sun_dust_1", "mix20_A_cloud_dust_1",
        "mix30_B_night_nodust_1", "mix50_B_cloud_nodust_1", "mix100_B_cloud_nodust_1"}

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(ROOT / "gen_images" / "burnt_mixed"))
ap.add_argument("--labels", default=str(ROOT / "gen_images" / "burnt_mixed" / "labels"))
ap.add_argument("--out", default=str(HERE / "yolo_ds"))
ap.add_argument("--glob", default="mix[0-9]*_*.png")
a = ap.parse_args()
SRC, LB, OUT = Path(a.src), Path(a.labels), Path(a.out)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def crop_box(f):
    H, W = f.shape[:2]
    return int(CROP[0] * W), int(CROP[1] * W), int(CROP[2] * H), int(CROP[3] * H)


def grid_points(mask, n=GRID_N):
    ys, xs = np.where(mask)
    gx = np.linspace(xs.min(), xs.max(), n).astype(int)
    gy = np.linspace(ys.min(), ys.max(), n).astype(int)
    return [[int(x), int(y)] for y in gy for x in gx if mask[y, x]]


def segments(model, proc, rgb, pts, pile):
    pile_area = float(pile.sum()); keep = []
    for i in range(0, len(pts), BATCH):
        chunk = pts[i:i + BATCH]
        inp = proc(images=rgb, input_points=[[[p] for p in chunk]],
                   input_labels=[[[1]] * len(chunk)], return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp, multimask_output=True)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
        scores = out.iou_scores.cpu()[0]
        for o in range(masks.shape[0]):
            best, best_s = None, -1.0
            for c in range(masks.shape[1]):
                m = masks[o, c].numpy() > 0; ar = float(m.sum())
                if ar < 0.0005 * pile_area or ar > 0.45 * pile_area: continue
                if float((m & pile).sum()) / max(ar, 1) < 0.80: continue
                s = float(scores[o, c])
                if s > best_s: best, best_s = m, s
            if best is not None: keep.append((best_s, best))
    keep.sort(key=lambda t: -t[0])
    out_masks = []
    for _, m in keep:
        ar = float(m.sum())
        if all(float((m & k).sum()) / min(ar, float(k.sum())) < NMS_IOU for k in out_masks):
            out_masks.append(m)
    return out_masks


def poly_lines(m, x0, y0, w, h):
    """แปลง mask เป็นบรรทัด YOLO-seg (พิกัดสัมพัทธ์ในกรอบ CROP) — เอา contour ใหญ่สุด"""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 40: return None
    c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).astype(np.float32)
    if len(c) < 3: return None
    c[:, 0] = (c[:, 0] - x0) / w; c[:, 1] = (c[:, 1] - y0) / h
    c = np.clip(c, 0.0, 1.0)
    return " ".join("%.5f %.5f" % (p[0], p[1]) for p in c)


def main():
    proc = Sam2Processor.from_pretrained(SAM_ID); sam = Sam2Model.from_pretrained(SAM_ID).to(dev).eval()
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
    stat = []
    for p in sorted(SRC.glob(a.glob)):
        q = LB / (p.stem + ".png")
        if not q.exists(): print("ข้าม (ไม่มี label):", p.stem); continue
        bgr = cv2.imread(str(p)); lab = cv2.imread(str(q), cv2.IMREAD_GRAYSCALE)
        pile, info = pile_mask(bgr)
        if pile is None:
            pile = lab > 0                                        # ไม่มีกองบนพื้น -> ใช้บริเวณที่ label ระบุว่าเป็นกอง
            if pile.sum() < 1000: print("ข้าม (ไม่มีกอง):", p.stem); continue
        segs = segments(sam, proc, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), grid_points(pile), pile)
        x0, x1, y0, y1 = crop_box(bgr); w, h = x1 - x0, y1 - y0
        sp = "val" if p.stem in HOLD else "train"
        lines, nf, nb, nd = [], 0, 0, 0
        for m in segs:
            v = lab[m]
            dec = v[(v == 1) | (v == 2)]
            if dec.size < 50: nd += 1; continue
            frac_b = float((dec == 2).mean())
            if frac_b >= PURITY: cls = 1; nb += 1
            elif frac_b <= 1.0 - PURITY: cls = 0; nf += 1
            else: nd += 1; continue
            ln = poly_lines(m[y0:y1, x0:x1] if False else m, x0, y0, w, h)
            if ln: lines.append("%d %s" % (cls, ln))
        cv2.imwrite(str(OUT / "images" / sp / (p.stem + ".jpg")), bgr[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        open(OUT / "labels" / sp / (p.stem + ".txt"), "w").write("\n".join(lines))
        stat.append(dict(file=p.stem, split=sp, n_seg=len(segs), fresh=nf, burnt=nb, dropped=nd))
        print("%-30s %-5s seg %4d  fresh %4d  burnt %4d  ทิ้ง %4d" % (p.stem, sp, len(segs), nf, nb, nd))
    open(OUT / "data.yaml", "w").write(
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: fresh_stalk\n  1: burnt_stalk\n" % OUT.as_posix())
    json.dump(stat, open(OUT / "build_stat.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tr = [s for s in stat if s["split"] == "train"]
    print("train %d ใบ · fresh %d ชิ้น · burnt %d ชิ้น · ทิ้ง %d" %
          (len(tr), sum(s["fresh"] for s in tr), sum(s["burnt"] for s in tr), sum(s["dropped"] for s in tr)))


if __name__ == "__main__":
    main()
