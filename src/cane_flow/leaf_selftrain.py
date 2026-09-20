# -*- coding: utf-8 -*-
"""self-training หนึ่งรอบสำหรับ leaf YOLO (inference บนเครื่องเท่านั้น — การเทรนทำบน Kaggle)

  python leaf_selftrain.py --model <best.pt> --lab <leaflab dir> --tag r3 [--n-fresh 300 --n-cut 100]

ทำ 3 อย่าง กับชุด TRAIN เท่านั้น (ภาพ split=test ไม่ถูกแตะเด็ดขาด):
  1. เก็บกรอบดี: ภาพ train ที่มี label แล้ว — เพิ่มกรอบ top ที่โมเดลมั่นใจ (conf >= KEEP) และไม่ซ้อนกรอบเดิม (IoU < 0.3)
  2. ตัดกรอบแย่: ลบกรอบ top เดิมที่โมเดลไม่เห็นเลย (ไม่มีกรอบทายใด conf >= DROP ที่ IoU >= 0.1)
  3. ภาพใหม่ pseudo-label: รถอ้อยทั้งลำกลางวัน + รถอ้อยสับ (ตัวอย่างลบ/ผ้าใบ) จาก roboflow_cane ที่ไม่ใช่วัน test และไม่ซ้ำกลุ่มรถ
     เก็บเฉพาะกรอบ conf >= KEEP · อ้อยสับเก็บทุกภาพ (ว่างได้ = ตัวอย่างลบ) · อ้อยทั้งลำเก็บเฉพาะภาพที่มีกรอบ (ภาพว่างอาจแค่หลุด)
กฎจากนิยามคลาส (RUBRIC: ยอด 30–120 px): กรอบ top ที่ด้านยาว > TOP_MAX px ไม่ใช่ยอด → ไม่เก็บ
ออก: <lab>/labels_<tag>/ (ครบทุกภาพ train+test; test คัดลอกจาก labels_test_final ถ้ามี ไม่งั้นจาก labels) + index_<tag>.json + <tag>_selftrain.json
"""
import argparse, json, os, random, shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

KEEP, DROP, TOP_MAX = 0.6, 0.05, 200
R = Path(__file__).resolve().parents[3]
RF = R / "gen_images" / "roboflow_cane"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--lab", required=True); ap.add_argument("--tag", default="r3")
ap.add_argument("--n-fresh", type=int, default=300); ap.add_argument("--n-cut", type=int, default=100); ap.add_argument("--imgsz", type=int, default=960)
a = ap.parse_args()
L = Path(a.lab); OUT = L / f"labels_{a.tag}"; OUT.mkdir(exist_ok=True)
m = YOLO(a.model)


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


def read(p, w, h):
    out = []
    if p.exists():
        for ln in p.read_text().split("\n"):
            q = ln.split()
            if len(q) == 5:
                c, cx, cy, bw, bh = int(q[0]), *map(float, q[1:])
                out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def write(p, boxes, w, h):
    p.write_text("".join(f"{c} {(b[0]+b[2])/2/w:.6f} {(b[1]+b[3])/2/h:.6f} {(b[2]-b[0])/w:.6f} {(b[3]-b[1])/h:.6f}\n" for c, b in boxes))


def preds(path, conf):
    r = m.predict(str(path), conf=conf, imgsz=a.imgsz, verbose=False)[0]
    out = []
    for c, b, s in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        if int(c) == 0 and max(b[2] - b[0], b[3] - b[1]) > TOP_MAX:
            continue
        out.append((int(c), b, s))
    return out, r.orig_shape


idx = json.load(open(L / "index.json"))
stats = dict(added_top=0, dropped_top=0, refined_imgs=0, new_fresh=0, new_cut=0, new_boxes=0)
test_src = L / "labels_test_final" if (L / "labels_test_final").exists() else L / "labels"
for x in idx:
    src = L / "labels" / (x["stem"] + ".txt")
    if x["split"] == "test":
        shutil.copy(test_src / (x["stem"] + ".txt"), OUT / (x["stem"] + ".txt"))
        continue
    pr, (h, w) = preds(x["path"], DROP)
    gt = read(src, w, h)
    keep = []
    for c, b in gt:
        if c == 0 and not any(pc == 0 and iou(b, pb) >= 0.1 for pc, pb, _ in pr):
            stats["dropped_top"] += 1
            continue
        keep.append((c, b))
    add = [(0, pb) for pc, pb, s in pr if pc == 0 and s >= KEEP and not any(c == 0 and iou(pb, b) >= 0.3 for c, b in keep)]
    stats["added_top"] += len(add); stats["refined_imgs"] += bool(add) or len(keep) != len(gt)
    write(OUT / (x["stem"] + ".txt"), keep + add, w, h)

# ภาพใหม่
g2i = json.load(open(R / "qa/own_model/cane_quality/rf_audit_out/truck_groups.json", encoding="utf-8"))["image_to_group"]
used = {g2i.get(x["file"]) for x in idx}
new = []
for cls, n in (("cane_fresh", a.n_fresh), ("cane_cut", a.n_cut)):
    c, seen = [], set()
    for sp in ("train", "valid", "test"):
        for p in sorted((RF / sp / cls).glob("*.jpg")):
            if p.name.startswith("20230118"): continue
            if cls == "cane_fresh" and not 7 <= int(p.name[9:11]) < 17: continue
            g = g2i.get(p.name)
            if g is None or g in used or g in seen: continue
            seen.add(g); c.append(p)
    random.Random(7).shuffle(c)
    for i, p in enumerate(c[:n]):
        pr, (h, w) = preds(p, KEEP)
        boxes = [(pc, pb) for pc, pb, _ in pr]
        if cls == "cane_fresh" and not boxes:
            continue
        write(OUT / (p.stem + ".txt"), boxes, w, h)
        stats["new_boxes"] += len(boxes); stats["new_fresh" if cls == "cane_fresh" else "new_cut"] += 1
        new.append(dict(id=f"P{len(new):03d}", file=p.name, stem=p.stem, split="train", camera="MPK00" if "MPK00" in p.name else "MPDC00",
                        mill_class=cls, w=w, h=h, path=str(p), source="pseudo"))
json.dump(idx + new, open(L / f"index_{a.tag}.json", "w"), indent=1)
stats.update(keep=KEEP, drop=DROP, top_max=TOP_MAX, model=a.model, n_index=len(idx) + len(new))
json.dump(stats, open(L / f"{a.tag}_selftrain.json", "w"), indent=1)
print(json.dumps(stats, ensure_ascii=False))
