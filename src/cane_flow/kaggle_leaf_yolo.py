# -*- coding: utf-8 -*-
"""Kaggle: เทรน YOLOv8 ตีกรอบ `top` (ยอดยาว) + `leaf_clump` (กระจุกใบแห้ง) จากกรอบที่คนวาดตาม leaf_label_GUIDE.md

อินพุต: DATA_URL (env หรือ Kaggle Secret) = zip ที่มี
  - export จาก Roboflow/CVAT แบบ YOLO (โฟลเดอร์ images/ + labels/ ชื่อไฟล์ตรงกับ leaf_label_set, ไม่สนว่า export แบ่ง split ยังไง)
  - manifest.json ของ leaf_label_set (ใช้ split จากไฟล์นี้เท่านั้น: test = วัน 20230118 ล็อกไว้ก่อนวาดกรอบ)
ออก: /kaggle/working/leaf_yolo.pt + leaf_eval.json · ห้ามเรียก MLflow บน Kaggle (kernel ถูก CANCEL) — log จากเครื่อง

เกณฑ์ผ่าน (ตั้งก่อนเห็นผล 2026-09-16): บน test 39 ใบ แต่ละคลาส precision >= 0.70 ที่ conf ที่เลือกจาก val
และ recall ระดับภาพ >= 0.70 (ภาพที่คนวาดคลาสนั้นไว้ ต้องมีกรอบคลาสนั้นอย่างน้อย 1 กรอบ)
แก้หลังรอบ 1 (บันทึกไว้ตรง ๆ): leaf_clump เป็น "พื้นที่" ขอบไม่ชัด → นับถูกเมื่อ IoU>=0.3 หรือจุดกลางกรอบทายอยู่ในกรอบเฉลย · top ยังใช้ IoU>=0.3 เท่าเดิม
รอบ 3 (self-training): เริ่มจาก init.pt ใน zip ถ้ามี · กรอบ top ที่ด้านยาว > 200 px ตัดทิ้งตอนทาย (RUBRIC: ยอด 30–120 px) ใช้ทั้ง val และ test
รอบ 4: splits.json มีค่า "val" ได้ (val ตายตัว ไม่เคยถูกเทรน) · เลือก conf แยกต่อคลาสจาก val · บันทึกผลทาย test ที่ conf 0.05 ลง test_preds.json
บั๊กรอบ 3: val สุ่มจาก train ที่ init model เคยเห็น → best.pt = epoch 1 → อย่า init จากโมเดลที่เทรนบน val
split: splits.json (stem→train/val/test) ถ้ามี ไม่งั้นใช้ manifest.json — ภาพเพิ่มจาก roboflow_cane เป็น train ทั้งหมด (ไม่มีวัน 20230118)
"""
import io, json, os, random, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)
from ultralytics import YOLO

CLASSES = ["top", "leaf_clump"]
TOP_MAX = 200
W = Path("/kaggle/working"); T = Path("/tmp/leaf")


def secret(k):
    if os.environ.get(k):
        return os.environ[k]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(k)
    except Exception:
        return None


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def yolo_boxes(txt, w, h):
    out = []
    if txt.exists():
        for ln in txt.read_text().split("\n"):
            p = ln.split()
            if len(p) == 5:
                c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
                out.append((c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return out


def main():
    raw = T / "raw"
    if not raw.exists():
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(secret("DATA_URL")).read())).extractall(raw)
    sp = list(raw.rglob("splits.json"))
    if sp:
        split_of = json.loads(sp[0].read_text(encoding="utf-8"))
    else:
        man = json.loads(next(raw.rglob("manifest.json")).read_text(encoding="utf-8"))
        split_of = {Path(r["file"]).stem: r["split"] for r in man["images"]}
    imgs = {p.stem: p for p in raw.rglob("*.jpg") if "images" in p.parts}
    labels = {p.stem: p for p in raw.rglob("*.txt") if "labels" in p.parts}
    # Roboflow เติม ".rf.<hash>" ต่อท้ายชื่อซ้ำอีกชั้นได้ — จับคู่ด้วย prefix ของชื่อต้นฉบับ
    def orig(stem):
        return stem if stem in split_of else next((s for s in split_of if stem.startswith(s[:40])), None)
    train_keys = [k for k in imgs if orig(k) and split_of[orig(k)] == "train"]
    test_keys = [k for k in imgs if orig(k) and split_of[orig(k)] == "test"]
    fixed_val = [k for k in imgs if orig(k) and split_of[orig(k)] == "val"]
    if fixed_val:
        val_keys, trn_keys = fixed_val, train_keys
    else:
        random.Random(0).shuffle(train_keys)
        val_keys, trn_keys = train_keys[:30], train_keys[30:]
    print(f"train {len(trn_keys)} · val {len(val_keys)} · test {len(test_keys)} · ภาพที่จับคู่ไม่ได้ {len(imgs) - len(train_keys) - len(test_keys)}", flush=True)
    for name, keys in (("train", trn_keys), ("val", val_keys), ("test", test_keys)):
        for sub in ("images", "labels"):
            (T / "ds" / sub / name).mkdir(parents=True, exist_ok=True)
        for k in keys:
            shutil.copy(imgs[k], T / "ds" / "images" / name / imgs[k].name)
            if k in labels:
                shutil.copy(labels[k], T / "ds" / "labels" / name / (imgs[k].stem + ".txt"))
    (T / "ds.yaml").write_text(f"path: {T/'ds'}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {CLASSES}\n")

    init = list(raw.rglob("init.pt"))
    m = YOLO(str(init[0]) if init else os.environ.get("YOLO_BASE", "yolov8m.pt"))
    print("init from", init[0] if init else "coco", flush=True)
    m.train(data=str(T / "ds.yaml"), imgsz=int(os.environ.get("YOLO_IMGSZ", 960)), epochs=int(os.environ.get("YOLO_EPOCHS", 150)),
            batch=8, patience=50, seed=0, project=str(T / "runs"), name="leaf", verbose=False, plots=False)
    best = YOLO(str(T / "runs" / "leaf" / "weights" / "best.pt"))
    shutil.copy(T / "runs" / "leaf" / "weights" / "best.pt", W / "leaf_yolo.pt")

    def score(keys, conf, dump=None):
        """คืน precision ระดับกรอบ (IoU>=0.3 กับกรอบคนคลาสเดียวกัน) และ recall ระดับภาพ ต่อคลาส"""
        st = {c: dict(tp=0, fp=0, img_hit=0, img_pos=0) for c in range(len(CLASSES))}
        for k in keys:
            cmin = min(conf.values()) if isinstance(conf, dict) else conf
            r = best.predict(str(imgs[k]), conf=cmin, imgsz=int(os.environ.get('YOLO_IMGSZ', 960)), verbose=False)[0]
            if dump is not None:
                dump[k] = [[int(c), [round(v, 1) for v in b], round(s, 3)] for c, b, s in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
            h, w = r.orig_shape
            gt = yolo_boxes(labels.get(k, Path("/none")), w, h)
            thr = conf if isinstance(conf, dict) else {0: conf, 1: conf}
            pr = [(int(c), b) for c, b, s in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist())
                  if s >= thr[int(c)] and not (int(c) == 0 and max(b[2] - b[0], b[3] - b[1]) > TOP_MAX)]
            for c in st:
                g = [b for cc, b in gt if cc == c]; p = [b for cc, b in pr if cc == c]
                for b in p:
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    hit = any(iou(b, x) >= 0.3 or (CLASSES[c] == "leaf_clump" and x[0] <= cx <= x[2] and x[1] <= cy <= x[3]) for x in g)
                    if hit: st[c]["tp"] += 1
                    else: st[c]["fp"] += 1
                if g:
                    st[c]["img_pos"] += 1; st[c]["img_hit"] += bool(p)
        return {CLASSES[c]: dict(precision=v["tp"] / max(1, v["tp"] + v["fp"]), n_boxes=v["tp"] + v["fp"],
                                 img_recall=v["img_hit"] / max(1, v["img_pos"]), n_pos_imgs=v["img_pos"]) for c, v in st.items()}

    # เลือก conf จาก val เท่านั้น (ค่าต่ำสุดที่ precision ทุกคลาส >= 0.70 ถ้าไม่มี ใช้ 0.25)
    grid = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)
    vs = {c: score(val_keys, c) for c in grid}
    conf = {}
    for ci, name in enumerate(CLASSES):   # ต่อคลาส: ค่าต่ำสุดที่ precision บน val >= 0.70 (ไม่มี → ค่าที่ precision สูงสุด)
        ok = [c for c in grid if vs[c][name]["n_boxes"] and vs[c][name]["precision"] >= 0.7]
        conf[ci] = ok[0] if ok else max(grid, key=lambda c: (vs[c][name]["precision"], -c))
    res = score(test_keys, conf)
    dump05 = {}
    score(test_keys, 0.05, dump05)                      # ผลทายดิบ conf >= 0.05 ไว้วิเคราะห์บนเครื่อง
    (W / "test_preds.json").write_text(json.dumps(dump05))
    val_at = {CLASSES[ci]: vs[c][CLASSES[ci]] for ci, c in conf.items()}
    passed = all(v["precision"] >= 0.7 and v["img_recall"] >= 0.7 for v in res.values())
    out = dict(conf={CLASSES[k]: v for k, v in conf.items()}, val=val_at, test=res, passed=passed, n_train=len(trn_keys), n_val=len(val_keys), n_test=len(test_keys),
               map50_test=float(best.val(data=str(T / "ds.yaml"), split="test", imgsz=int(os.environ.get("YOLO_IMGSZ", 960)), verbose=False, plots=False).box.map50))
    (W / "leaf_eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False), flush=True)
    print("PASS" if passed else "NOT PASS", flush=True)


main()
