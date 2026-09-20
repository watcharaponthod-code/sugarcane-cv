# -*- coding: utf-8 -*-
"""จำแนกสภาพอ้อยต่อคันจากภาพ CCTV จริง 4 คลาส (cane_burn / cane_fresh / cane_cut / cane_net)
ชุดข้อมูล: gen_images/roboflow_cane — Roboflow Universe "aimlsugarcane/cane-classification-classify" v6 (CC BY 4.0)

**แบ่ง split ใหม่ตามวัน ไม่ใช้ split เดิมของ Roboflow**: ภาพเป็นเฟรมต่อเนื่องของรถคันเดียว 89% ของกลุ่มรถ
กระจายข้าม train/valid/test เดิม → ตัวเลขบน split เดิมคือการวัดตัวเอง ห้ามใช้
วันที่/กล้อง/เวลา อ่านจากชื่อไฟล์ YYYYMMDD-HHMMSS_CAM...

รัน:
  python cane_cls_train.py --arch dino      # DINOv2-base แช่แข็ง + MLP (baseline เร็ว, cache feature)
  python cane_cls_train.py --arch convnext  # fine-tune convnext_tiny ทั้งตัว lr 1e-4
ทุก hyperparameter ตั้งจาก valid เท่านั้น ห้ามแตะ test ระหว่างพัฒนา
"""
import re, json, time, argparse, random
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SRC = ROOT / "gen_images" / "roboflow_cane"
CLASSES = ["cane_burn", "cane_fresh", "cane_cut", "cane_net"]
BURN = 0
GROUPS_JSON = None      # rf_audit_out/truck_groups.json (db2b2-0a) — หน่วยจริงคือ "คันรถ" ไม่ใช่เฟรม
CONFLICTS = None        # rf_audit_out/label_conflicts.json — ภาพที่ label ขัดกันเอง 407 ใบ (เพดานความแม่น)
SPLIT_DAYS = {"train": ["20230113", "20230116", "20230117"], "valid": ["20230118"], "test": ["20230119", "20230115"]}
FOLDS = [["20230113"], ["20230115", "20230116"], ["20230117"], ["20230118"], ["20230119"]]   # 0115 เล็กเกิน (65 ใบ) รวมกับวันข้างเคียง

ap = argparse.ArgumentParser()
ap.add_argument("--arch", choices=("dino", "convnext", "effnet"), default="convnext")
ap.add_argument("--size", type=int, default=448); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default=None)
ap.add_argument("--keep-net", action="store_true", help="เก็บคลาส cane_net ไว้ในเมตริกหลัก (ดีฟอลต์ = ตัดออก เพราะ 'ถูกคลุมตาข่าย' อยู่คนละแกน ไม่ mutually exclusive กับชนิดอ้อย)")
ap.add_argument("--days-train", default=None, help="กำหนดวันของ train เอง คั่นด้วย + (ใช้ตอนเทรนตัวส่งมอบ)")
ap.add_argument("--days-valid", default=None)
ap.add_argument("--days-test", default=None)
ap.add_argument("--only-camera", default=None, choices=("MPDC00", "MPK00"), help="ใช้กล้องเดียวทั้ง train/valid/test — วัดว่า 'ระบบหนึ่งกล้อง หนึ่งมุม' ทำได้แค่ไหน")
ap.add_argument("--cross-camera", default=None, choices=("MPDC00", "MPK00"), help="เทรนด้วยกล้องนี้กล้องเดียว แล้วทดสอบบนอีกกล้อง — การทดสอบชี้ขาดว่าโมเดลเรียนอ้อยหรือเรียนกล้อง")
ap.add_argument("--save-full", action="store_true", help="เทรนตัวใช้งานจริง: train = ทุกวันทุกกล้องยกเว้นวัน valid · valid = วันเดียวไว้เลือก epoch · เซฟเป็น <tag>_full.pt")
ap.add_argument("--full-valid-day", default="20230119", help="วันที่กันไว้เป็น valid ของ --save-full (ไม่ใช่ blind test)")
ap.add_argument("--cv", action="store_true", help="หมุนวันเป็น 5 fold: test = fold นั้น · valid = fold ถัดไป · train = ที่เหลือ (hyperparameter เดิมทุก fold)")
a = ap.parse_args()
a.tag = a.tag or ("cane_cls_" + a.arch)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = HERE / "cane_cls_out"; OUT.mkdir(exist_ok=True)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_audit():
    global GROUPS_JSON, CONFLICTS
    d = HERE / "rf_audit_out"
    try: GROUPS_JSON = json.load(open(d / "truck_groups.json", encoding="utf-8"))["image_to_group"]
    except Exception: GROUPS_JSON = {}
    try: CONFLICTS = set(json.load(open(d / "label_conflicts.json", encoding="utf-8"))["images"])
    except Exception: CONFLICTS = set()
    print("audit: กลุ่มรถ %d ภาพ · ภาพที่ label ขัดกัน %d ใบ" % (len(GROUPS_JSON), len(CONFLICTS)))


def scan(split_days):
    """อ่านทุกโฟลเดอร์ของ Roboflow แล้วแบ่งใหม่ตามวันในชื่อไฟล์"""
    day_of = {d: sp for sp, ds in split_days.items() for d in ds}
    items = {"train": [], "valid": [], "test": []}
    for sp0 in ("train", "valid", "test"):
        for ci, c in enumerate(CLASSES):
            for p in sorted((SRC / sp0 / c).glob("*.jpg")):
                m = re.match(r"(\d{8})-(\d{6})_([A-Za-z0-9]+)", p.name)
                if not m: continue
                sp = day_of.get(m.group(1))
                if not sp: continue
                if c == "cane_net" and not a.keep_net: continue
                if a.only_camera and m.group(3) != a.only_camera: continue
                if a.cross_camera:
                    same = (m.group(3) == a.cross_camera)
                    if sp in ("train", "valid") and not same: continue      # เทรน/วาลิด = กล้องเดียว
                    if sp == "test" and same: continue                      # ทดสอบ = อีกกล้องเท่านั้น
                items[sp].append((p, ci, m.group(1), m.group(3)))
    return items


class DS(Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        p, y, _d, _c = self.rows[i]
        im = cv2.imread(str(p))
        if self.train:
            if random.random() < 0.5: im = im[:, ::-1]
            s = random.uniform(0.85, 1.0); H, W = im.shape[:2]           # ครอปเล็กน้อย ไม่ตัดกระบะทิ้ง
            h, w = int(H * s), int(W * s)
            y0, x0 = random.randint(0, H - h), random.randint(0, W - w)
            im = im[y0:y0 + h, x0:x0 + w]
            v = im.astype(np.float32) * random.uniform(0.85, 1.15) + random.uniform(-15, 15)   # สว่าง/คอนทราสต์เบา ๆ
            v *= np.array([random.uniform(0.93, 1.07), 1.0, random.uniform(0.93, 1.07)], np.float32)  # ไม่แรงจนเปลี่ยนสีอ้อย
            im = np.clip(v, 0, 255).astype(np.uint8)
        im = cv2.resize(im, (a.size, a.size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).permute(2, 0, 1).float() / 255.0
        return x, y


def build_model():
    if a.arch == "dino":
        from transformers import AutoModel
        bb = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
        for q in bb.parameters(): q.requires_grad_(False)
        head = nn.Sequential(nn.Linear(bb.config.hidden_size * 2, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, len(CLASSES))).to(dev)

        def fwd(x, grad_bb=False):
            with torch.no_grad():
                h = bb(x).last_hidden_state
                f = torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)           # CLS + mean ของ patch
            return head(f)
        return fwd, list(head.parameters()), head
    import torchvision.models as tvm
    if a.arch == "convnext":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(CLASSES))
    else:
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    m = m.to(dev)
    return (lambda x, grad_bb=True: m(x)), list(m.parameters()), m


@torch.no_grad()
def predict(fwd, loader):
    logits, ys = [], []
    for x, y in loader:
        x = ((x.to(dev) - MEAN.to(dev)) / STD.to(dev))
        logits.append(fwd(x).float().cpu()); ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def metrics(logits, ys, tag):
    pred = logits.argmax(1).numpy(); y = ys.numpy()
    n = len(CLASSES)
    cm = np.zeros((n, n), int)
    for t, p in zip(y, pred): cm[t, p] += 1
    acc = float((pred == y).mean())
    out = dict(split=tag, n=len(y), accuracy=round(acc, 4), confusion=cm.tolist(), per_class={})
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i].sum() - tp
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        out["per_class"][c] = dict(n=int(cm[i].sum()), precision=round(float(pr), 4), recall=round(float(rc), 4),
                                   f1=round(float(2 * pr * rc / max(pr + rc, 1e-9)), 4))
    tp = int(((pred == BURN) & (y == BURN)).sum()); fp = int(((pred == BURN) & (y != BURN)).sum())
    fn = int(((pred != BURN) & (y == BURN)).sum()); tn = int(((pred != BURN) & (y != BURN)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    out["burn_vs_rest"] = dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(float(pr), 4), recall=round(float(rc), 4),
                               f1=round(float(2 * pr * rc / max(pr + rc, 1e-9)), 4))
    return out


def show(m):
    print("[%s] n=%d acc=%.4f" % (m["split"], m["n"], m["accuracy"]))
    print("  confusion (แถว=จริง คอลัมน์=ทาย) " + " ".join("%10s" % c[5:] for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print("    %-11s" % c[5:] + " ".join("%10d" % v for v in m["confusion"][i]))
    for c, v in m["per_class"].items():
        print("  %-11s n=%4d P=%.3f R=%.3f F1=%.3f" % (c[5:], v["n"], v["precision"], v["recall"], v["f1"]))
    b = m["burn_vs_rest"]
    print("  BURN vs REST  P=%.3f R=%.3f F1=%.3f  (tp %d fp %d fn %d tn %d)" % (b["precision"], b["recall"], b["f1"], b["tp"], b["fp"], b["fn"], b["tn"]))


def baselines(rows, ys):
    """baseline ที่ไม่ดูภาพเลย — โมเดลต้องชนะให้ชัด ไม่งั้นตัวเลขไม่มีความหมาย
    camera-only = ทายว่า burn เมื่อกล้อง MPDC00 · majority = ทายคลาสที่เยอะสุดของ train เสมอ"""
    y = ys.numpy()
    cam_pred = np.array([BURN if r[3] == "MPDC00" else 1 for r in rows])
    out = {}
    for name, pr in (("camera_only", cam_pred),):
        tp = int(((pr == BURN) & (y == BURN)).sum()); fp = int(((pr == BURN) & (y != BURN)).sum())
        fn = int(((pr != BURN) & (y == BURN)).sum())
        p2 = tp / max(tp + fp, 1); r2 = tp / max(tp + fn, 1)
        out[name] = dict(precision=round(float(p2), 4), recall=round(float(r2), 4),
                         f1=round(float(2 * p2 * r2 / max(p2 + r2, 1e-9)), 4), tp=tp, fp=fp, fn=fn)
        print("  [baseline %s] BURN R=%.3f P=%.3f F1=%.3f (tp %d fp %d fn %d)" % (name, r2, p2, out[name]["f1"], tp, fp, fn))
    maj = int(np.bincount(y, minlength=len(CLASSES)).argmax())
    out["majority_class"] = dict(cls=CLASSES[maj], accuracy=round(float((y == maj).mean()), 4))
    print("  [baseline majority=%s] acc=%.4f" % (CLASSES[maj][5:], out["majority_class"]["accuracy"]))
    return out


def by_group(rows, logits, ys):
    """รวมผลเป็นรายคันรถด้วยเสียงข้างมากของเฟรมในกลุ่ม — หน้างานตัดสินต่อคัน ไม่ใช่ต่อเฟรม"""
    if not GROUPS_JSON: return None
    pred = logits.argmax(1).numpy(); y = ys.numpy()
    g = {}
    for i, r in enumerate(rows):
        gid = GROUPS_JSON.get(r[0].name)
        if gid is None: continue
        g.setdefault(gid, ([], []))[0].append(pred[i]); g[gid][1].append(y[i])
    if not g: return None
    gp = np.array([np.bincount(v[0], minlength=len(CLASSES)).argmax() for v in g.values()])
    gy = np.array([np.bincount(v[1], minlength=len(CLASSES)).argmax() for v in g.values()])
    tp = int(((gp == BURN) & (gy == BURN)).sum()); fp = int(((gp == BURN) & (gy != BURN)).sum())
    fn = int(((gp != BURN) & (gy == BURN)).sum())
    p2 = tp / max(tp + fp, 1); r2 = tp / max(tp + fn, 1)
    out = dict(n_groups=len(g), n_burn=int((gy == BURN).sum()), accuracy=round(float((gp == gy).mean()), 4),
               precision=round(float(p2), 4), recall=round(float(r2), 4),
               f1=round(float(2 * p2 * r2 / max(p2 + r2, 1e-9)), 4), tp=tp, fp=fp, fn=fn)
    print("  [ต่อคันรถ] n=%d n_burn=%d acc=%.4f BURN R=%.3f P=%.3f (tp %d fp %d fn %d)"
          % (out["n_groups"], out["n_burn"], out["accuracy"], r2, p2, tp, fp, fn))
    return out


def conflict_share(rows, logits, ys):
    """สัดส่วนของภาพที่ทายผิดซึ่งอยู่ในรายการ label ขัดกันเอง — ถ้าสูง แปลว่า error ที่เหลือคือ label noise"""
    if not CONFLICTS: return None
    pred = logits.argmax(1).numpy(); y = ys.numpy()
    bad = [i for i in range(len(y)) if pred[i] != y[i]]
    if not bad: return None
    inc = sum(1 for i in bad if rows[i][0].name in CONFLICTS)
    base = sum(1 for r in rows if r[0].name in CONFLICTS) / max(len(rows), 1)
    out = dict(n_wrong=len(bad), n_wrong_in_conflicts=inc, share_wrong=round(inc / len(bad), 4), share_all=round(base, 4))
    print("  [label ขัดกัน] ทายผิด %d ใบ อยู่ในรายการขัดกัน %d ใบ (%.1f%% เทียบพื้นฐานทั้ง test %.1f%%)"
          % (len(bad), inc, 100 * out["share_wrong"], 100 * base))
    return out


def by_camera(rows, logits, ys):
    """แยกผลรายกล้อง — จำเป็นเพราะกล้องกับคลาสผูกกันเกือบสมบูรณ์ (burn 94% มาจาก MPDC00, fresh 79% จาก MPK00)
    โมเดลจึงทายถูกได้บางส่วนจากการรู้ว่าเป็นกล้องไหน ค่าที่ต่ำกว่าระหว่างสองกล้องคือตัวแทนความสามารถจริง"""
    pred = logits.argmax(1).numpy(); y = ys.numpy()
    cams = sorted({r[3] for r in rows})
    out = {}
    for cam in cams:
        idx = np.array([i for i, r in enumerate(rows) if r[3] == cam])
        if not len(idx): continue
        p2, y2 = pred[idx], y[idx]
        tp = int(((p2 == BURN) & (y2 == BURN)).sum()); fp = int(((p2 == BURN) & (y2 != BURN)).sum())
        fn = int(((p2 != BURN) & (y2 == BURN)).sum())
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        out[cam] = dict(n=int(len(idx)), n_burn=int((y2 == BURN).sum()), accuracy=round(float((p2 == y2).mean()), 4),
                        precision=round(float(pr), 4), recall=round(float(rc), 4), tp=tp, fp=fp, fn=fn)
        print("  [กล้อง %s] n=%d n_burn=%d acc=%.4f BURN R=%.3f P=%.3f (tp %d fp %d fn %d)"
              % (cam, out[cam]["n"], out[cam]["n_burn"], out[cam]["accuracy"], rc, pr, tp, fp, fn))
    return out


def error_grid(rows, logits, ys, path, k=12):
    pred = logits.argmax(1).numpy(); y = ys.numpy()
    conf = logits.softmax(1).max(1).values.numpy()
    bad = [i for i in range(len(y)) if pred[i] != y[i]]
    bad.sort(key=lambda i: -conf[i])                                     # ผิดแบบมั่นใจที่สุดก่อน
    bad = bad[:k]
    if not bad: return 0
    tiles = []
    for i in bad:
        im = cv2.resize(cv2.imread(str(rows[i][0])), (320, 320))
        cv2.rectangle(im, (0, 0), (320, 44), (0, 0, 0), -1)
        cv2.putText(im, "true %s" % CLASSES[y[i]][5:], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(im, "pred %s %.2f" % (CLASSES[pred[i]][5:], conf[i]), (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 160, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % 4: tiles.append(np.zeros((320, 320, 3), np.uint8))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return len(bad)


def run_one(items, quiet=False):
    """เทรน 1 รอบด้วย split ที่ items ให้มา คืน (valid metrics, test metrics, logits, ys)"""
    dl = {sp: DataLoader(DS(items[sp], sp == "train"), batch_size=a.bs, shuffle=(sp == "train"),
                         num_workers=a.workers, pin_memory=True, drop_last=(sp == "train")) for sp in items}
    fwd, params, module = build_model()
    cnt = np.bincount([r[1] for r in items["train"]], minlength=len(CLASSES)).astype(np.float32)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, device=dev); w = w / w.mean()   # ถ่วงคลาสไม่สมดุล (net 140 vs cut 2541)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * max(1, len(dl["train"])))
    t0 = time.time(); best = (-1, None)
    for ep in range(a.epochs):
        if hasattr(module, "train"): module.train()
        tot = k = 0
        for x, y in dl["train"]:
            x = ((x.to(dev, non_blocking=True) - MEAN.to(dev)) / STD.to(dev)); y = y.to(dev)
            loss = crit(fwd(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item(); k += 1
        if hasattr(module, "eval"): module.eval()
        lg, ys = predict(fwd, dl["valid"]); mv = metrics(lg, ys, "valid")
        print("ep %d loss %.3f  valid acc %.4f  burn R %.3f P %.3f  %.0fs"
              % (ep + 1, tot / max(k, 1), mv["accuracy"], mv["burn_vs_rest"]["recall"], mv["burn_vs_rest"]["precision"], time.time() - t0), flush=True)
        if mv["burn_vs_rest"]["f1"] > best[0]:                            # เลือก epoch จาก valid เท่านั้น
            best = (mv["burn_vs_rest"]["f1"], {k2: v.detach().cpu().clone() for k2, v in module.state_dict().items()}, ep + 1, mv)
    module.load_state_dict(best[1]); module.eval()
    lv, yv = predict(fwd, dl["valid"])                                   # ต้องมี p_burn ของ valid ด้วย เพราะ knob ทุกตัวจูนบน valid เท่านั้น
    pv = lv.softmax(1)[:, BURN].tolist()
    best[3]["preds"] = [[r[0].name, int(yv[i]), int(lv[i].argmax()), round(pv[i], 5), r[3], r[2]] for i, r in enumerate(items["valid"])]
    lg, ys = predict(fwd, dl["test"]); mt = metrics(lg, ys, "test")
    return best[3], mt, lg, ys, best[2], module, round(time.time() - t0)


def counts_line(items, sp, days):
    cc = np.bincount([r[1] for r in items[sp]], minlength=len(CLASSES))
    return "%-5s n=%4d วัน %s  %s" % (sp, len(items[sp]), "+".join(days), dict(zip([c[5:] for c in CLASSES], cc.tolist())))


def save_ckpt(module, path_stem, extra):
    """เซฟน้ำหนัก + metadata ที่คนโหลดต้องรู้

    **หัวโมเดลเป็น Linear(..., 4) เสมอ** เพราะ CLASSES ตายตัว 4 คลาส
    `--keep-net` กรองเฉพาะตอน scan ภาพ ไม่ได้เปลี่ยนขนาดหัว → ถ้าเทรนโดยไม่มี cane_net
    เอาต์พุตช่องที่ 3 (index 3) จะไม่เคยถูกเทรนเลย **ห้ามเอาไปตีความ** ให้ตัดทิ้งหรือ mask ก่อน softmax
    """
    torch.save(dict(state=module.state_dict(), arch=a.arch, size=a.size, classes=CLASSES,
                    head_out=len(CLASSES), trained_classes=[c for c in CLASSES if a.keep_net or c != "cane_net"],
                    keep_net=bool(a.keep_net),
                    warning="head is Linear(...,4) always; with keep_net=False index 3 (cane_net) is untrained - mask it",
                    normalize=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], bgr_to_rgb=True),
                    **extra), HERE / (path_stem + ".pt"))
    json.dump(dict(arch=a.arch, size=a.size, classes=CLASSES, head_out=len(CLASSES),
                   trained_classes=[c for c in CLASSES if a.keep_net or c != "cane_net"],
                   keep_net=bool(a.keep_net),
                   warning="head is Linear(...,4) always; with keep_net=False index 3 (cane_net) is untrained - mask it",
                   normalize=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], bgr_to_rgb=True),
                   **{k: v for k, v in extra.items() if k != "state"}),
              open(HERE / (path_stem + ".ptinfo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main():
    load_audit()
    if a.days_train:
        SPLIT_DAYS["train"] = a.days_train.split("+")
        if a.days_valid: SPLIT_DAYS["valid"] = a.days_valid.split("+")
        if a.days_test: SPLIT_DAYS["test"] = a.days_test.split("+")
    if a.cv: return main_cv()
    if a.save_full: return main_full()
    items = scan(SPLIT_DAYS)
    for sp in ("train", "valid", "test"): print(counts_line(items, sp, SPLIT_DAYS[sp]))
    print("arch=%s size=%d bs=%d lr=%g epochs=%d · %s" % (a.arch, a.size, a.bs, a.lr, a.epochs, dev))
    mv, mt, lg, ys, ep, module, secs = run_one(items)
    print("เลือก epoch %d จาก valid burn-F1 %.4f" % (ep, mv["burn_vs_rest"]["f1"]))
    show(mv); show(mt)
    mt["by_camera"] = by_camera(items["test"], lg, ys)
    mt["by_group"] = by_group(items["test"], lg, ys)
    mt["baselines"] = baselines(items["test"], ys)
    mt["conflicts"] = conflict_share(items["test"], lg, ys)
    _pb = lg.softmax(1)[:, BURN].tolist()
    mt["preds"] = [[r[0].name, int(ys[i]), int(lg[i].argmax()), round(_pb[i], 5), r[3], r[2]] for i, r in enumerate(items["test"])]
    nb = error_grid(items["test"], lg, ys, OUT / (a.tag + "_errors.jpg"))
    save_ckpt(module, a.tag, dict(mode="single-split", split_days=SPLIT_DAYS, best_epoch=ep))
    json.dump(dict(tag=a.tag, arch=a.arch, size=a.size, bs=a.bs, lr=a.lr, epochs=a.epochs, seed=a.seed,
                   split_days=SPLIT_DAYS, best_epoch=ep, train_s=secs,
                   counts={sp: int(len(items[sp])) for sp in items}, valid=mv, test=mt, n_error_tiles=nb),
              open(HERE / (a.tag + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", a.tag)


def main_full():
    """เทรนตัวใช้งานจริงบน **ทุกวัน ทุกกล้อง** (กันไว้วันเดียวเป็น valid สำหรับเลือก epoch)

    **ตัวเลขของโมเดลตัวนี้ไม่ใช่ blind test** เพราะไม่มีวันไหนถูกกันไว้เป็น test เลย
    ตัวเลขที่อ้างได้คือผล 5-fold เท่านั้น — ตัวนี้มีไว้เอาไปใช้งาน ไม่ใช่เอาไปรายงานความแม่น
    """
    vd = a.full_valid_day
    all_days = [d for f in FOLDS for d in f]
    # ห้ามใส่ vd ทั้งใน valid และ test: scan() สร้าง day_of = {วัน: split} ซึ่ง key ซ้ำแล้วตัวหลังทับตัวหน้า
    # (test จะทับ valid ทำให้ valid ว่างเปล่า) → scan ด้วย valid อย่างเดียว แล้วค่อยก็อปไปเป็น test
    sd = {"train": [d for d in all_days if d != vd], "valid": [vd], "test": []}
    # กันบั๊กเดิมซ้ำ: scan() สร้าง day_of = {วัน: split} ซึ่ง key ซ้ำแล้วตัวหลังทับตัวหน้าเงียบ ๆ
    seen = [d for k in ("train", "valid", "test") for d in sd[k]]
    assert len(seen) == len(set(seen)), "วันซ้ำข้าม split: %s" % sd
    assert sd["valid"], "valid ว่าง"
    items = scan(sd)
    assert items["valid"], "valid ไม่มีภาพเลย (วัน %s ไม่มีอยู่จริง หรือถูกกรองหมด)" % sd["valid"]
    assert items["train"], "train ไม่มีภาพเลย"
    items["test"] = list(items["valid"])
    print("=== --save-full · train = %s · valid/test = %s (ไม่ใช่ blind test)" % ("+".join(sd["train"]), vd))
    for sp in ("train", "valid"): print("  " + counts_line(items, sp, sd[sp]))
    mv, mt, lg, ys, ep, module, secs = run_one(items)
    print("เลือก epoch %d จาก valid burn-F1 %.4f" % (ep, mv["burn_vs_rest"]["f1"]))
    save_ckpt(module, a.tag + "_full",
              dict(mode="save-full", split_days=sd, best_epoch=ep, train_s=secs,
                   not_blind_test=True,
                   note="valid day is also reported as test here; these numbers are NOT a blind test"))
    json.dump(dict(tag=a.tag + "_full", mode="save-full", split_days=sd, best_epoch=ep,
                   counts={sp: int(len(items[sp])) for sp in items}, valid=mv,
                   not_blind_test=True), open(HERE / (a.tag + "_full.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved", a.tag + "_full")


def main_cv():
    """หมุนวัน: test = fold i · valid = fold i+1 · train = ที่เหลือ — hyperparameter เดิมทุก fold ไม่จูนรายรอบ"""
    rows = []
    for i, te in enumerate(FOLDS):
        va = FOLDS[(i + 1) % len(FOLDS)]
        tr = [d for j, f in enumerate(FOLDS) if j not in (i, (i + 1) % len(FOLDS)) for d in f]
        sd = {"train": tr, "valid": va, "test": te}
        items = scan(sd)
        print("\n=== fold %d/%d · test=%s valid=%s" % (i + 1, len(FOLDS), "+".join(te), "+".join(va)))
        for sp in ("train", "valid", "test"): print("  " + counts_line(items, sp, sd[sp]))
        if not items["test"] or not items["valid"]: print("  ข้าม fold นี้ (ว่าง)"); continue
        mv, mt, lg, ys, ep, module, secs = run_one(items)
        b = mt["burn_vs_rest"]
        print("  fold %d: test acc %.4f · BURN R=%.3f P=%.3f (n_burn %d, tp %d fp %d fn %d) · ep %d · %ds"
              % (i + 1, mt["accuracy"], b["recall"], b["precision"], mt["per_class"]["cane_burn"]["n"], b["tp"], b["fp"], b["fn"], ep, secs))
        mt["by_camera"] = by_camera(items["test"], lg, ys)
        pb = lg.softmax(1)[:, BURN].tolist()
        mt["preds"] = [[r[0].name, int(ys[i]), int(lg[i].argmax()), round(pb[i], 5), r[3], r[2]] for i, r in enumerate(items["test"])]
        mt["by_group"] = by_group(items["test"], lg, ys)
        mt["baselines"] = baselines(items["test"], ys)
        mt["conflicts"] = conflict_share(items["test"], lg, ys)
        error_grid(items["test"], lg, ys, OUT / ("%s_fold%d_errors.jpg" % (a.tag, i + 1)))
        save_ckpt(module, "%s_fold%d" % (a.tag, i + 1),                      # เดิมโหมด cv ไม่เซฟน้ำหนักเลย ทิ้ง checkpoint ฟรีทุกรอบ
                  dict(mode="cv-fold", fold=i + 1, test_days=te, valid_days=va,
                       train_days=[d for f in FOLDS for d in f if f != te and f != va],
                       best_epoch=ep, valid_burn_f1=mv["burn_vs_rest"]["f1"]))
        rows.append(dict(fold=i + 1, test_days=te, valid_days=va, best_epoch=ep, train_s=secs, valid=mv, test=mt))
        del module; torch.cuda.empty_cache()
    R = np.array([r["test"]["burn_vs_rest"]["recall"] for r in rows], float)
    P = np.array([r["test"]["burn_vs_rest"]["precision"] for r in rows], float)
    A = np.array([r["test"]["accuracy"] for r in rows], float)
    print("\n=== สรุป %d fold (test = วันที่ไม่เคยเห็น) ===" % len(rows))
    print("%-14s %8s %8s %8s %8s" % ("fold(test)", "n_burn", "acc", "burn R", "burn P"))
    for r in rows:
        print("%-14s %8d %8.4f %8.3f %8.3f" % ("+".join(d[4:] for d in r["test_days"]), r["test"]["per_class"]["cane_burn"]["n"],
                                               r["test"]["accuracy"], r["test"]["burn_vs_rest"]["recall"], r["test"]["burn_vs_rest"]["precision"]))
    print("%-14s %8s %8.4f %8.3f %8.3f" % ("เฉลี่ย", "", A.mean(), R.mean(), P.mean()))
    print("%-14s %8s %8.4f %8.3f %8.3f" % ("ส่วนเบี่ยงเบน", "", A.std(ddof=1), R.std(ddof=1), P.std(ddof=1)))
    print("%-14s %8s %8.4f %8.3f %8.3f" % ("ต่ำสุด", "", A.min(), R.min(), P.min()))
    json.dump(dict(tag=a.tag, arch=a.arch, size=a.size, bs=a.bs, lr=a.lr, epochs=a.epochs, seed=a.seed, folds=FOLDS,
                   summary=dict(recall_mean=round(float(R.mean()), 4), recall_sd=round(float(R.std(ddof=1)), 4), recall_min=round(float(R.min()), 4),
                                precision_mean=round(float(P.mean()), 4), precision_sd=round(float(P.std(ddof=1)), 4), precision_min=round(float(P.min()), 4),
                                acc_mean=round(float(A.mean()), 4), acc_sd=round(float(A.std(ddof=1)), 4)), rows=rows),
              open(HERE / (a.tag + "_cv.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", a.tag + "_cv")


if __name__ == "__main__":
    main()
