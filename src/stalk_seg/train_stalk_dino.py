# -*- coding: utf-8 -*-
"""stalk_seg dino — DINOv2 แช่แข็ง + หัว 1x1 conv 2 ชั้น แยกพิกเซล 3 คลาส (0 พื้น/อื่น · 1 ลำสด · 2 ลำไหม้)
เหตุผลที่เปลี่ยนจาก LR-ASPP: fine-tune ทั้งตัวบน 24 ใบ = จำสไตล์ generator (v2 ตกจาก MAE 8 ในบ้านเป็น 17 ข้ามชุด)
DINOv2 เทรน self-supervised บนภาพจริงหลากหลาย feature ทนข้ามโดเมนกว่า และเทรนแค่หัวเล็ก ๆ จึง overfit น้อย
รัน: python train_stalk_dino.py --labels ../../../gen_images/burnt_mixed/labels [--model facebook/dinov2-base]
hyperparameter ทั้งหมดตั้งจาก hold-out ของชุดแรกเท่านั้น ห้ามปรับด้วยตัวเลข g2"""
import re, json, time, argparse, random
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--size", type=int, default=518)          # 518 = 37 patch x 14
ap.add_argument("--model", default="facebook/dinov2-small"); ap.add_argument("--layers", type=int, default=2, help="concat hidden state กี่ชั้นสุดท้าย")
ap.add_argument("--hid", type=int, default=256); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="stalk_seg_dino_s")
ap.add_argument("--labels", default=None); ap.add_argument("--src", default=None)
ap.add_argument("--add-train", default="")
ap.add_argument("--unfreeze", type=int, default=0, help="ปลดล็อกกี่บล็อกท้ายของ backbone (lr = lr/10) · 0 = แช่แข็งทั้งตัว")
ap.add_argument("--scale-lo", type=float, default=0.7, help="ขอบล่างของ random-scale crop ตอน augment (0.7 = สูตรเดิม · 0.4 = ทนสเกลกว้างขึ้น)")
ap.add_argument("--add-ratio", type=float, default=1.0, help="จำนวนตัวอย่างจากชุดเสริมต่อ 1 ตัวอย่างของ g1 ในแต่ละ epoch (1.0 = 1:1 · 3.0 = 1:3) สุ่มใหม่ทุก epoch")
ap.add_argument("--add-dirs", default="", help="ชุดเสริม (เช่น synth_c1) รูปแบบ imgdir:labeldir[:nocrop] คั่นหลายชุดด้วย , — label เป็น PNG 0/1/2/255 ชื่อเดียวกับภาพ")
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
HERE = Path(__file__).resolve().parent
SRC = Path(a.src) if a.src else HERE.parent.parent.parent / "gen_images" / "burnt_mixed"
OUT = HERE / (a.tag + "_out"); OUT.mkdir(exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRIGHT_V, BRIGHT_S, DARK_V, DARK_S, TEX_MIN = 90, 45, 60, 70, 20
BAY = (0.42, 0.60, 0.30, 0.88); CROP = (0.36, 0.66, 0.24, 0.92)
HOLD = {"mix0_A_night_nodust_1", "mix10_B_sun_dust_1", "mix20_A_cloud_dust_1", "mix30_B_night_nodust_1", "mix50_B_cloud_nodust_1", "mix100_B_cloud_nodust_1"}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def pseudo_label(f):
    H, W = f.shape[:2]; hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV); S, V = hsv[..., 1], hsv[..., 2]
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32); tex = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F, 3)), -1, (9, 9))
    bay = np.zeros((H, W), bool); bay[int(BAY[2] * H):int(BAY[3] * H), int(BAY[0] * W):int(BAY[1] * W)] = True
    bright = (S >= BRIGHT_S) & (V > BRIGHT_V); dark = (V < DARK_V) & (S < DARK_S) & (tex > TEX_MIN) & ~bright
    lab = np.zeros((H, W), np.uint8); lab[bay] = 255; lab[bay & bright] = 1; lab[bay & dark] = 2
    return lab


def crop(img, lab=None):
    H, W = img.shape[:2]; x0, x1, y0, y1 = int(CROP[0] * W), int(CROP[1] * W), int(CROP[2] * H), int(CROP[3] * H)
    im = cv2.resize(img[y0:y1, x0:x1], (a.size, a.size), interpolation=cv2.INTER_AREA)
    return im if lab is None else (im, cv2.resize(lab[y0:y1, x0:x1], (a.size, a.size), interpolation=cv2.INTER_NEAREST))


LB = Path(a.labels) if a.labels else None
n_hand = 0


def load_label(p, f):
    global n_hand
    if LB is not None:
        q = LB / (p.stem + ".png")
        if q.exists():
            lb = cv2.imread(str(q), cv2.IMREAD_GRAYSCALE)
            if lb is not None and lb.shape == f.shape[:2]:
                n_hand += 1; return lb
    return pseudo_label(f)


files = sorted(p for p in SRC.glob("mix*_*.png") if re.match(r"mix\d+_", p.stem))
data = {}
for p in files:
    f = cv2.imread(str(p)); im, lab = crop(f, load_label(p, f)); rule = crop(f, pseudo_label(f))[1]
    data[p.stem] = (im, lab, int(re.match(r"mix(\d+)", p.stem).group(1)), rule)
add = {}
for n in [x for x in a.add_train.split(",") if x]:
    q = SRC / n; f = cv2.imread(str(q))
    if f is None or LB is None or not (LB / (q.stem + ".png")).exists():
        print("ข้าม add-train:", n); continue
    add[q.stem] = crop(f, load_label(q, f))
for spec in [x for x in a.add_dirs.split(",") if x]:
    parts = spec.split(":")
    idir, ldir = Path(parts[0]), Path(parts[1])
    nocrop = len(parts) > 2 and parts[2] == "nocrop"
    n0 = len(add)
    for q in sorted(list(idir.glob("*.png")) + list(idir.glob("*.jpg"))):
        lp = ldir / (q.stem + ".png")
        if not lp.exists(): continue
        f = cv2.imread(str(q)); lb = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        if f is None or lb is None or lb.shape != f.shape[:2]: continue
        if nocrop:
            im = cv2.resize(f, (a.size, a.size), interpolation=cv2.INTER_AREA)
            lab2 = cv2.resize(lb, (a.size, a.size), interpolation=cv2.INTER_NEAREST)
        else:
            im, lab2 = crop(f, lb)
        add["%s/%s" % (idir.name, q.stem)] = (im, lab2)
    print("ชุดเสริม %s: +%d ภาพ (crop=%s)" % (idir.name, len(add) - n0, not nocrop))

base = [k for k in data if k not in HOLD]                            # g1 ชุดจริง
extra = list(add)                                                    # ชุดเสริม (mixed_topdown_01 / synth_c1)
hold = [k for k in data if k in HOLD]
pool = {**{k: data[k][:2] for k in data}, **add}
n_extra_ep = min(len(extra), int(round(a.add_ratio * len(base)))) if extra else 0
print("labels แก้มือ %d/%d · g1 %d ใบ + ชุดเสริม %d ใบ (สุ่ม %d ใบ/epoch = 1:%.0f) hold %d · %s"
      % (n_hand, len(files), len(base), len(extra), n_extra_ep, a.add_ratio, len(hold), dev))


def augment(im, lab, rng):
    if rng.random() < 0.5: im, lab = im[:, ::-1], lab[:, ::-1]
    if rng.random() < 0.5: im, lab = im[::-1], lab[::-1]
    s = rng.uniform(a.scale_lo, 1.0); H, W = lab.shape; h, w = int(H * s), int(W * s)
    y, x = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
    im = cv2.resize(np.ascontiguousarray(im[y:y + h, x:x + w]), (W, H), interpolation=cv2.INTER_LINEAR)
    lab = cv2.resize(np.ascontiguousarray(lab[y:y + h, x:x + w]), (W, H), interpolation=cv2.INTER_NEAREST)
    v = im.astype(np.float32)
    v *= np.array([rng.uniform(0.75, 1.25), 1.0, rng.uniform(0.75, 1.25)], np.float32)   # BGR: เลียนไฟส้มกลางคืน / แสงเย็น
    v = np.clip(v * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25), 0, 255)
    v = 255.0 * np.power(v / 255.0, rng.uniform(0.5, 1.8))                                # gamma: เงาลึก/แดดจัด
    im = np.clip(v, 0, 255).astype(np.uint8)
    if rng.random() < 0.3: im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3: im = cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 80))])[1], 1)
    return im, lab


bb = AutoModel.from_pretrained(a.model).to(dev).eval()
for q in bb.parameters(): q.requires_grad_(False)
tune = []
if a.unfreeze > 0:
    blocks = bb.encoder.layer
    for blk in blocks[-a.unfreeze:]:
        for q in blk.parameters(): q.requires_grad_(True); tune.append(q)
    bb.train()
    print("ปลดล็อก %d บล็อกท้าย (%d พารามิเตอร์) lr = lr/10" % (a.unfreeze, sum(q.numel() for q in tune)))
DIM = bb.config.hidden_size * a.layers
G = a.size // bb.config.patch_size
head = nn.Sequential(nn.Conv2d(DIM, a.hid, 1), nn.GELU(), nn.Conv2d(a.hid, 3, 1)).to(dev)
print("%s: dim %d x %d = %d · grid %dx%d · หัว %.0fk พารามิเตอร์" % (a.model, bb.config.hidden_size, a.layers, DIM, G, G, sum(q.numel() for q in head.parameters()) / 1e3))


def feats(ims):
    x = torch.from_numpy(np.stack([im[..., ::-1].copy() for im in ims])).to(dev).permute(0, 3, 1, 2).float() / 255.0
    ctx = torch.enable_grad() if a.unfreeze > 0 else torch.no_grad()
    with ctx:
        h = bb((x - MEAN) / STD, output_hidden_states=True).hidden_states
        z = torch.cat([h[-i][:, 1:] for i in range(1, a.layers + 1)], -1)      # ทิ้ง CLS token
    return z.permute(0, 2, 1).reshape(len(ims), DIM, G, G)


def logits(ims, size):
    return F.interpolate(head(feats(ims)), size=(size, size), mode="bilinear", align_corners=False)


groups = [dict(params=list(head.parameters()), lr=a.lr)] + ([dict(params=tune, lr=a.lr / 10)] if tune else [])
opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=1e-4)
steps = a.epochs * max(1, (len(base) + n_extra_ep) // a.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps)
crit = nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor([0.5, 1.0, 1.5], device=dev))
rng = np.random.default_rng(a.seed); t0 = time.time()
for ep in range(a.epochs):
    head.train()
    train = base + (random.sample(extra, n_extra_ep) if n_extra_ep else [])   # สุ่มชุดเสริมใหม่ทุก epoch ให้สัดส่วนคงที่
    random.shuffle(train); tot = 0.0
    for i in range(0, len(train) - a.bs + 1, a.bs):
        b = [augment(*pool[k], rng) for k in train[i:i + a.bs]]
        y = torch.from_numpy(np.stack([v[1] for v in b]).astype(np.int64)).to(dev)
        loss = crit(logits([v[0] for v in b], a.size), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
    if ep % 10 == 9 or ep == a.epochs - 1:
        print("ep %3d loss %.3f  %.0fs" % (ep + 1, tot / max(1, len(train) // a.bs), time.time() - t0), flush=True)


@torch.no_grad()
def predict(im):
    head.eval(); bb.eval()
    p = logits([im], a.size).softmax(1)
    p = p + torch.flip(logits([np.ascontiguousarray(im[:, ::-1])], a.size).softmax(1), [3])   # TTA flip
    return p.argmax(1)[0].cpu().numpy().astype(np.uint8)


def fr(m):
    n1, n2 = int((m == 1).sum()), int((m == 2).sum()); return 100.0 * n2 / max(n1 + n2, 1), n1, n2


rows = []
for k, (im, lab, ordered, rule) in data.items():
    pred = predict(im); mf, n1, n2 = fr(pred)
    rows.append(dict(file=k, split="hold" if k in HOLD else "train", ordered=ordered,
                     truth_burnt=round(fr(lab)[0], 1), model_burnt=round(mf, 1), rule_burnt=round(fr(rule)[0], 1), fresh_px=n1, burnt_px=n2))
    v = im.copy(); v[pred == 1] = (0.5 * v[pred == 1] + [0, 120, 120]).astype(np.uint8); v[pred == 2] = (0.4 * v[pred == 2] + [0, 0, 160]).astype(np.uint8)
    cv2.putText(v, "%s ordered %d truth %.0f model %.0f" % (k, ordered, rows[-1]["truth_burnt"], mf), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / (k + ".jpg")), v, [cv2.IMWRITE_JPEG_QUALITY, 80])


def stats(rs, key, ref):
    x = np.array([r[ref] for r in rs], float); y = np.array([r[key] for r in rs], float)
    return dict(n=len(rs), corr=round(float(np.corrcoef(x, y)[0, 1]), 3), mae=round(float(np.mean(np.abs(x - y))), 1))


KEYS = ("model_burnt", "rule_burnt")
summary = {sp: {ref: {k: stats([r for r in rows if r["split"] == sp], k, ref) for k in KEYS} for ref in ("truth_burnt", "ordered")} for sp in ("hold", "train")}
print(json.dumps(summary["hold"], indent=1))
for r in rows:
    if r["split"] == "hold":
        print("  HOLD %-28s ordered %3d  truth %5.1f  model %5.1f  rule %5.1f" % (r["file"], r["ordered"], r["truth_burnt"], r["model_burnt"], r["rule_burnt"]))
torch.save(dict(head=head.state_dict(), model=a.model, layers=a.layers, hid=a.hid, size=a.size,
                backbone=bb.state_dict() if a.unfreeze > 0 else None, unfreeze=a.unfreeze), HERE / (a.tag + ".pt"))
json.dump(dict(tag=a.tag, backbone=a.model, frozen=True, layers=a.layers, hid=a.hid, size=a.size, epochs=a.epochs, bs=a.bs, lr=a.lr, seed=a.seed,
               labels=a.labels, scale_lo=a.scale_lo, unfreeze=a.unfreeze, hand_labels=n_hand, hold=sorted(HOLD), add_dirs=a.add_dirs, add_ratio=a.add_ratio, n_extra=len(add), n_extra_per_epoch=n_extra_ep, add_train=sorted(add), train_s=round(time.time() - t0), summary=summary, rows=rows),
          open(HERE / (a.tag + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", a.tag)
