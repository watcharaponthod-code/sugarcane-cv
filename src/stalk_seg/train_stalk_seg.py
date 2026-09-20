# -*- coding: utf-8 -*-
"""stalk_seg — LR-ASPP MobileNetV3 (เหมือน v3.1) แยกพิกเซล 3 คลาส: 0 พื้น/อื่น · 1 ลำสด · 2 ลำไหม้ บนภาพ AI 30 ใบ
label = pseudo-label จากกฎ CV ใน stalk_mix.py (สด = มีสี, ไหม้ = ดำมีลาย) พิกเซลที่กฎตัดสินไม่ได้ = ignore (255)
hold-out 6 ใบ (1 ต่อระดับ % ครอบ night/dust) ที่เหลือ 24 เทรน · วัด burnt% = ไหม้/(ไหม้+สด) เทียบ % ที่สั่ง เทียบกับกฎ CV เดิม
รัน: python train_stalk_seg.py [--epochs 40] → stalk_seg_v0.pt, stalk_seg_v0.json, stalk_seg_out/*.jpg
เครื่อง GTX 1060: 24 ภาพ 512² ≈ 3 s/epoch"""
import re, json, time, argparse, random
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--size", type=int, default=512); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="stalk_seg_v0")
ap.add_argument("--labels", default=None, help="โฟลเดอร์ label PNG (0/1/2/255) ต่อภาพ; ไม่ระบุ = pseudo-label จากกฎเหมือน v0")
ap.add_argument("--src", default=None, help="โฟลเดอร์ภาพเทรน (ดีฟอลต์ gen_images/burnt_mixed)")
ap.add_argument("--extra", default="mixed_topdown_01.png", help="ภาพนอกชุด คั่นด้วย , (เทสต์ข้าม generator)")
ap.add_argument("--add-train", default="", help="ภาพนอกชุด mix* ที่ให้เข้าชุดเทรนด้วย (คั่น ,) ต้องมี label ใน --labels; ไม่เข้าสถิติ hold/train")
ap.add_argument("--eval-dir", default=None, help="โฟลเดอร์ภาพชุดที่สอง: รันโมเดล+กฎแล้วรายงานแยก ไม่เทรน")
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
HERE = Path(__file__).resolve().parent; SRC = Path(a.src) if a.src else HERE.parent.parent.parent / "gen_images" / "burnt_mixed"; OUT = HERE / "stalk_seg_out"; OUT.mkdir(exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRIGHT_V, BRIGHT_S, DARK_V, DARK_S, TEX_MIN = 90, 45, 60, 70, 20        # = stalk_mix.py
BAY = (0.42, 0.60, 0.30, 0.88); CROP = (0.36, 0.66, 0.24, 0.92)          # กฎใช้ BAY แคบ; อินพุตโมเดลใช้ CROP กว้างกว่าให้เห็นบริบท
HOLD = {"mix0_A_night_nodust_1", "mix10_B_sun_dust_1", "mix20_A_cloud_dust_1", "mix30_B_night_nodust_1", "mix50_B_cloud_nodust_1", "mix100_B_cloud_nodust_1"}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def pseudo_label(f):
    H, W = f.shape[:2]; hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV); S, V = hsv[..., 1], hsv[..., 2]
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32); tex = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F, 3)), -1, (9, 9))
    bay = np.zeros((H, W), bool); bay[int(BAY[2] * H):int(BAY[3] * H), int(BAY[0] * W):int(BAY[1] * W)] = True
    bright = (S >= BRIGHT_S) & (V > BRIGHT_V); dark = (V < DARK_V) & (S < DARK_S) & (tex > TEX_MIN) & ~bright
    lab = np.zeros((H, W), np.uint8)                       # นอก bay = พื้น (0)
    lab[bay] = 255                                          # ใน bay ที่ตัดสินไม่ได้ = ignore
    lab[bay & bright] = 1; lab[bay & dark] = 2
    return lab


def crop(img, lab=None):
    H, W = img.shape[:2]; x0, x1, y0, y1 = int(CROP[0] * W), int(CROP[1] * W), int(CROP[2] * H), int(CROP[3] * H)
    im = cv2.resize(img[y0:y1, x0:x1], (a.size, a.size), interpolation=cv2.INTER_AREA)
    if lab is None: return im
    return im, cv2.resize(lab[y0:y1, x0:x1], (a.size, a.size), interpolation=cv2.INTER_NEAREST)


LB = Path(a.labels) if a.labels else None
n_hand = 0


def load_label(p, f):
    """label แก้มือถ้ามี ไม่งั้น fallback = pseudo-label จากกฎ (เท่ากับ v0)"""
    global n_hand
    if LB is not None:
        q = LB / (p.stem + ".png")
        if q.exists():
            lb = cv2.imread(str(q), cv2.IMREAD_GRAYSCALE)
            if lb is not None and lb.shape == f.shape[:2]:
                n_hand += 1
                return lb
    return pseudo_label(f)


files = sorted(p for p in SRC.glob("mix*_*.png") if re.match(r"mix\d+_", p.stem))
data = {}
for p in files:
    f = cv2.imread(str(p)); im, lab = crop(f, load_label(p, f)); rule = crop(f, pseudo_label(f))[1]
    data[p.stem] = (im, lab, int(re.match(r"mix(\d+)", p.stem).group(1)), rule)
print(f"labels: แก้มือ {n_hand} / {len(files)} ภาพ (ที่เหลือ fallback = กฎ)")
add = {}
for n in [x for x in a.add_train.split(",") if x]:
    q = SRC / n; f = cv2.imread(str(q))
    if f is None: print("ข้าม add-train (อ่านไม่ได้):", n); continue
    lb = load_label(q, f)
    if LB is None or not (LB / (q.stem + ".png")).exists(): print("ข้าม add-train (ไม่มี label):", n); continue
    add[q.stem] = crop(f, lb)
train = [k for k in data if k not in HOLD]; hold = [k for k in data if k in HOLD]
pool = {**{k: data[k][:2] for k in data}, **add}                     # ชุดที่ป้อนเข้า loop เทรน (add ไม่เข้าสถิติ)
train += list(add)
print(f"train {len(train)} (+{len(add)} นอกชุด mix*) hold {len(hold)} device {dev}")


def to_x(ims):                                                      # list BGR uint8 -> normalized RGB tensor
    x = torch.from_numpy(np.stack([im[..., ::-1].copy() for im in ims])).to(dev).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD


def augment(im, lab, rng):
    if rng.random() < 0.5: im, lab = im[:, ::-1], lab[:, ::-1]
    if rng.random() < 0.5: im, lab = im[::-1], lab[::-1]
    s = rng.uniform(0.7, 1.0); H, W = lab.shape; h, w = int(H * s), int(W * s); y, x = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
    im = cv2.resize(np.ascontiguousarray(im[y:y + h, x:x + w]), (W, H), interpolation=cv2.INTER_LINEAR); lab = cv2.resize(np.ascontiguousarray(lab[y:y + h, x:x + w]), (W, H), interpolation=cv2.INTER_NEAREST)
    im = np.clip(im.astype(np.float32) * rng.uniform(0.6, 1.4) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)   # gain/offset: กลางคืน/แดดจัด
    if rng.random() < 0.3: im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.5, 1.5))
    return im, lab


model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1, num_classes=3).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
steps = a.epochs * max(1, len(train) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
crit = nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor([0.5, 1.0, 1.5], device=dev))   # ไหม้มีพิกเซลน้อย
rng = np.random.default_rng(a.seed); t0 = time.time()
for ep in range(a.epochs):
    model.train(); random.shuffle(train); tot = 0.0
    for i in range(0, len(train) - a.bs + 1, a.bs):
        batch = [augment(*pool[k], rng) for k in train[i:i + a.bs]]
        x = to_x([b[0] for b in batch]); y = torch.from_numpy(np.stack([b[1] for b in batch]).astype(np.int64)).to(dev)
        loss = crit(model(x)["out"], y); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
    if ep % 5 == 4 or ep == a.epochs - 1: print(f"ep {ep + 1:3d} loss {tot / max(1, len(train) // a.bs):.3f}  {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def predict(im):
    model.eval(); p = model(to_x([im]))["out"].argmax(1)[0].cpu().numpy().astype(np.uint8); return p


def frac_from(mask, one, two):
    n1, n2 = int((mask == one).sum()), int((mask == two).sum()); return 100.0 * n2 / max(n1 + n2, 1), n1, n2


rows = []
for k, (im, lab, ordered, rule) in data.items():
    pred = predict(im); m_frac, n1, n2 = frac_from(pred, 1, 2); r_frac, _, _ = frac_from(rule, 1, 2); t_frac, _, _ = frac_from(lab, 1, 2)
    rows.append(dict(file=k, split="hold" if k in HOLD else "train", ordered=ordered, truth_burnt=round(t_frac, 1), model_burnt=round(m_frac, 1), rule_burnt=round(r_frac, 1), fresh_px=n1, burnt_px=n2))
    v = im.copy(); v[pred == 1] = (0.5 * v[pred == 1] + [0, 120, 120]).astype(np.uint8); v[pred == 2] = (0.4 * v[pred == 2] + [0, 0, 160]).astype(np.uint8)
    cv2.putText(v, f"{k} ordered {ordered} model {m_frac:.0f} rule {r_frac:.0f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"{k}.jpg"), v, [cv2.IMWRITE_JPEG_QUALITY, 80])


def stats(rs, key, ref="truth_burnt"):
    x = np.array([r[ref] for r in rs], float); y = np.array([r[key] for r in rs], float)
    return dict(n=len(rs), corr=round(float(np.corrcoef(x, y)[0, 1]), 3), mae=round(float(np.mean(np.abs(x - y))), 1))


KEYS = ("model_burnt", "rule_burnt")
summary = {sp: {ref: {key: stats([r for r in rows if r["split"] == sp], key, ref) for key in KEYS} for ref in ("truth_burnt", "ordered")} for sp in ("hold", "train")}
summary["all"] = {ref: {key: stats(rows, key, ref) for key in KEYS} for ref in ("truth_burnt", "ordered")}
print(json.dumps(summary, indent=1))
for r in rows:
    if r["split"] == "hold": print(f"  HOLD {r['file']:28s} ordered {r['ordered']:3d}  truth {r['truth_burnt']:5.1f}  model {r['model_burnt']:5.1f}  rule {r['rule_burnt']:5.1f}")


def run_extra(paths, tag):
    """ภาพนอกชุดเทรน: ไม่มี label → รายงานเฉพาะ model vs rule"""
    out = []
    for q in paths:
        f = cv2.imread(str(q))
        if f is None: continue
        im = crop(f); pred = predict(im); mf, _, _ = frac_from(pred, 1, 2); rf, _, _ = frac_from(crop(f, pseudo_label(f))[1], 1, 2)
        lbp = LB / (q.stem + ".png") if LB else None
        tf = frac_from(crop(f, cv2.imread(str(lbp), cv2.IMREAD_GRAYSCALE))[1], 1, 2)[0] if lbp and lbp.exists() else None
        out.append(dict(file=q.name, model_burnt=round(mf, 1), rule_burnt=round(rf, 1), label_burnt=None if tf is None else round(tf, 1)))
        print(f"  {tag} {q.name:32s} model {mf:5.1f}  rule {rf:5.1f}  label {'-' if tf is None else f'{tf:5.1f}'}")
        v = im.copy(); v[pred == 1] = (0.5 * v[pred == 1] + [0, 120, 120]).astype(np.uint8); v[pred == 2] = (0.4 * v[pred == 2] + [0, 0, 160]).astype(np.uint8)
        cv2.putText(v, f"{q.stem} model {mf:.0f} rule {rf:.0f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"{a.tag}_{tag}_{q.stem}.jpg"), v, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return out


extra = run_extra([SRC / n for n in a.extra.split(",") if n], "EXTRA") if a.extra else []
if add: print(f"  (หมายเหตุ: {', '.join(add)} อยู่ในชุดเทรน ตัวเลขข้างบนจึงไม่ใช่ blind test)")
g2 = []
if a.eval_dir and Path(a.eval_dir).is_dir():
    g2 = run_extra(sorted(Path(a.eval_dir).glob("*.png")) + sorted(Path(a.eval_dir).glob("*.jpg")), "G2")
    print(f"eval-dir {a.eval_dir}: {len(g2)} ภาพ")
torch.save(model.state_dict(), HERE / f"{a.tag}.pt")
json.dump(dict(tag=a.tag, model="lraspp_mobilenet_v3_large", num_classes=3, size=a.size, crop=CROP, epochs=a.epochs, bs=a.bs, lr=a.lr, seed=a.seed, labels=a.labels, hand_labels=n_hand, hold=sorted(HOLD), add_train=sorted(add), train_s=round(time.time() - t0), summary=summary, rows=rows, extra=extra, g2=g2),
          open(HERE / f"{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", a.tag)
