# -*- coding: utf-8 -*-
"""Kaggle: dirty-area level — เทียบ backbone แช่แข็ง (effnet-b0 / DINOv2-S / DINOv2-B) × 3 ระดับ / 2 ระดับ
CV เหมือน trash_train.py ทุกอย่าง: raster frac · ตัด EXCLUDE_STEMS · แบ่งตามกลุ่มรถ · MPK00 กันเป็นข้ามกล้อง · เฉลี่ย 10 init
เพิ่ม: เฉลี่ย 5 การแบ่ง fold (split seed 0-4) · เขียน results.json + .pt (ตัวที่ชนะ baseline > 10 จุด) — log เข้า MLflow ทำจากเครื่องด้วย log_kaggle_dirty.py
ห้ามเรียก MLflow client บน Kaggle: set_experiment แล้ว kernel ถูก CANCEL ใน ~15 วิ (probe ยืนยัน 2026-09-16) · DATA_URL จาก env/Kaggle Secret
DATA_URL = presigned URL ของ datasets/cane_trash_pilot.zip ใน Railway bucket models-dev (อายุ 7 วัน)
sanity: effb0 3 ระดับ split-seed 0 ต้องได้ ~0.534 เท่ากับ trash_train.py --cpu
"""
import os, sys, json, zipfile, subprocess, urllib.request, io
from collections import Counter
from pathlib import Path

import faulthandler, threading, time; faulthandler.enable()
# heartbeat ให้เห็นว่า kernel ยังไม่ตาย (สาเหตุ CANCEL จริงคือ MLflow client ไม่ใช่ความเงียบ)
threading.Thread(target=lambda: [print("hb", int(time.time()), flush=True) or time.sleep(10) for _ in iter(int, 1)], daemon=True).start()
import numpy as np, cv2, torch, torch.nn as nn
print("numpy", np.__version__, "torch", torch.__version__, flush=True)

def secret(k):
    # env ก่อน แล้วค่อย Kaggle Secret (Add-ons → Secrets: DATA_URL)
    if os.environ.get(k):
        return os.environ[k]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(k)
    except Exception:
        return None


W = Path("/kaggle/working")
hit = sorted(Path("/kaggle/input").rglob("meta/stats.json"))       # dataset แนบมา (mount ซ้อนได้หลายชั้น)
if hit:
    D = hit[0].parent.parent
else:
    D = Path("/tmp/data")          # ไม่ใช่ /kaggle/working — ไม่งั้น kernels output ดาวน์โหลดภาพกลับมาทั้งชุด
    if not D.exists():
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(secret("DATA_URL")).read())).extractall(D)
print("data:", D)

EXCLUDE_STEMS = {"20230117-012653_MPDC00-jpg_cane_jpg"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pick_device():
    # ponytail: P100 + torch cu128 ไม่มี kernel sm_60 → ลองจริงก่อน พังก็ CPU (ภาพ 154 ใบ CPU ก็พอ)
    if torch.cuda.is_available():
        try:
            torch.nn.functional.conv2d(torch.ones(1, 1, 3, 3, device="cuda"), torch.ones(1, 1, 1, 1, device="cuda"))
            return torch.device("cuda")
        except Exception as e:
            print("GPU ใช้ไม่ได้ → CPU:", str(e)[:80])
    return torch.device("cpu")


def load_rows():
    st = json.loads((D / "meta/stats.json").read_text(encoding="utf-8"))
    ras = json.loads((D / "meta/frac_raster.json").read_text(encoding="utf-8"))["by_file"]
    rows = [r for r in st["rows"] if r["frac_dirty"] is not None and r["stem"] not in EXCLUDE_STEMS and r["file"] in ras]
    g2i = json.loads((D / "meta/truck_groups.json").read_text(encoding="utf-8"))["image_to_group"]
    stem2grp = {}
    for k, v in g2i.items():
        stem2grp.setdefault(k.split(".rf.")[0], v)
    for r in rows:
        r["frac_dirty"] = ras[r["file"]]
        r["group"] = stem2grp.get(r["stem"], "stem:" + r["stem"])
    return rows


def backbone(name, dev):
    if name == "effb0_384":
        import torchvision.models as tvm
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1).to(dev).eval()
        return 384, lambda x: m.avgpool(m.features(x)).flatten(1)
    arch = {"dinov2_s_448": "dinov2_vits14", "dinov2_b_448": "dinov2_vitb14"}[name]
    m = torch.hub.load("facebookresearch/dinov2", arch).to(dev).eval()
    def f(x):
        o = m.forward_features(x)                       # CLS + mean patch token = มาตรฐาน linear-probe ของ DINOv2
        return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], 1)
    return 448, f


@torch.no_grad()
def features(rows, name, dev, bs=8):
    size, fn = backbone(name, dev)
    out = []
    for i in range(0, len(rows), bs):
        xs = []
        for r in rows[i:i + bs]:
            im = cv2.resize(cv2.imread(str(D / "rf_trash" / r["split"] / r["file"])), (size, size), interpolation=cv2.INTER_AREA)
            xs.append(torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).permute(2, 0, 1).float() / 255.0)
        x = (torch.stack(xs).to(dev) - MEAN.to(dev)) / STD.to(dev)
        out.append(fn(x).float().cpu().numpy())
        print(f"  {name} {min(i + bs, len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(out)


def fit_logistic(X, y, n_cls, epochs=300, lr=0.05, wd=1e-2, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    w = nn.Linear(Xt.shape[1], n_cls)
    opt = torch.optim.AdamW(w.parameters(), lr=lr, weight_decay=wd)
    cnt = np.bincount(y, minlength=n_cls).astype(np.float32)
    cw = torch.tensor(cnt.sum() / (n_cls * np.maximum(cnt, 1)), dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(); nn.functional.cross_entropy(w(Xt), yt, weight=cw).backward(); opt.step()
    return w, mu, sd


def predict(head, Z):
    w, mu, sd = head
    with torch.no_grad():
        return w(torch.tensor((Z - mu) / sd, dtype=torch.float32)).argmax(1).numpy()


def macro_f1(y, p, n_cls):
    fs = []
    for c in range(n_cls):
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum()); fn = int(((p != c) & (y == c)).sum())
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


def cv(X, y, cam, grp, n_cls, split_seed, folds=5, inits=10):
    hold = cam == "MPK00"
    Xm, ym, gm = X[~hold], y[~hold], grp[~hold]
    groups = sorted(set(gm)); rng = np.random.RandomState(split_seed); rng.shuffle(groups)
    accs, f1s, base = [], [], []
    for te_g in [set(groups[i::folds]) for i in range(folds)]:
        te = np.array([g in te_g for g in gm])
        aa, ff = [], []
        for si in range(inits):
            p = predict(fit_logistic(Xm[~te], ym[~te], n_cls, seed=si), Xm[te])
            aa.append(float((p == ym[te]).mean())); ff.append(macro_f1(ym[te], p, n_cls))
        accs.append(np.mean(aa)); f1s.append(np.mean(ff))
        base.append(float((ym[te] == np.bincount(ym[~te], minlength=n_cls).argmax()).mean()))
    cam_base = float(np.mean(ym == np.bincount(ym, minlength=n_cls).argmax()))   # = ทายคลาสเด่นของ MPDC00 ทุกใบ
    return dict(acc=float(np.mean(accs)), acc_sd=float(np.std(accs)), f1=float(np.mean(f1s)),
                base=max(float(np.mean(base)), cam_base))


def main():
    dev = pick_device()
    rows = load_rows()
    cam = np.array([r["cam"] for r in rows]); grp = np.array([r["group"] for r in rows])
    frac = np.array([r["frac_dirty"] for r in rows])
    targets = {"3lvl": (3, np.digitize(frac, [1 / 3, 2 / 3])), "2lvl": (2, (frac >= 0.5).astype(int))}
    print(f"{len(rows)} ภาพ · ต้นฉบับ {len({r['stem'] for r in rows})} · กลุ่มรถ {len(set(grp))} · {dev}", dict(Counter(cam)))
    results = []
    for bb in ("effb0_384", "dinov2_s_448", "dinov2_b_448"):
        X = features(rows, bb, dev)
        for tname, (n_cls, y) in targets.items():
            per = []
            for s in range(5):
                per.append(cv(X, y, cam, grp, n_cls, s))
                print(f"  cv {bb} {tname} split{s} acc {per[-1]['acc']:.3f} base {per[-1]['base']:.3f}", flush=True)
            r = dict(backbone=bb, target=tname, n_cls=n_cls,
                     acc=float(np.mean([p["acc"] for p in per])), acc_sd_folds=float(np.mean([p["acc_sd"] for p in per])),
                     f1=float(np.mean([p["f1"] for p in per])), baseline=float(np.mean([p["base"] for p in per])),
                     acc_split0=per[0]["acc"])
            r["margin"] = r["acc"] - r["baseline"]
            hold = cam == "MPK00"
            head_all = fit_logistic(X[~hold], y[~hold], n_cls, seed=0)
            px = predict(head_all, X[hold])
            r["xcam_acc"] = float((px == y[hold]).mean())
            r["xcam_baseline"] = float(np.bincount(y[hold], minlength=n_cls).max() / hold.sum())
            if r["margin"] > 0.10:
                w, mu, sd = head_all
                torch.save(dict(state=w.state_dict(), mu=mu, sd=sd, backbone=bb, n_cls=n_cls,
                                cuts=[1 / 3, 2 / 3] if n_cls == 3 else [0.5], metrics=r), W / f"dirty_{bb}_{tname}.pt")
            print(json.dumps(r, ensure_ascii=False))
            results.append(r)
    (W / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("DONE", len(results), "configs")

main()
