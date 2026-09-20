# -*- coding: utf-8 -*-
"""pilot: ทำนาย dirty-area fraction เป็น 3 ระดับ (น้อย/กลาง/มาก) จากภาพ CCTV

  python trash_train.py --cpu            # feature ค้าง backbone + หัว logistic · 5-fold ตามกลุ่มรถ
  python trash_train.py                  # ใช้ GPU ถ้าว่าง (เร็วกว่านิดเดียว งานนี้เล็ก)

**นี่คือ pilot 64 ภาพต้นฉบับ ไม่ใช่ผลิตภัณฑ์** รายงานได้แค่ "มีสัญญาณ / ไม่มีสัญญาณ" เทียบ baseline
ห้ามอ้างความแม่น · ต้องชนะทั้งสอง baseline ชัดเจนถึงจะคุ้มขอ label เพิ่มจากโรงงาน

ออกแบบตาม trash_audit.md: ทิ้ง split ของ Roboflow · แบ่งตามกลุ่มรถ · MPK00 เป็นชุดข้ามกล้องแยก
backbone แช่แข็ง เพราะ 64 ภาพ fine-tune ทั้งตัวคือการจำข้อสอบ
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import cv2, numpy as np, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SRC = ROOT / "gen_images" / "rf_trash"
OUT = HERE / "trash_audit_out"
SPLITS = ("train", "valid", "test")
# ตัดทิ้ง: สำเนากระจายข้าม train/valid + label สองมาตรฐาน (dirty ต่างกัน 0.494) — ดู trash_audit.md
EXCLUDE_STEMS = {"20230117-012653_MPDC00-jpg_cane_jpg"}
CUTS = (1 / 3, 2 / 3)
LEVELS = ["น้อย", "กลาง", "มาก"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def level_of(f):
    return 0 if f < CUTS[0] else (1 if f < CUTS[1] else 2)


def load_rows(frac_source="raster", drop_swallowed=False):
    """frac_source: raster = |dirty|/|dirty ∪ clean| จาก mask (ถูกต้อง) · coco = ผลรวม area (นับทับซ้อนซ้ำ)"""
    st = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
    rows = [r for r in st["rows"] if r["frac_dirty"] is not None and r["stem"] not in EXCLUDE_STEMS]
    if frac_source == "raster":
        rj = json.loads((OUT / "frac_raster.json").read_text(encoding="utf-8"))
        ras = rj["by_file"]
        if drop_swallowed:
            bad = set(rj.get("swallowed_clean_stems", []))
            rows = [r for r in rows if r["stem"] not in bad]
        rows = [r for r in rows if r["file"] in ras]
        for r in rows:
            r["frac_dirty"] = ras[r["file"]]
    g2i = json.loads((HERE / "rf_audit_out" / "truck_groups.json").read_text(encoding="utf-8"))["image_to_group"]
    stem2grp = {}
    for k, v in g2i.items():
        stem2grp.setdefault(k.split(".rf.")[0], v)
    for r in rows:
        r["group"] = stem2grp.get(r["stem"], "stem:" + r["stem"])   # ไม่รู้จักกลุ่ม = กลุ่มของตัวเอง
        r["y"] = level_of(r["frac_dirty"])
    return rows


@torch.no_grad()
def features(rows, dev, size=384, bs=8):
    cache = OUT / f"feat_effb0_{size}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        got = {k: v for k, v in zip(z["keys"], z["feat"])}
        if all(r["file"] in got for r in rows):
            print(f"ใช้ feature จาก {cache.name}")
            return np.stack([got[r["file"]] for r in rows])
    import torchvision.models as tvm
    m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1).to(dev).eval()
    feats = []
    for i in range(0, len(rows), bs):
        xs = []
        for r in rows[i:i + bs]:
            im = cv2.imread(str(SRC / r["split"] / r["file"]))
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
            xs.append(torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).permute(2, 0, 1).float() / 255.0)
        x = (torch.stack(xs).to(dev) - MEAN.to(dev)) / STD.to(dev)
        f = m.avgpool(m.features(x)).flatten(1)
        feats.append(f.cpu().numpy())
        print(f"  feature {min(i+bs, len(rows))}/{len(rows)}", end="\r")
    F = np.concatenate(feats)
    np.savez(cache, keys=np.array([r["file"] for r in rows]), feat=F)
    print(f"\nเขียน {cache.name}")
    return F


def fit_logistic(X, y, n_cls=3, epochs=300, lr=0.05, wd=1e-2, seed=0):
    """หัว logistic ตัวเดียว — ข้อมูล 64 ภาพ อะไรที่ใหญ่กว่านี้คือการจำข้อสอบ

    seed คงที่: การสุ่ม init หัวทำให้ผลเหวี่ยง ~0.06 acc ระหว่างรัน (วัดแล้ว 0.451/0.479/0.512)
    ถ้าไม่ตรึงไว้ ผลรันซ้ำจะเทียบกันไม่ได้ และเปิดช่องให้เลือกเมล็ดที่ดูดี
    """
    torch.manual_seed(seed)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    w = nn.Linear(Xt.shape[1], n_cls)
    opt = torch.optim.AdamW(w.parameters(), lr=lr, weight_decay=wd)
    cnt = np.bincount(y, minlength=n_cls).astype(np.float32)
    cw = torch.tensor(cnt.sum() / (n_cls * np.maximum(cnt, 1)), dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(w(Xt), yt, weight=cw)
        loss.backward()
        opt.step()
    return lambda Z: w(torch.tensor((Z - mu) / sd, dtype=torch.float32)).argmax(1).numpy()


def macro_f1(y, p, n_cls=3):
    fs = []
    for c in range(n_cls):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="เมล็ดของการแบ่ง fold — ตรึงไว้เพื่อให้รันซ้ำได้ผลเดิมเป๊ะ")
    ap.add_argument("--frac", choices=("raster", "coco"), default="raster",
                    help="ที่มาของสัดส่วนเป้าหมาย — raster คือค่าที่ถูก ส่วน coco เก็บไว้เทียบย้อนหลังเท่านั้น")
    ap.add_argument("--drop-swallowed", action="store_true",
                    help="ตัดภาพที่ polygon สะอาดถูกกลืนหมด (ค่า 1.000 ปลอมจากการวาดพลาด)")
    ap.add_argument("--inits", type=int, default=10,
                    help="จำนวน init ของหัว logistic ที่เฉลี่ยกัน — การสุ่ม init เดียวเหวี่ยงถึง 20 จุดต่อ fold")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    rows = load_rows(a.frac, a.drop_swallowed)
    tag = f"[frac={a.frac}{'+drop-swallowed' if a.drop_swallowed else ''} · inits={a.inits} · split-seed={a.seed}]"
    print(f"ภาพ {len(rows)} ใบ (ต้นฉบับ {len({r['stem'] for r in rows})} ใบ · กลุ่มรถ {len({r['group'] for r in rows})}) · {dev.type}")
    print("  ระดับ:", {LEVELS[k]: v for k, v in sorted(Counter(r["y"] for r in rows).items())})
    print("  กล้อง:", dict(Counter(r["cam"] for r in rows)))

    X = features(rows, dev, a.size)
    y = np.array([r["y"] for r in rows])
    cam = np.array([r["cam"] for r in rows])
    grp = np.array([r["group"] for r in rows])

    # MPK00 = ชุดทดสอบข้ามกล้อง แยกออกไปเลย ไม่เข้า fold
    hold = cam == "MPK00"
    Xm, ym, gm = X[~hold], y[~hold], grp[~hold]
    print(f"\nMPDC00 เข้า CV {len(ym)} ใบ · MPK00 กันไว้ข้ามกล้อง {int(hold.sum())} ใบ")

    groups = sorted(set(gm))
    rng = np.random.RandomState(a.seed)
    rng.shuffle(groups)
    folds = [set(groups[i::a.folds]) for i in range(a.folds)]

    accs, f1s, base_maj, base_cam = [], [], [], []
    for k, te_g in enumerate(folds, 1):
        te = np.array([g in te_g for g in gm])
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        # เฉลี่ยหลาย init: init เดียวให้ผลต่างกันได้ถึง 20 จุดบน fold เดียวกัน (วัดแล้ว fold4 0.576 vs 0.788)
        aa, ff = [], []
        for si in range(a.inits):
            pred = fit_logistic(Xm[~te], ym[~te], seed=si)(Xm[te])
            aa.append(float((pred == ym[te]).mean()))
            ff.append(macro_f1(ym[te], pred))
        acc, f1 = float(np.mean(aa)), float(np.mean(ff))
        maj = int(np.bincount(ym[~te], minlength=3).argmax())
        accs.append(acc); f1s.append(f1)
        base_maj.append(float((ym[te] == maj).mean()))
        print(f"  fold {k}: n_test {int(te.sum()):3d} · acc {acc:.3f} (init แกว่ง {min(aa):.3f}-{max(aa):.3f}) · macroF1 {f1:.3f} · baseline ทายคลาสหลัก {base_maj[-1]:.3f}")

    print(f"\n=== ผล 5-fold (MPDC00 เท่านั้น) ===")
    print(f"  โมเดล {tag} acc {np.mean(accs):.3f} ±{np.std(accs):.3f} · macroF1 {np.mean(f1s):.3f} ±{np.std(f1s):.3f}")
    print(f"  baseline ทายคลาสหลักเสมอ acc {np.mean(base_maj):.3f} ±{np.std(base_maj):.3f}")
    # baseline กล้อง: ทายระดับที่พบบ่อยที่สุดของกล้องนั้น
    cam_major = {c: int(np.bincount(y[cam == c], minlength=3).argmax()) for c in set(cam)}
    # เทียบให้ยุติธรรม: CV วัดบน MPDC00 เท่านั้น baseline กล้องจึงต้องวัดบน MPDC00 เท่านั้นด้วย
    bc = float(np.mean(ym == cam_major["MPDC00"]))
    bc_all = float(np.mean([cam_major[c] == yy for c, yy in zip(cam, y)]))
    print(f"  baseline ทายจากกล้องอย่างเดียว acc {bc:.3f} (บน MPDC00 · ทายว่า '{LEVELS[cam_major['MPDC00']]}' ทุกใบ) · ทั้งชุดรวม MPK00 {bc_all:.3f} เทียบ CV ตรง ๆ ไม่ได้")

    # ข้ามกล้อง: เทรนบน MPDC00 ทั้งหมด ทดสอบบน MPK00
    xres = None
    if hold.sum() >= 3:
        pred = fit_logistic(Xm, ym, seed=0)(X[hold])
        base_h = float(np.max(np.bincount(y[hold], minlength=3)) / hold.sum())
        xres = dict(n=int(hold.sum()), acc=float((pred == y[hold]).mean()), baseline_majority=base_h,
                    pred=dict(Counter(LEVELS[p] for p in pred)), true=dict(Counter(LEVELS[t] for t in y[hold])))
        print(f"\n=== ข้ามกล้อง MPDC00 -> MPK00 (n={xres['n']}) ===")
        print(f"  acc {xres['acc']:.3f} · baseline ทายคลาสเด่นของ MPK00 {base_h:.3f}"
              + (" — โมเดลไม่ชนะ baseline" if xres['acc'] <= base_h else " — โมเดลชนะ baseline"))
        print(f"  ทาย {xres['pred']} · จริง {xres['true']}")

    verdict = ("มีสัญญาณ" if np.mean(accs) > max(np.mean(base_maj), bc) + 0.10 else "ยังไม่เห็นสัญญาณ")
    print(f"\nสรุป pilot (64 ภาพต้นฉบับ ห้ามอ้างความแม่น · ข้ามกล้องยังไม่ผ่าน) {tag}: **{verdict}** — โมเดล {np.mean(accs):.3f} vs baseline สูงสุด {max(np.mean(base_maj), bc):.3f}")

    (OUT / "train_result.json").write_text(json.dumps({
        "n_images": len(rows), "n_source": len({r["stem"] for r in rows}), "n_groups": len(set(grp)),
        "excluded_stems": sorted(EXCLUDE_STEMS), "cuts": CUTS, "size": a.size, "seed": a.seed, "inits_averaged": a.inits, "frac_source": a.frac,
        "cv_acc_mean": float(np.mean(accs)), "cv_acc_sd": float(np.std(accs)),
        "cv_macro_f1_mean": float(np.mean(f1s)), "cv_macro_f1_sd": float(np.std(f1s)),
        "baseline_majority": float(np.mean(base_maj)), "baseline_camera": bc,
        "cross_camera": xres, "verdict": verdict,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"เขียน {OUT/'train_result.json'}")


if __name__ == "__main__":
    main()
