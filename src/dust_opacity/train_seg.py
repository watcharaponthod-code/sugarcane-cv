# -*- coding: utf-8 -*-
"""Train a clean-license 2-class (dust / not-dust) segmenter on sim_topdown_lite frames.
Models (torchvision, BSD-3): lraspp_mobilenet_v3_large (default) | deeplabv3_mobilenet_v3_large. Backbone = torchvision
ImageNet-1k MobileNetV3-Large weights (BSD-3 release; flagged in the report). Head trained from scratch.
Augmentation (train only, c6 spec (c)): gain U(0.7,1.3), gaussian blur p=.5 sigma U(0.3,1.5), jpeg p=.5 q U(40,95), hflip p=.5.
Usage: python train_seg.py [--model lraspp|deeplabv3] [--backbone mobilenet|resnet50] [--data v2|v3|v4] [--hardneg-weight W] [--epochs 12] [--seed 0] [--bs 8]
Writes: qa/own_model/<model>_seed<seed>.pt, <model>_seed<seed>_train.json"""
import sys, json, time, math, argparse
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import sim_data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="lraspp", choices=["lraspp", "deeplabv3"])
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--limit", type=int, default=0, help="debug: cap train frames")
ap.add_argument("--data", default="v2", choices=["v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"], help="v2 = c6 dataset.py (round 1); v3-v10 = qa dataset_<v>.py (rounds 2-9)")
ap.add_argument("--backbone", default="mobilenet", choices=["mobilenet", "resnet50"], help="resnet50 => torchvision deeplabv3_resnet50 (BSD-3); requires --model deeplabv3")
ap.add_argument("--hardneg-weight", type=float, default=1.0, help="loss multiplier for samples the dataset manifest marks as hard negatives (1.0 = off)")
ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
ap.add_argument("--sched", default="onecycle", choices=["onecycle", "cosine"], help="cosine = linear warmup (--warmup-epochs) then cosine to 0")
ap.add_argument("--warmup-epochs", type=float, default=1.0, help="cosine schedule only")
ap.add_argument("--backbone-lr-mult", type=float, default=1.0, help="lr multiplier for the pretrained backbone (0.1 = lr/10)")
ap.add_argument("--dropout", type=float, default=0.0, help="dropout inserted before the final 1x1 conv of the head")
ap.add_argument("--crop", type=int, default=0, help="random square crop at train time (0 = full frame); eval is always full frame")
ap.add_argument("--thin-weight", type=float, default=0.0, help="pixel weight w = 1 + T*[0.03 < alpha < 0.25] (density only; 0 = uniform)")
ap.add_argument("--best", default="last", choices=["last", "val_mae", "val_iou"], help="checkpoint selection metric (last = round-2 behaviour)")
ap.add_argument("--thr-sweep", action="store_true", help="after training sweep thr on val for max pixel-F1 and record it")
ap.add_argument("--freeze-backbone", action="store_true", help="train the head only; the pretrained backbone cannot adapt to the generator")
ap.add_argument("--grayscale", action="store_true", help="luma-only input, replicated to 3 channels, at train AND eval; colour cues in the generator become unusable")
ap.add_argument("--save-frac", type=float, nargs="*", default=[], metavar="F",
                help="also write <tag>_ep00pNN.pt part-way through epoch 1 (e.g. 0.25 0.5 0.75) - the peak may precede the first epoch boundary")
ap.add_argument("--save-every-epoch", action="store_true", help="also write <tag>_epNN.pt after every epoch, for probing how a metric moves as the model fits the generator")
ap.add_argument("--thr-grid", type=float, nargs=3, default=(0.10, 0.50, 0.02), metavar=("LO", "HI", "STEP"), help="threshold sweep grid")
ap.add_argument("--train-hw", type=int, nargs=2, default=None, metavar=("H", "W"), help="override sim_data.TRAIN_HW render size (default 384 704); eval always uses the full frame at this size")
ap.add_argument("--cache-dir", default=None, help="render the frame cache into disk-backed .npy here and train off memmaps (RAM holds only a batch); reused if shape matches")
ap.add_argument("--cache-y8", action="store_true", help="cache the density target as uint8 (alpha*255): ~4x less RAM, precision the train path already had")
ap.add_argument("--amp", action="store_true", help="mixed precision (Ampere/Turing only; pointless on Pascal)")
ap.add_argument("--tag", default=None, help="override output tag (default derived from model/density/data/seed)")
ap.add_argument("--density", action="store_true", help="continuous target alpha_screen in [0,1]: 1 logit, loss = soft BCE + L1 (e3 v3 direction)")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)
LUMA = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(DEV)
sim_data.use(args.data)
if args.train_hw: sim_data.TRAIN_HW = tuple(args.train_hw)
# density metrics binarise the continuous target; use the generator's own thresholds (v2/v3 = 0.29/0.05, v4 = 0.12/0.035)
GT_POS = float(getattr(sim_data.c6_module(), "GT_POS", 0.29)); GT_NEG = float(getattr(sim_data.c6_module(), "GT_NEG", 0.05))
TAG = args.tag or f"{args.model}_{'density_' if args.density else ''}{'' if args.data == 'v2' else args.data + '_'}{'' if args.backbone == 'mobilenet' else args.backbone + '_'}seed{args.seed}"
print("device", DEV, torch.__version__, torchvision.__version__, TAG, flush=True)


def build_model(name):
    nc = 1 if args.density else 2
    if args.backbone == "resnet50":
        if name != "deeplabv3": ap.error("--backbone resnet50 is only available with --model deeplabv3")
        return deeplabv3_resnet50(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1, num_classes=nc, aux_loss=False)
    kw = dict(weights=None, weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1, num_classes=nc)
    return (lraspp_mobilenet_v3_large if name == "lraspp" else deeplabv3_mobilenet_v3_large)(**kw)


def add_dropout(m, p_drop):
    """insert Dropout(p) immediately before the final 1x1 classifier conv of the head."""
    if p_drop <= 0: return m
    if hasattr(m.classifier, "high_classifier"):                     # LRASPP head: two 1x1 classifiers, summed
        m.classifier.high_classifier = nn.Sequential(nn.Dropout2d(p_drop), m.classifier.high_classifier)
        m.classifier.low_classifier = nn.Sequential(nn.Dropout2d(p_drop), m.classifier.low_classifier)
    else:                                                            # DeepLabHead = Sequential([..., final conv])
        head = list(m.classifier); m.classifier = nn.Sequential(*head[:-1], nn.Dropout2d(p_drop), head[-1])
    return m


def hardneg_flags(keys):
    """per-sample 0/1 hard-negative flags from the dataset manifest; None when weighting is off or unavailable."""
    if args.hardneg_weight == 1.0:
        return None
    m = sim_data.c6_module()
    fn = next((getattr(m, n) for n in ("sample_meta", "is_hardneg", "is_hard_negative", "hard_negative", "manifest", "sample_manifest") if hasattr(m, n)), None)
    if fn is None:
        print("WARN: --hardneg-weight given but the dataset exposes no manifest hook; weighting DISABLED", flush=True)
        return None
    def flag(k):
        r = fn(k)
        if isinstance(r, dict): return bool(r.get("is_hardneg", r.get("hard_negative", r.get("hardneg", r.get("hard_neg", False)))))
        return bool(r)
    f = np.array([flag(k) for k in keys], np.float32)
    print(f"hard negatives: {int(f.sum())}/{len(f)} samples x{args.hardneg_weight} (hook {fn.__name__})", flush=True)
    return f


def to_tensor(u8):                            # (B,H,W) gray -> replicate 3ch | (B,H,W,3) BGR -> RGB ; normalized (B,3,H,W)
    x = torch.from_numpy(np.ascontiguousarray(u8)).to(DEV).float().div_(255)
    x = x.unsqueeze(1).repeat(1, 3, 1, 1) if x.ndim == 3 else x[..., [2, 1, 0]].permute(0, 3, 1, 2).contiguous()   # NCHW: channels_last is ~10x slower on Pascal
    if args.grayscale:                        # BT.601 luma, then replicated - same tensor shape, no colour left
        x = (x * LUMA).sum(1, keepdim=True).repeat(1, 3, 1, 1)
    return (x - MEAN) / STD


def augment_local(img, lab, rng):              # fallback only; mirrors c6 spec (c)
    img = img.astype(np.float32) * rng.uniform(0.7, 1.3)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 96))]); img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    if rng.random() < 0.5:
        img, lab = img[:, ::-1].copy(), lab[:, ::-1].copy()
    return img, lab


try:                                          # same augmentation function as c6's CLIP-decoder (applied at TRAIN_HW, not full-res)
    augment = sim_data.c6_module().augment; AUG_SRC = getattr(sim_data.c6_module(), "AUG_DESC", "c6 own_model.dataset.augment (gain U(.7,1.3); blur p.5 k{3,5}; jpeg p.5 q U(40,89); hflip p.5)") + " applied at TRAIN_HW"
except Exception as _e:                       # noqa: BLE001
    augment, AUG_SRC = augment_local, f"local fallback ({type(_e).__name__})"


def render_cached(keys, hw, density, name):
    """render into /cache/<name>_{X,Y}.npy (X uint8, Y uint8 = alpha*255) and return read-only memmaps."""
    d = Path(args.cache_dir); d.mkdir(parents=True, exist_ok=True)
    ch = (3,) if sim_data.is_colour() else ()
    xs = (len(keys), hw[0], hw[1]) + ch; ys = (len(keys), hw[0], hw[1])
    px, py = d / f"{name}_X.npy", d / f"{name}_Y.npy"
    if px.exists() and py.exists():
        try:
            X, Y = np.load(px, mmap_mode="r"), np.load(py, mmap_mode="r")
            if X.shape == xs and Y.shape == ys:
                print(f"  cache hit {px.name} {X.shape}", flush=True); return X, Y
        except Exception as e:                                        # noqa: BLE001
            print(f"  cache unusable ({type(e).__name__}), re-rendering", flush=True)
    X = np.lib.format.open_memmap(px, mode="w+", dtype=np.uint8, shape=xs)
    Y = np.lib.format.open_memmap(py, mode="w+", dtype=np.uint8, shape=ys)
    for k, key in enumerate(keys):
        x_, y_ = sim_data.render(key, hw, density)
        X[k] = x_; Y[k] = np.rint(y_ * 255) if density else y_
        if k % 200 == 0: print(f"  rendered {k}/{len(keys)}", flush=True)
    X.flush(); Y.flush(); del X, Y
    return np.load(px, mmap_mode="r"), np.load(py, mmap_mode="r")


def yf(a):
    """cached target -> float alpha in [0,1] (identity unless --cache-y8)."""
    return a.astype(np.float32) / 255.0 if args.cache_y8 else a


def crop_pair(img, lab, rng):
    """random square crop at train time; eval always runs on the full frame."""
    c = args.crop
    if not c: return img, lab
    h, w_ = img.shape[:2]
    if h < c or w_ < c:
        if not getattr(crop_pair, "_warned", False):
            print(f"WARN: --crop {c} > frame {h}x{w_}; crop DISABLED", flush=True); crop_pair._warned = True
        return img, lab
    y0, x0 = int(rng.integers(0, h - c + 1)), int(rng.integers(0, w_ - c + 1))
    return img[y0:y0 + c, x0:x0 + c], lab[y0:y0 + c, x0:x0 + c]


@torch.no_grad()
def predict_prob(model, X, bs=8):
    model.eval()
    return np.concatenate([torch.sigmoid(model(to_tensor(X[i:i + bs]))["out"])[:, 0].float().cpu().numpy() for i in range(0, len(X), bs)])


def sweep_thr(prob, Y, lo=None, hi=None, step=None):
    """max pixel-F1 on VAL over the thr grid (never on test). positives = alpha >= GT_POS, ambiguous band excluded."""
    lo, hi, step = args.thr_grid if lo is None else (lo, hi, step)
    pos = GT_POS * 255 if args.cache_y8 else GT_POS      # compare in the cache's own units
    neg = GT_NEG * 255 if args.cache_y8 else GT_NEG
    grid = np.round(np.arange(lo, hi + 1e-9, step), 3)
    tp = np.zeros(len(grid)); fp = np.zeros(len(grid)); fn = np.zeros(len(grid))
    for i in range(0, len(Y), 32):                       # chunked: Y may be a disk memmap
        y = np.asarray(Y[i:i + 32]); g = y >= pos; valid = g | (y <= neg); gv = g & valid
        p = prob[i:i + 32]
        for c, t in enumerate(grid):
            pr = (p > t) & valid
            tp[c] += (pr & gv).sum(); fp[c] += (pr & ~gv).sum(); fn[c] += (~pr & gv).sum()
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1.0)
    prec = tp / np.maximum(tp + fp, 1.0); rec = tp / np.maximum(tp + fn, 1.0)
    iou = tp / np.maximum(tp + fp + fn, 1.0)
    c = int(f1.argmax())
    return dict(thr=float(grid[c]), val_f1=float(f1[c]), val_iou_at_thr=float(iou[c]),
                thr_curve=[dict(thr=float(t), f1=float(a), iou=float(b), precision=float(d),
                                recall=float(e), tp=int(u), fp=int(v), fn=int(w))
                           for t, a, b, d, e, u, v, w in zip(grid, f1, iou, prec, rec, tp, fp, fn)])


@torch.no_grad()
def evaluate(model, X, Y, bs=8):
    """Density: accumulate the confusion table over the whole threshold grid in one pass, then
    report quality AT THE EPOCH'S OWN BEST-F1 THRESHOLD and stamp that threshold into the row.
    A fixed cut is also kept, but under a name that says which cut it is - an IoU with no
    threshold beside it cannot be compared to anything."""
    model.eval(); mae = 0.0
    if not args.density:
        inter = union = tp = fp = fn = 0
        for i in range(0, len(X), bs):
            out = model(to_tensor(X[i:i + bs]))["out"]; y = Y[i:i + bs]
            pred = out.argmax(1).cpu().numpy(); valid = y != sim_data.IGNORE
            p, g = (pred == 1) & valid, (y == 1) & valid
            inter += (p & g).sum(); union += (p | g).sum()
            tp += (p & g).sum(); fp += (p & ~g).sum(); fn += (~p & g).sum()
        return dict(dust_iou=float(inter / max(union, 1)), dust_precision=float(tp / max(tp + fp, 1)),
                    dust_recall=float(tp / max(tp + fn, 1)))

    lo, hi, step = args.thr_grid
    grid = np.round(np.arange(lo, hi + 1e-9, step), 3)
    FIXED = 0.5                                        # the historical cut, kept for continuity only
    gtp = np.zeros(len(grid)); gfp = np.zeros(len(grid)); gfn = np.zeros(len(grid))
    ftp = ffp = ffn = 0
    for i in range(0, len(X), bs):
        out = model(to_tensor(X[i:i + bs]))["out"]
        y = yf(np.asarray(Y[i:i + bs]))
        prob = torch.sigmoid(out)[:, 0].float().cpu().numpy()
        mae += float(np.abs(prob - y).sum())
        valid = (y >= GT_POS) | (y <= GT_NEG); g = (y >= GT_POS) & valid
        for c, t in enumerate(grid):
            pr = (prob > t) & valid
            gtp[c] += (pr & g).sum(); gfp[c] += (pr & ~g).sum(); gfn[c] += (~pr & g).sum()
        pf = (prob > FIXED) & valid
        ftp += (pf & g).sum(); ffp += (pf & ~g).sum(); ffn += (~pf & g).sum()

    f1 = 2 * gtp / np.maximum(2 * gtp + gfp + gfn, 1.0)
    c = int(f1.argmax())
    den = max(gtp[c] + gfp[c] + gfn[c], 1.0)
    return dict(thr=float(grid[c]),                    # the threshold every figure below is measured at
                dust_iou=float(gtp[c] / den),
                dust_precision=float(gtp[c] / max(gtp[c] + gfp[c], 1.0)),
                dust_recall=float(gtp[c] / max(gtp[c] + gfn[c], 1.0)),
                dust_f1=float(f1[c]),
                dust_iou_at_0p50=float(ftp / max(ftp + ffp + ffn, 1)),
                dust_precision_at_0p50=float(ftp / max(ftp + ffp, 1)),
                dust_recall_at_0p50=float(ftp / max(ftp + ffn, 1)),
                mae_alpha=mae / (len(X) * X.shape[1] * X.shape[2]))


tr_keys, va_keys, which = sim_data.get_split(args.seed)
if args.limit: tr_keys, va_keys = tr_keys[:args.limit], va_keys[:max(8, args.limit // 5)]
HN = hardneg_flags(tr_keys)
Wtr = None if HN is None else 1.0 + (args.hardneg_weight - 1.0) * HN
print("split:", which, len(tr_keys), len(va_keys), "| augment:", AUG_SRC, flush=True)
t0 = time.time()
HW = sim_data.TRAIN_HW   # render_all binds its default at import, so pass the (possibly overridden) size explicitly
if args.cache_dir:
    args.cache_y8 = args.cache_y8 or args.density          # the disk cache always stores alpha as uint8
    base = f"{args.data}_seed{args.seed}_{HW[0]}x{HW[1]}{'_d' if args.density else ''}"
    Xtr, Ytr = render_cached(tr_keys, HW, args.density, f"{base}_train{len(tr_keys)}")
    Xva, Yva = render_cached(va_keys, HW, args.density, f"{base}_val{len(va_keys)}")
else:
    Xtr, Ytr = sim_data.render_all(tr_keys, HW, density=args.density, y8=args.cache_y8)
    Xva, Yva = sim_data.render_all(va_keys, HW, density=args.density, y8=args.cache_y8)
print(f"cache: X {Xtr.nbytes / 1e9:.2f}+{Xva.nbytes / 1e9:.2f} GB, Y {Ytr.nbytes / 1e9:.2f}+{Yva.nbytes / 1e9:.2f} GB ({Ytr.dtype}) "
      f"{'on disk (memmap)' if args.cache_dir else 'in RAM'}", flush=True)
if args.density:
    _pos = GT_POS * 255 if args.cache_y8 else GT_POS      # compare in the cache's own units, no full-size float copy
    print(f"rendered in {time.time() - t0:.0f}s; density target: mean alpha {Ytr.mean() / (255 if args.cache_y8 else 1):.4f}, "
          f"frac>={GT_POS} {(Ytr >= _pos).mean():.4f}, frames with dust {(Ytr >= _pos).any(axis=(1, 2)).mean():.2f}", flush=True)
else: print(f"rendered in {time.time() - t0:.0f}s; train dust-pixel frac {(Ytr == 1).mean():.4f}, ignore {(Ytr == 255).mean():.4f}, frames with dust {(Ytr == 1).any(axis=(1, 2)).mean():.2f}", flush=True)

model = add_dropout(build_model(args.model), args.dropout).to(DEV)
if args.freeze_backbone:
    for q in model.backbone.parameters(): q.requires_grad_(False)
    model.backbone.eval()                                            # keep BN running stats from ImageNet
n_train_p = sum(q.numel() for q in model.parameters() if q.requires_grad)
print(f"trainable params {n_train_p:,} / {sum(q.numel() for q in model.parameters()):,}"
      f"{' (backbone frozen)' if args.freeze_backbone else ''}", flush=True)
if args.backbone_lr_mult == 1.0 or args.freeze_backbone:
    groups = [dict(params=[q for q in model.parameters() if q.requires_grad], lr=args.lr)]
else:                                                                # pretrained backbone gets lr * mult, head keeps lr
    bb = [q for q in model.backbone.parameters() if q.requires_grad]; bb_ids = {id(q) for q in bb}
    groups = [dict(params=bb, lr=args.lr * args.backbone_lr_mult),
              dict(params=[q for q in model.parameters() if q.requires_grad and id(q) not in bb_ids], lr=args.lr)]
opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
spe = (len(Xtr) + args.bs - 1) // args.bs; steps = args.epochs * spe
if args.sched == "cosine":
    wu = max(1, int(round(args.warmup_epochs * spe)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / wu if t < wu else 0.5 * (1 + math.cos(math.pi * (t - wu) / max(1, steps - wu))))
else:
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
FRAC_AT = {max(1, int(round(f * spe))): f for f in args.save_frac if 0 < f < 1}
if FRAC_AT: print("intra-epoch checkpoints at batches", sorted(FRAC_AT), "of", spe, flush=True)
scaler = torch.amp.GradScaler("cuda", enabled=args.amp and DEV.type == "cuda")
crit = nn.CrossEntropyLoss(ignore_index=sim_data.IGNORE)
rng = np.random.default_rng(args.seed)
log = []; best_score, best_epoch, best_state = float("inf"), None, None
for ep in range(args.epochs):
    model.train()
    if args.freeze_backbone: model.backbone.eval()               # frozen BN stays in eval mode
    perm = rng.permutation(len(Xtr)); tl = 0; n = 0; te = time.time()
    for i in range(0, len(perm), args.bs):
        idx = perm[i:i + args.bs]
        w = None if Wtr is None else torch.from_numpy(Wtr[idx]).to(DEV)
        with torch.autocast("cuda", enabled=args.amp and DEV.type == "cuda"):
            if args.density:                          # c6's augment flips label too: pass alpha as uint8 (x255), then back to [0,1]
                pairs = [crop_pair(*augment(np.array(Xtr[j]), np.array(Ytr[j]) if args.cache_y8 else np.rint(Ytr[j] * 255).astype(np.uint8), rng), rng) for j in idx]
                x = to_tensor(np.stack([p_[0] for p_ in pairs])); y = torch.from_numpy(np.stack([p_[1] for p_ in pairs]).astype(np.float32) / 255).to(DEV)[:, None]
                out = model(x)["out"]
                if w is None and args.thin_weight == 0:
                    loss = F.binary_cross_entropy_with_logits(out, y) + F.l1_loss(torch.sigmoid(out), y)
                else:                                 # same value at w==1 / T==0, only reduced per pixel then per sample
                    pix = F.binary_cross_entropy_with_logits(out, y, reduction="none") + (torch.sigmoid(out) - y).abs()
                    if args.thin_weight:              # push thin haze: w_pix = 1 + T*[0.03 < alpha < 0.25]
                        wp = 1.0 + args.thin_weight * ((y > 0.03) & (y < 0.25)).to(pix.dtype)
                        per = (pix * wp).sum((1, 2, 3)) / wp.sum((1, 2, 3))
                    else:
                        per = pix.mean((1, 2, 3))
                    loss = per.mean() if w is None else (per * w).sum() / w.sum()
            else:
                ims, labs = zip(*[crop_pair(*augment(np.array(Xtr[j]), np.array(Ytr[j]), rng), rng) for j in idx])
                x = to_tensor(np.stack(ims)); y = torch.from_numpy(np.stack(labs)).long().to(DEV)
                out = model(x)["out"]
                if w is None:
                    loss = crit(out, y)
                else:
                    pix = F.cross_entropy(out, y, ignore_index=sim_data.IGNORE, reduction="none")
                    cnt = (y != sim_data.IGNORE).sum((1, 2)).clamp(min=1)
                    loss = (pix.sum((1, 2)) / cnt * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item() * len(idx); n += len(idx)
        if ep == 0 and FRAC_AT:                        # first epoch only: that is where the peak is suspected
            f = FRAC_AT.get(i // args.bs + 1)
            if f is not None:
                torch.save(model.state_dict(), HERE / f"{TAG}_ep00p{int(round(f * 100)):02d}.pt")
                print(f"saved {TAG}_ep00p{int(round(f * 100)):02d}.pt", flush=True)
    ev = evaluate(model, Xva, Yva)
    log.append(dict(epoch=ep + 1, train_loss=tl / n, sec=round(time.time() - te, 1), **ev))
    print(json.dumps(log[-1]), flush=True)
    if args.save_every_epoch:                          # a per-epoch trail, so an external metric can be walked back over training
        torch.save(model.state_dict(), HERE / f"{TAG}_ep{ep + 1:02d}.pt")
    if args.best != "last":                            # val MAE alpha is the round-3 selector (high IoU can just be memorisation)
        score = ev.get("mae_alpha", 1e9) if args.best == "val_mae" else -ev["dust_iou"]
        if score < best_score:
            best_score, best_epoch, best_state = score, ep + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    print(f"restoring best epoch {best_epoch} by {args.best} ({best_score:.5f})", flush=True)
    model.load_state_dict(best_state)
THR = sweep_thr(predict_prob(model, Xva), Yva) if (args.thr_sweep and args.density) else {}
if THR: print("thr sweep:", json.dumps(THR), flush=True)
torch.save(model.state_dict(), HERE / f"{TAG}.pt")     # state_dict only: loadable by any torch version
json.dump(dict(tag=TAG, model=args.model, backbone=args.backbone, hardneg_weight=args.hardneg_weight, **THR,
               best_select=args.best, best_epoch=best_epoch, val_mae_alpha=log[-1].get("mae_alpha") if args.best == "last" else (best_score if args.best == "val_mae" else None),
               wd=args.wd, sched=args.sched, warmup_epochs=args.warmup_epochs, backbone_lr_mult=args.backbone_lr_mult,
               dropout=args.dropout, crop=args.crop, thin_weight=args.thin_weight,
               freeze_backbone=bool(args.freeze_backbone), grayscale=bool(args.grayscale),
               save_frac=list(args.save_frac),
               n_trainable_params=n_train_p, amp=bool(args.amp), cache_y8=bool(args.cache_y8), cache_dir=args.cache_dir, thr_grid=list(args.thr_grid), dataset_stamp=which, gt_pos=GT_POS, gt_neg=GT_NEG,
               n_hardneg=None if HN is None else int(HN.sum()), density=bool(args.density), color=bool(Xtr.ndim == 4), temporal=bool(getattr(sim_data.c6_module(), "TEMPORAL", False)), data=args.data, data_version=getattr(sim_data.c6_module(), "VERSION", "v2"), seed=args.seed, epochs=args.epochs, bs=args.bs, lr=args.lr, split=which,
               n_train=len(Xtr), n_val=len(Xva), train_hw=list(HW),
               render_hw=list(HW), train_input_hw=[args.crop, args.crop] if args.crop else list(HW),
               eval_hw=list(HW), eval_full_frame=True, device=str(DEV), torch=torch.__version__,
               torchvision=torchvision.__version__, cv2=cv2.__version__, python=sys.version.split()[0],
               backbone_weights=("torchvision ResNet50_Weights.IMAGENET1K_V1 (BSD-3 release)" if args.backbone == "resnet50"
                                 else "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V1 (BSD-3 release)"),
               augment=AUG_SRC,
               n_params=sum(p.numel() for p in model.parameters()), log=log, total_sec=round(time.time() - t0)),
          open(HERE / f"{TAG}_train.json", "w"), indent=1)
print("saved", TAG, f"{time.time() - t0:.0f}s")
