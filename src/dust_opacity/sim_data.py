# -*- coding: utf-8 -*-
"""Training data for the clean-license baseline. Primary source = c6's gen_video/own_model/dataset.py (team-shared, so the
baseline and c6's CLIP-decoder train on IDENTICAL samples): build_split(seed) -> (train_keys, val_keys); frame_and_mask(key)
-> (gray uint8 1629x893, label int8 {1 dust, 0 bg, -1 ignore}). Fallback (only if that import fails) = plain sim_topdown_lite
scenario A/B frames with a local random split; `which` string + sha256 of dataset.py are recorded so nobody can confuse the two.
Frames are downscaled to TRAIN_HW and cached in RAM (uint8). Nothing here reads clips 1-3 or the 4 stills."""
import sys, hashlib
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gen_video" / "topdown_sim")); sys.path.insert(0, str(ROOT / "gen_video"))
import sim_topdown_lite as S  # noqa: E402

TRAIN_HW = (384, 704)          # 1629x893 -> 704x384 keeps aspect (1.824 vs 1.833)
IGNORE = 255
C6_PATH = ROOT / "gen_video" / "own_model" / "dataset.py"
V3_PATH = Path(__file__).resolve().parent / "dataset_v3.py"
_c6 = None; SOURCE = "v2"                     # "v2" = c6's gen_video/own_model/dataset.py ; "v3"/"v4" = qa/own_model/dataset_<src>.py


def use(source):
    """select the generator BEFORE the first render: sim_data.use("v3")"""
    global SOURCE, _c6
    SOURCE = source; _c6 = None


def c6_module():
    global _c6
    if _c6 is None:
        import importlib
        _c6 = importlib.import_module("own_model.dataset") if SOURCE == "v2" else importlib.import_module(f"dataset_{SOURCE}")
    return _c6


def c6_stamp():
    m = c6_module(); path = C6_PATH if SOURCE == "v2" else V3_PATH.with_name(f"dataset_{SOURCE}.py")
    return f"{'c6:own_model/dataset.py' if SOURCE == 'v2' else f'qa:own_model/dataset_{SOURCE}.py'} VERSION={getattr(m, 'VERSION', '?')} sha256={hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"


def _setup_local():
    if not hasattr(S, "H"):
        S.set_scale(1.0); S.setup_scene()


def render_local(idx, scen):
    _setup_local()
    img, events, _, _ = S.build_frame(idx, S.DUST3_FN[scen])
    alpha = np.zeros(img.shape, np.float32)
    for _bay, a in events:
        alpha = 1.0 - (1.0 - alpha) * (1.0 - a)
    lab = np.full(alpha.shape, IGNORE, np.uint8)
    lab[alpha >= S.GT_POS_OPACITY] = 1
    lab[alpha <= S.GT_NEG_OPACITY] = 0
    return img, lab


_rec = []
_orig_dust_layer = S.dust_layer


def _dust_layer_rec(*a, **k):                      # records every blob alpha so we can build the screen-composed density target
    t_local, alpha = _orig_dust_layer(*a, **k); _rec.append(alpha); return t_local, alpha


S.dust_layer = _dust_layer_rec


def render(key, hw=TRAIN_HW, density=False):
    """key = c6 sample key (int) or local (scenario, idx). -> (gray uint8, target) at hw.
    density=False: target = label uint8 {0,1,255}; density=True: target = float32 alpha_screen = 1-prod(1-alpha_i) in [0,1]."""
    _rec.clear(); m = None if isinstance(key, tuple) else c6_module()
    if isinstance(key, tuple):
        img, lab = render_local(key[1], key[0])
    elif hasattr(m, "frame_alpha"):                                       # v3.x: alpha comes straight from the generator (colour or gray)
        img, a = m.frame_alpha(key)
        lab = a if density else np.where(a <= m.GT_NEG, 0, np.where(a >= m.GT_POS, 1, IGNORE)).astype(np.uint8)
    else:
        img, lab = m.frame_and_mask(key)
        lab = np.where(lab < 0, IGNORE, lab.astype(np.int16)).astype(np.uint8)
        if density:
            lab = np.zeros(img.shape[:2], np.float32)
            for a in _rec: lab = 1.0 - (1.0 - lab) * (1.0 - a)
    if hw is not None:
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
        lab = cv2.resize(lab, (hw[1], hw[0]), interpolation=cv2.INTER_AREA if density else cv2.INTER_NEAREST)
    return img, lab


def get_split(seed=0):
    """-> (train_keys, val_keys, which)."""
    try:
        tr, va = c6_module().build_split(seed)
        return list(tr), list(va), c6_stamp()
    except Exception as e:                                             # noqa: BLE001
        keys = [(s, i) for s in ("A", "B") for i in range(S.N)]
        rng = np.random.default_rng(seed); rng.shuffle(keys); n_val = len(keys) // 5
        return keys[n_val:], keys[:n_val], f"FALLBACK local sim A/B random 80/20 seed={seed} (c6 import failed: {type(e).__name__}: {str(e)[:80]})"


def is_colour():
    return bool(getattr(c6_module(), "COLOR", False)) if SOURCE != "v2" else False


def render_all(keys, hw=TRAIN_HW, log_every=200, density=False, y8=False):
    """y8=True stores a density target as uint8 (alpha*255) -- 4x less RAM for the cache, same precision the
    train path already used when it passed alpha through augment() as uint8."""
    ch = (3,) if is_colour() else ()
    ydt = np.uint8 if (y8 or not density) else np.float32
    X = np.zeros((len(keys), hw[0], hw[1]) + ch, np.uint8); Y = np.zeros((len(keys), hw[0], hw[1]), ydt)
    for k, key in enumerate(keys):
        x_, y_ = render(key, hw, density)
        X[k] = x_
        Y[k] = np.rint(y_ * 255) if (density and y8) else y_
        if log_every and k % log_every == 0:
            print(f"  rendered {k}/{len(keys)}", flush=True)
    return X, Y


if __name__ == "__main__":                                             # self-check
    tr, va, which = get_split(0)
    print("split:", which, len(tr), len(va))
    assert not set(tr) & set(va)
    img, lab = render(tr[0], None)
    assert img.shape == (893, 1629) and lab.shape == img.shape and set(np.unique(lab)) <= {0, 1, 255}
    X, Y = render_all(tr[:4], log_every=0)
    assert X.shape == (4,) + TRAIN_HW and Y.dtype == np.uint8
    print("dust frac in 4 samples", [round(float((y == 1).mean()), 3) for y in Y], "OK")
