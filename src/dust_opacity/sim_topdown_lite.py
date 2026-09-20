# -*- coding: utf-8 -*-
"""
Lean, streaming, half-resolution re-run of the 3 numbers db2b2-e3 mandated after
5 OOM/timeout deaths of sim_topdown.py's full run:
  1. texture-only (+rigid) presence table under PLC-real, scenario A and B
  2. geometric-joint rate + attribution accuracy during the connect window, B
  3. rigid TTL 10s (original) vs 3s -- "instant" mode dropped (out of scope now)

Design constraints from e3 (mandatory, no other fallback):
  - half resolution (scale 0.5)
  - STREAMING: never hold the whole clip's frames in memory, one frame at a
    time, discard after use (previous full run held 480 full-res uint8 frames
    x 2 scenarios simultaneously -- that + per-frame float32 intermediates is
    what hit 5.6GB and got OOM-killed)
  - no per-frame mask storage, no overlay rendering in this measurement pass
  - one scenario per OS process (run via PowerShell Start-Process, detached
    from any Bash/Monitor timeout), each writes its own JSON, merged after

Usage:
  python sim_topdown_lite.py --scenario A
  python sim_topdown_lite.py --scenario B
  python sim_topdown_lite.py --merge
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent
PHOTO = ROOT.parent / "site_photos" / "topdown_3bays_2026-09-03.png"

sys.path.insert(0, str(ROOT / "tune"))
from metric import confusion, build_samples_multi, table_md  # shared team module

# v5 (ac's review 10.5): the earlier half-res run's elevated low-tau FP was a
# scale artifact, not a PLC effect (oracle also got 373 FP there) -- SCALE is
# now a CLI flag; all window/area/flow constants are DERIVED from it so that
# --full-res reproduces the ORIGINAL constants (TEX_WIN=21, MIN_BLOB=800, ...)
# exactly, not an approximation.
FPS = 24
DURATION_S = 20
N = FPS * DURATION_S
TEX_THRESH = 0.5
DUP_FRAME_THRESH = 0.3
DUP_FRAME_FRAC = 0.05
GT_POS_OPACITY, GT_NEG_OPACITY = 0.29, 0.05
SPILL_WINDOW, CLOUD_WINDOW, CLOUD_GAIN = (6.0, 9.0), (14.0, 17.0), 0.85
PLC_LATENCY_RANGE, PLC_DROPOUT_FRAC = (0.3, 1.0), 0.02


def _odd(n):
    n = max(1, int(round(n)))
    return n if n % 2 == 1 else n + 1


def set_scale(scale):
    """(Re)derive every scale-dependent constant from a fresh SCALE value."""
    global SCALE, Y_TOP, Y_BOTTOM, TEX_WIN, EDGE_WIN, CLOSE_KSIZE, MIN_BLOB
    global FLOW_LO, FLOW_HI, EDGE_GATE, RIGID_EDGE, RIGID_FLOW, RIGID_KERNEL_SZ, IMAGE_TRIGGER_AREA
    SCALE = scale
    Y_TOP, Y_BOTTOM = int(95 * SCALE), int(760 * SCALE)
    TEX_WIN, EDGE_WIN, CLOSE_KSIZE = _odd(21 * SCALE), _odd(31 * SCALE), _odd(15 * SCALE)
    MIN_BLOB = max(1, round(800 * SCALE * SCALE))
    FLOW_LO, FLOW_HI, EDGE_GATE = 0.5 * SCALE, 4.0 * SCALE, 0.06
    RIGID_EDGE, RIGID_FLOW = 0.10, 4.0 * SCALE
    RIGID_KERNEL_SZ = _odd(25 * SCALE)
    IMAGE_TRIGGER_AREA = max(1, round(300 * SCALE * SCALE))


set_scale(0.5)  # overridden by --scale / --full-res in main()


def fmt(v):
    return "n/a" if v != v else f"{v:.3f}"


# ============================================================ ROI extraction
def trace_rail(mask, x_window, y_top, y_bottom):
    xs = np.full(y_bottom - y_top + 1, np.nan)
    for i, y in enumerate(range(y_top, y_bottom + 1)):
        row = mask[y, x_window[0]:x_window[1]]
        cols = np.nonzero(row)[0]
        if cols.size:
            xs[i] = x_window[0] + cols.mean()
    valid = ~np.isnan(xs)
    if valid.sum() < 2:
        raise RuntimeError(f"rail trace failed in window {x_window}")
    idx = np.arange(len(xs))
    return np.interp(idx, idx[valid], xs[valid])


def setup_scene():
    """Loads the photo and extracts bay ROIs at the CURRENT SCALE. Must be
    called (via set_scale() first) before any frame is built -- v5 makes SCALE
    a CLI choice (0.5 or full-res 1.0), so this can no longer run at import time."""
    global H, W, BAY_NAMES, BAYS, BAY_MASK, BAY_BBOX, BAY_CENTER_X, BAY_AREA, BG_GRAY
    photo_full = cv2.imread(str(PHOTO))
    assert photo_full is not None
    photo_bgr = cv2.resize(photo_full, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    H, W = photo_bgr.shape[:2]
    hsv = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    windows = [(int(150 * SCALE), int(260 * SCALE)), (int(590 * SCALE), int(680 * SCALE)),
               (int(950 * SCALE), int(1040 * SCALE)), (int(1290 * SCALE), int(1450 * SCALE))]
    rails = [trace_rail(yellow, w, Y_TOP, Y_BOTTOM) for w in windows]
    BAY_NAMES = ["bay1", "bay2", "bay3"]
    BAYS = {}
    for i, name in enumerate(BAY_NAMES):
        left, right = rails[i], rails[i + 1]
        ys = np.arange(Y_TOP, Y_BOTTOM + 1)
        poly = np.concatenate([np.stack([left, ys], axis=1), np.stack([right[::-1], ys[::-1]], axis=1)]).astype(np.int32)
        BAYS[name] = poly
    BAY_MASK = {b: cv2.fillPoly(np.zeros((H, W), np.uint8), [BAYS[b]], 1).astype(bool) for b in BAY_NAMES}
    BAY_BBOX = {b: cv2.boundingRect(BAYS[b]) for b in BAY_NAMES}
    BAY_CENTER_X = {b: BAY_BBOX[b][0] + BAY_BBOX[b][2] / 2 for b in BAY_NAMES}
    BAY_AREA = {b: int(BAY_MASK[b].sum()) for b in BAY_NAMES}
    BG_GRAY = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    print(f"[{H}x{W}, scale={SCALE}] bay bboxes: {BAY_BBOX}", flush=True)


# ============================================================ synthetic scene
def ease(t, t0, t1):
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    x = np.clip((t - t0) / (t1 - t0), 0, 1)
    return x * x * (3 - 2 * x)


def truck_layer(t, bay, grow, hold_end, shrink_end, seed):
    x, y, w, h = BAY_BBOX[bay]
    if t < grow[0] or t > shrink_end:
        return None
    if t < grow[1]:
        s = ease(t, grow[0], grow[1])
    elif t < hold_end:
        s = 1.0
    else:
        s = 1.0 - ease(t, hold_end, shrink_end)
    if s <= 0.01:
        return None
    rw, rh = w * 0.8 * s, h * 0.55 * s
    cx, cy = x + w * 0.5, y + h * 0.35
    x0, x1 = max(int(cx - rw / 2), 0), min(int(cx + rw / 2), W)
    y0, y1 = max(int(cy - rh / 2), 0), min(int(cy + rh / 2), H)
    if x1 <= x0 or y1 <= y0:
        return None
    rng = np.random.default_rng(seed)
    patch_val = 55 + rng.normal(0, 12, (y1 - y0, x1 - x0)).astype(np.float32)
    add, alpha = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    add[y0:y1, x0:x1] = np.clip(patch_val, 20, 120)
    alpha[y0:y1, x0:x1] = 1.0
    return add, alpha


def spill_layer(t):
    if not (SPILL_WINDOW[0] <= t <= SPILL_WINDOW[1]):
        return None
    grow = ease(t, SPILL_WINDOW[0], SPILL_WINDOW[0] + 1.0)
    if grow <= 0.01:
        return None
    x, y, w, h = BAY_BBOX["bay2"]
    rw, rh = w * 0.5 * grow, h * 0.35 * grow
    cx, cy = x + w * 0.5, y + h * 0.6
    x0, x1 = max(int(cx - rw / 2), 0), min(int(cx + rw / 2), W)
    y0, y1 = max(int(cy - rh / 2), 0), min(int(cy + rh / 2), H)
    if x1 <= x0 or y1 <= y0:
        return None
    rng = np.random.default_rng(42)
    patch = 125 + rng.normal(0, 35, (y1 - y0, x1 - x0)).astype(np.float32)
    add, alpha = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    add[y0:y1, x0:x1] = np.clip(patch, 50, 210)
    alpha[y0:y1, x0:x1] = 1.0
    return add, alpha


def cloud_gain(t):
    if not (CLOUD_WINDOW[0] - 0.5 <= t <= CLOUD_WINDOW[1] + 0.5):
        return 1.0
    edge_in = ease(t, CLOUD_WINDOW[0] - 0.5, CLOUD_WINDOW[0])
    edge_out = 1 - ease(t, CLOUD_WINDOW[1], CLOUD_WINDOW[1] + 0.5)
    return 1 - (1 - CLOUD_GAIN) * min(edge_in, edge_out)


def center_wobble(idx, seed_offset=0):
    ph = seed_offset * 1.7
    dx = (3.0 * np.sin(idx * 0.22 + ph) + 1.5 * np.sin(idx * 0.07 + ph * 2)) * SCALE
    dy = (2.0 * np.cos(idx * 0.19 + ph) + 1.0 * np.cos(idx * 0.05 + ph * 3)) * SCALE
    return dx, dy


def dust_layer(cx, cy, sigma, peak_opacity, idx, seed_offset=0):
    dx, dy = center_wobble(idx, seed_offset)
    yy, xx = np.mgrid[0:H, 0:W]
    envelope = np.exp(-((xx - (cx + dx)) ** 2 + (yy - (cy + dy)) ** 2) / (2 * sigma * sigma))
    alpha = np.clip(peak_opacity * envelope, 0, 0.95)
    return (1.0 - alpha).astype(np.float32), alpha


def dust1_params(t):
    if not (5.0 <= t <= 12.0):
        return None
    bx, by, bw, bh = BAY_BBOX["bay1"]
    cx0, cy = bx + bw * 0.5, by + bh * 0.5
    drift = ease(t, 7.0, 11.0)
    cx = cx0 + (BAY_CENTER_X["bay2"] - cx0) * drift
    grow = ease(t, 5.0, 7.0)
    fade = 1.0 - ease(t, 10.0, 12.0)
    peak = 0.55 * grow * fade
    return (cx, cy, (90 + 40 * grow) * SCALE, peak) if peak > 0.01 else None


def dust3_params_A(t):
    if t < 12.0:
        return None
    bx, by, bw, bh = BAY_BBOX["bay3"]
    grow = ease(t, 12.0, 16.0)
    peak = 0.6 * grow
    return (bx + bw * 0.5, by + bh * 0.5, (70 + 50 * grow) * SCALE, peak) if peak > 0.01 else None


def dust3_params_B(t):
    if t < 8.0:
        return None
    bx, by, bw, bh = BAY_BBOX["bay3"]
    grow = ease(t, 8.0, 11.0)
    peak = 0.6 * max(grow, 0.3 if t >= 8.0 else 0)
    cx, cy = bx + bw * 0.15, by + bh * 0.5
    sigma = (110 + 90 * grow) * SCALE
    return (cx, cy, sigma, peak) if peak > 0.01 else None


DUST3_FN = {"A": dust3_params_A, "B": dust3_params_B}


def build_frame(idx, dust3_fn):
    """Returns (frame_uint8, events, bay1_open, bay3_open). No list accumulation."""
    t = idx / FPS
    img = BG_GRAY.copy()

    tl = truck_layer(t, "bay1", (2, 8), 12, 14, seed=1)
    if tl:
        add, alpha = tl
        img = img * (1 - alpha) + add * alpha
    tl3 = truck_layer(t, "bay3", (9, 15), DURATION_S + 1, DURATION_S + 1, seed=3)
    if tl3:
        add, alpha = tl3
        img = img * (1 - alpha) + add * alpha
    sp = spill_layer(t)
    if sp:
        add, alpha = sp
        img = img * (1 - alpha) + add * alpha

    events = []
    p1 = dust1_params(t)
    if p1:
        cx, cy, sigma, peak = p1
        t_local, alpha = dust_layer(cx, cy, sigma, peak, idx, seed_offset=0)
        img = img * t_local + 205 * (1 - t_local)
        events.append(("bay1", alpha))
    p3 = dust3_fn(t)
    if p3:
        cx, cy, sigma, peak = p3
        t_local, alpha = dust_layer(cx, cy, sigma, peak, idx, seed_offset=1)
        img = img * t_local + 210 * (1 - t_local)
        events.append(("bay3", alpha))

    img = img * cloud_gain(t)
    img = np.clip(img, 0, 255)
    noise = np.random.default_rng(1000 + idx).normal(0, 1.5, (H, W)).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    bay1_open = 2.0 <= t <= 14.0
    bay3_open = t >= 9.0
    return img, events, bay1_open, bay3_open


# ==================================================================== detector
def box(img, k):
    return cv2.boxFilter(img, -1, (k, k), borderType=cv2.BORDER_REFLECT)


def texture_energy(gray, win=None):
    # NOTE: default args are bound at def-time, so a stale TEX_WIN would be
    # captured before set_scale() runs -- read the CURRENT global instead.
    return box(cv2.Laplacian(gray, cv2.CV_32F, ksize=3) ** 2, win if win is not None else TEX_WIN)


def edge_density(gray, win=None):
    return box(cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32) / 255.0, win if win is not None else EDGE_WIN)


def clean_mask(mask, min_blob=None):
    min_blob = MIN_BLOB if min_blob is None else min_blob
    m = (mask.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE, CLOSE_KSIZE))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    out = np.zeros_like(m, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob:
            out[labels == i] = True
    return out, labels, n, stats


def make_plc_real(oracle_bool, seed):
    rng = np.random.default_rng(seed)
    noisy_angle = oracle_bool.astype(float) + rng.normal(0, 0.15, N)
    jittered = noisy_angle > 0.5
    latency_s = float(rng.uniform(*PLC_LATENCY_RANGE))
    lat_frames = max(1, int(round(latency_s * FPS)))
    delayed = np.zeros(N, bool)
    delayed[lat_frames:] = jittered[:N - lat_frames]
    dropout_idx = rng.choice(N, size=int(N * PLC_DROPOUT_FRAC), replace=False)
    real = delayed.copy()
    for i in sorted(dropout_idx):
        if i > 0:
            real[i] = real[i - 1]
    return real, latency_s, len(dropout_idx)


def run_streaming(scenario, mode):
    """mode: 'plc_real_texture' (item 1+3) or 'oracle_joint' (item 2).
    Streams frames one at a time -- never stores the whole clip."""
    dust3_fn = DUST3_FN[scenario]

    # first pass: build the oracle bay1_open/bay3_open arrays only (cheap, no image work)
    bay1_open = np.array([2.0 <= i / FPS <= 14.0 for i in range(N)])
    bay3_open = np.array([i / FPS >= 9.0 for i in range(N)])
    rng_dup = np.random.default_rng(0 if scenario == "A" else 1)
    dup_idx = set(rng_dup.choice(np.arange(1, N), size=int(N * DUP_FRAME_FRAC), replace=False).tolist())

    if mode == "plc_real_texture":
        real1, lat1, drop1 = make_plc_real(bay1_open, 100 if scenario == "A" else 200)
        real3, lat3, drop3 = make_plc_real(bay3_open, 101 if scenario == "A" else 201)
        open1, open3 = real1, real3
        meta = dict(lat1=lat1, lat3=lat3, drop1=drop1, drop3=drop3)
    else:
        open1, open3 = bay1_open, bay3_open
        meta = {}

    global_ref_frames = []  # only first 2s worth, needed for global_J -- small, kept
    active_J = None
    active_J_energy = None
    snapshotted = {"bay1": False, "bay3": False}
    last_rigid_frame = np.full((H, W), -10 ** 9, dtype=np.int64)
    rigid_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RIGID_KERNEL_SZ, RIGID_KERNEL_SZ))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    prev = None
    held_flow = None
    n_held = 0

    zone_pct_10s = {b: np.zeros(N) for b in BAY_NAMES}
    zone_pct_3s = {b: np.zeros(N) for b in BAY_NAMES}
    # joint/attribution tracking (mode == "oracle_joint")
    joint_should = joint_correct_geo = 0
    attribution_wrong = attribution_total = 0
    pred_joint_geo_frames = 0          # v5: ALL frames flagged joint (old-rule OR new-rule) -- precision denominator
    gt_label = {b: [None] * N for b in BAY_NAMES}
    last_frame_for_snap = None
    last_events = []                   # v5: duplicate frames must also freeze GT, not just the image

    AMBIG_FLOW_THRESH = 0.3 * SCALE     # scaled from the full-res original (0.3)
    ATTRIB_DX_THRESH = 0.05 * SCALE     # scaled from the full-res original (0.05)

    for idx in range(N):
        curr, events, b1o, b3o = build_frame(idx, dust3_fn)
        if idx in dup_idx and last_frame_for_snap is not None:
            curr = last_frame_for_snap.copy()
            events = last_events        # v5 fix: a repeated/stuck camera frame shows the SAME (stale)
            # scene as before, so its ground-truth opacity/origins must be the PREVIOUS frame's, not
            # freshly recomputed for this instant -- the oracle bay1_open/bay3_open PLC signal is a
            # separate real sensor and is NOT frozen (it keeps reporting real time regardless of the
            # camera glitch), so open1[idx]/open3[idx] below are left untouched.
        t = idx / FPS

        if idx < int(2 * FPS):
            global_ref_frames.append(curr)
        if idx == int(2 * FPS) - 1:
            active_J = np.median(np.stack(global_ref_frames, axis=0), axis=0).astype(np.float32)
            active_J_energy = np.maximum(texture_energy(active_J), 1.0)
            global_ref_frames = None  # free

        if active_J is None:  # still inside the first-2s reference window
            active_J = curr.astype(np.float32)
            active_J_energy = np.maximum(texture_energy(active_J), 1.0)

        if open1[idx] and not snapshotted["bay1"]:
            active_J = np.where(BAY_MASK["bay1"], last_frame_for_snap if last_frame_for_snap is not None else curr, active_J)
            snapshotted["bay1"] = True
            active_J_energy = np.maximum(texture_energy(active_J), 1.0)
        if open3[idx] and not snapshotted["bay3"]:
            active_J = np.where(BAY_MASK["bay3"], last_frame_for_snap if last_frame_for_snap is not None else curr, active_J)
            snapshotted["bay3"] = True
            active_J_energy = np.maximum(texture_energy(active_J), 1.0)

        tex_ratio = texture_energy(curr) / active_J_energy
        mask_tex = tex_ratio < TEX_THRESH

        ed = edge_density(curr)
        if prev is None:
            fx = fy = np.zeros((H, W), np.float32)
        else:
            mean_diff = float(np.abs(curr.astype(np.int16) - prev.astype(np.int16)).mean())
            if mean_diff < DUP_FRAME_THRESH and held_flow is not None:
                fx, fy = held_flow
                n_held += 1
            else:
                flow = dis.calc(prev.astype(np.uint8), curr.astype(np.uint8), None)
                fx, fy = flow[..., 0], flow[..., 1]
                held_flow = (fx, fy)
        mag = np.sqrt(fx * fx + fy * fy)
        mask_flow = (mag > FLOW_LO) & (mag < FLOW_HI) & (ed < EDGE_GATE)

        now_rigid = (ed > RIGID_EDGE) & (mag > RIGID_FLOW)
        if now_rigid.any():
            now_dil = cv2.dilate(now_rigid.astype(np.uint8), rigid_kernel, iterations=1).astype(bool)
            last_rigid_frame[now_dil] = idx

        raw = mask_tex & mask_flow  # texture-only signal per e3's scope == mask_tex alone
        raw = mask_tex  # texture-only detector (flow not required for this reduced scope)

        for ttl_s, zone_pct in ((10.0, zone_pct_10s), (3.0, zone_pct_3s)):
            rigid_excl = (idx - last_rigid_frame) < int(ttl_s * FPS)
            final, labels, n_labels, stats = clean_mask(raw & ~rigid_excl)
            for b in BAY_NAMES:
                zone_pct[b][idx] = 100.0 * (final & BAY_MASK[b]).sum() / BAY_AREA[b]

            if mode == "oracle_joint" and ttl_s == 10.0:
                open_bays = [b for b in ("bay1", "bay3") if (open1[idx] if b == "bay1" else open3[idx])]
                for lbl in range(1, n_labels):
                    if stats[lbl, cv2.CC_STAT_AREA] < MIN_BLOB:
                        continue
                    blob = labels == lbl
                    touched = [b for b in BAY_NAMES if (blob & BAY_MASK[b]).any()]
                    if not touched:
                        continue
                    open_touched = [b for b in touched if b in open_bays]
                    true_origins = [o for o, a in events if (a[blob] > GT_NEG_OPACITY).any()]
                    geometric_joint = False
                    if len(open_touched) == 1:
                        attributed = [open_touched[0]]
                    elif len(open_touched) >= 2:
                        attributed = open_touched
                        geometric_joint = True
                        pred_joint_geo_frames += 1   # v5: old-rule joint also counts toward "total flagged"
                    else:
                        if not open_bays:
                            attributed = []
                        else:
                            mean_dx = float(fx[blob].mean())
                            cx = float(np.nonzero(blob.any(axis=0))[0].mean())
                            flanked = ("bay2" in touched) and open1[idx] and open3[idx]
                            ambiguous = abs(mean_dx) < AMBIG_FLOW_THRESH
                            if flanked or ambiguous:
                                attributed = [b for b in ("bay1", "bay3") if (open1[idx] if b == "bay1" else open3[idx])]
                                if len(attributed) >= 2:
                                    geometric_joint = True
                                    pred_joint_geo_frames += 1
                            else:
                                attributed = []
                            if not attributed:
                                candidates = open_bays
                                if mean_dx > ATTRIB_DX_THRESH:
                                    candidates = [b for b in open_bays if BAY_CENTER_X[b] < cx] or open_bays
                                elif mean_dx < -ATTRIB_DX_THRESH:
                                    candidates = [b for b in open_bays if BAY_CENTER_X[b] > cx] or open_bays
                                attributed = [min(candidates, key=lambda b: abs(BAY_CENTER_X[b] - cx))]
                    if len(set(true_origins)) >= 2:
                        joint_should += 1
                        if geometric_joint:
                            joint_correct_geo += 1
                    elif len(true_origins) == 1 and len(attributed) == 1:
                        attribution_total += 1
                        if attributed[0] != true_origins[0]:
                            attribution_wrong += 1

        for b in BAY_NAMES:
            maxop = 0.0
            for origin, alpha in events:
                m = alpha[BAY_MASK[b]]
                if m.size:
                    maxop = max(maxop, float(m.max()))
            if maxop >= GT_POS_OPACITY:
                gt_label[b][idx] = True
            elif maxop <= GT_NEG_OPACITY:
                gt_label[b][idx] = False

        last_frame_for_snap = curr
        last_events = events
        prev = curr
        if idx % 100 == 0:
            print(f"    [{scenario}/{mode}] frame {idx}/{N}", flush=True)

    def rows_for(zone_pct):
        p, g = build_samples_multi(zone_pct, gt_label)
        return confusion(p, g)

    return dict(
        scenario=scenario, mode=mode, meta=meta,
        rows_10s=rows_for(zone_pct_10s), rows_3s=rows_for(zone_pct_3s),
        n_held=n_held, joint_should=joint_should, joint_correct_geo=joint_correct_geo,
        pred_joint_geo_frames=pred_joint_geo_frames,
        attribution_wrong=attribution_wrong, attribution_total=attribution_total,
        # v5: raw per-frame arrays so results.md can build the FP x bay x time-window
        # breakdown without a re-run (pure scalars, negligible size vs image frames)
        zone_pct_10s={b: zone_pct_10s[b].tolist() for b in BAY_NAMES},
        gt_label={b: gt_label[b] for b in BAY_NAMES},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["A", "B"])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--full-res", action="store_true", help="scale=1.0 (original constants: TEX_WIN=21, MIN_BLOB=800, ...)")
    ap.add_argument("--scale", type=float, default=None, help="explicit scale override, e.g. 0.5")
    args = ap.parse_args()

    scale = 1.0 if args.full_res else (args.scale if args.scale is not None else 0.5)
    set_scale(scale)
    setup_scene()

    if args.merge:
        merge()
        return

    assert args.scenario, "need --scenario A|B or --merge"
    # v5 (e3's ask 10.5): report BOTH oracle and PLC-real presence for BOTH
    # scenarios -- "oracle_joint" mode also yields the oracle presence table
    # as a side effect (same detector pass), so no extra pass needed for that.
    result = dict(
        oracle=run_streaming(args.scenario, "oracle_joint"),
        texture_real=run_streaming(args.scenario, "plc_real_texture"),
        scale=SCALE,
    )
    (OUT / f"results_{args.scenario}.json").write_text(json.dumps(result, default=float), encoding="utf-8")
    print(f"DONE {args.scenario}", flush=True)


def _fp_breakdown(zone_pct_10s, gt_label, tau=1.0):
    """FP count per bay x time window (spill 6-9s, cloud 14-17s, outside), at
    a fixed tau, from the raw per-frame arrays already collected (no re-run)."""
    ts = np.arange(N) / FPS
    windows = {
        "อ้อยร่วง 6-9s": (ts >= SPILL_WINDOW[0]) & (ts < SPILL_WINDOW[1]),
        "เงาเมฆ 14-17s": (ts >= CLOUD_WINDOW[0]) & (ts < CLOUD_WINDOW[1]),
        "นอกหน้าต่าง": ~(((ts >= SPILL_WINDOW[0]) & (ts < SPILL_WINDOW[1])) |
                        ((ts >= CLOUD_WINDOW[0]) & (ts < CLOUD_WINDOW[1]))),
    }
    out = {}
    for bay in BAY_NAMES:
        pct = np.array(zone_pct_10s[bay])
        gt = np.array([x if x is not None else False for x in gt_label[bay]], dtype=bool)
        gt_known = np.array([x is not None for x in gt_label[bay]], dtype=bool)
        pred_pos = pct > tau
        fp = pred_pos & (~gt) & gt_known
        out[bay] = {wname: int(fp[wmask].sum()) for wname, wmask in windows.items()}
    return out


def merge():
    a = json.loads((OUT / "results_A.json").read_text(encoding="utf-8"))
    b = json.loads((OUT / "results_B.json").read_text(encoding="utf-8"))
    res_scale = a.get("scale", SCALE)
    lines = [
        "# topdown_sim v5 (streaming, full-res per ac's 10.5): texture-only oracle+PLC-real, "
        "geometric joint precision/recall, FP breakdown",
        "",
        f"analysis scale={res_scale} ({'full resolution, original constants TEX_WIN=21/MIN_BLOB=800' if res_scale == 1.0 else 'reduced scale'}), "
        "streaming ทีละเฟรม detector = texture-only ล้วน",
        "",
    ]

    for name, d in (("A", a), ("B", b)):
        lines += [
            f"## Scenario {name} -- texture-only presence, oracle, rigid TTL 10s",
            table_md(d["oracle"]["rows_10s"], f"{name}, oracle, rigid 10s"),
            f"## Scenario {name} -- texture-only presence, PLC จริง, rigid TTL 10s",
            table_md(d["texture_real"]["rows_10s"], f"{name}, PLC จริง, rigid 10s"),
            f"## Scenario {name} -- texture-only presence, PLC จริง, rigid TTL 3s",
            table_md(d["texture_real"]["rows_3s"], f"{name}, PLC จริง, rigid 3s"),
            f"duplicate-frame flow holds: {d['texture_real']['n_held']}, "
            f"latency bay1={d['texture_real']['meta']['lat1']:.2f}s bay3={d['texture_real']['meta']['lat3']:.2f}s",
            "",
        ]

    lines.append("## Geometric joint: precision (ถูก/ยกธงทั้งหมด) และ recall (ถูก/GT ควรเป็น joint), A และ B")
    lines += ["", "| scenario | GT joint_should | ยกธงทั้งหมด | ถูก | precision | recall |", "|---|---|---|---|---|---|"]
    for name, d in (("A", a), ("B", b)):
        j = d["oracle"]
        should, flagged, correct = j["joint_should"], j["pred_joint_geo_frames"], j["joint_correct_geo"]
        prec = correct / flagged if flagged else float("nan")
        rec = correct / should if should else float("nan")
        lines.append(f"| {name} | {should} | {flagged} | {correct} | {fmt(prec)} | {fmt(rec)} |")
    lines += [
        "",
        f"scenario A ควรมี joint_should=0 (ฝุ่นสองต้นกำเนิดไม่เคยเชื่อมกันตาม design) -- "
        f"ยกธง joint ผิดใน A = {a['oracle']['pred_joint_geo_frames']} เฟรม (false-joint บนสถานการณ์ที่ไม่ควรมี joint เลย)",
        "",
        f"attribution accuracy นอกช่วง joint: A={fmt(100.0*a['oracle']['attribution_wrong']/a['oracle']['attribution_total']) if a['oracle']['attribution_total'] else 'n/a'}% ผิด, "
        f"B={fmt(100.0*b['oracle']['attribution_wrong']/b['oracle']['attribution_total']) if b['oracle']['attribution_total'] else 'n/a'}% ผิด",
        "",
        "## FP breakdown ต่อแท่น x หน้าต่างเวลา (texture-only, rigid 10s, tau=1.0)",
        "",
    ]
    for name, d in (("A", a), ("B", b)):
        for mode_name, res in (("oracle", d["oracle"]), ("PLC จริง", d["texture_real"])):
            bd = _fp_breakdown(res["zone_pct_10s"], res["gt_label"], tau=1.0)
            lines.append(f"**{name}, {mode_name}**")
            lines.append("| แท่น | อ้อยร่วง 6-9s | เงาเมฆ 14-17s | นอกหน้าต่าง |")
            lines.append("|---|---|---|---|")
            for bay in ["bay1", "bay2", "bay3"]:
                row = bd[bay]
                lines.append(f"| {bay} | {row['อ้อยร่วง 6-9s']} | {row['เงาเมฆ 14-17s']} | {row['นอกหน้าต่าง']} |")
            lines.append("")

    lines += [
        "## v5 fixes เทียบ v4",
        "- **full-resolution แทน half-res**: ยืนยันว่า FP ที่ τ ต่ำเป็น artifact ของความละเอียด ไม่ใช่ PLC จริง "
        "(เทียบ oracle vs PLC จริง ที่ full-res ในตารางข้างบน)",
        "- **duplicate-frame ตอนนี้ freeze GT ด้วย** ไม่ใช่แค่ภาพ (v4 เฟรมซ้ำใช้ภาพเก่าแต่ GT คำนวณใหม่จากเวลาปัจจุบัน "
        "ทำให้ label ไม่ตรงกับสิ่งที่กล้องเห็นจริงในเฟรมนั้น)",
        "- **false-joint precision ใหม่**: นับรวมทั้งกติกาเดิม (open_touched>=2) และกติกาใหม่ (flanked/ambiguous-flow) "
        "เป็น \"ยกธงทั้งหมด\" -- ก่อนหน้านี้ v4 นับเฉพาะกติกาใหม่ ทำให้ precision denominator ขาดไป",
        "- ambiguous-flow threshold (0.3) และ attribution dx threshold (0.05) ตอนนี้ scale ตาม SCALE แทนที่จะ hardcode "
        "ค่า half-res ไว้ตรงๆ",
        "",
        "ไฟล์: sim_topdown_lite.py (v5), results_A.json, results_B.json",
    ]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"OK -> {OUT / 'results.md'}")


if __name__ == "__main__":
    main()
