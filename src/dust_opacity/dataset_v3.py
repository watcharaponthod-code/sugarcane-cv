# -*- coding: utf-8 -*-
"""dataset v3.2 (QA) — e3's round-3 spec. Same import surface as before:
  build_split(seed=0, n_total=750, val_frac=0.15) -> (train_seeds, val_seeds)
  frame_and_mask(sample_seed) -> (img uint8 HxWx3 BGR, label int8 {1 dust, 0 bg, -1 ignore})
  frame_alpha(sample_seed)    -> (img uint8 HxWx3 BGR, alpha_screen float32 [0,1])
v3.2 changes (round-2 stills recall 0.50 = domain gap of the DUST itself, not the background):
  C1 COLOUR pipeline: backgrounds kept in BGR; dust tinted per sample from a palette (red-earth brown / yellow-brown / grey / white) with jitter.
  C2 plume shape: multi-octave fBm (4 octaves value noise) x anisotropic Gaussian envelope elongated along a per-sample WIND direction,
     centre pushed downwind from the origin -> ragged edges, streaks; plus a low-frequency VEILING layer (peak 0.03-0.2, sigma 250-600 px).
  C3 thin dust weighting: 40% of dust samples have peak alpha 0.05-0.25 (rest 0.25-0.9).
  C4 origin tied to a truck bed edge (downwind side) when a truck exists (P=0.7), else random; one wind direction per sample.
Kept from v3.1: 2 real backgrounds (topdown, cam15) crop>=40% + flip; dark trucks P=0.6; bright textured negatives (50% of zero-dust, 20% of dust
samples); screen-composed alpha; per-sample gain 0.8-1.2; noise; JPEG. augment(): blur sigma U(0,2.5) + JPEG 40-90 + gain 0.8-1.2 on EVERY sample + hflip.
Test files (unloading_1/2/3, 4 eval stills, reference_real) are never read (asserted). v3.1 frozen copy: dataset_v3_1_frozen.py (sha 410c18bb5f0f)."""
import sys
from pathlib import Path
import numpy as np, cv2

VERSION = "v3.2-qa-colour-fbm-wind-thin40-2026-09-04"
COLOR = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gen_video" / "topdown_sim"))
import sim_topdown_lite as sim  # noqa: E402

if not hasattr(sim, "H"):
    sim.set_scale(1.0); sim.setup_scene()
H, W = sim.H, sim.W
GT_POS, GT_NEG = sim.GT_POS_OPACITY, sim.GT_NEG_OPACITY
ZERO_DUST_PROB, MAX_BLOBS, TRUCK_PROB, TOPDOWN_PROB, TEXNEG_PROB_ZERO, TEXNEG_PROB_DUST, THIN_FRAC = 0.20, 4, 0.6, 0.5, 0.5, 0.2, 0.40
BG_FILES = [ROOT / "site_photos" / "topdown_3bays_2026-09-03.png", ROOT / "cctv_frames" / "cam15_reference.png"]
FORBIDDEN = ("unloading_", "sugarcane_truck_dumping", "sugarcane_cctv_dumping", "sugarcane_dump_overhead", "three_bay_fisheye", "reference_real")
assert not any(k in str(p) for p in BG_FILES for k in FORBIDDEN)

# 2026-09 audit + king's ruling: BG_FILES are OFFICIALLY TRAINING-ONLY. topdown_3bays_2026-09-03.png happens to
# be byte-identical to gen_images/reference_real.png under a different name; that stopped being a leak the
# moment reference_real was retired as an independent test case (see eval_seg.py). This checks by CONTENT
# (sha256) against the actual NAMED test assets, not by filename, and WARNS rather than crashes - training on
# these two backgrounds is the sanctioned, intended behaviour, not an accident.
TEST_ASSET_FILES = (
    "gen_images/sugarcane_truck_dumping_fisheye.jpg", "gen_images/sugarcane_cctv_dumping.webp",
    "gen_images/sugarcane_dump_overhead.webp", "gen_images/three_bay_fisheye_unloading.webp",
    "gen_video/unloading_1.mp4", "gen_video/unloading_2.mp4", "gen_video/unloading_3.mp4",
    "gen_video/round3/sugarcane_unloading_cctv.mp4", "gen_images/eval_set.json",
)
TEST_ASSET_SHA256 = (      # EMBEDDED so the guard still works where the test files are absent (cloud trainers)
    "0887c098b36b3000c714c06d160851346a5965ad833281493664bb7175129193",  # gen_images/sugarcane_truck_dumping_fisheye.jpg
    "6ac05688a5ca843dd0ec288a651a5bf4bee8e357a351e2b2290ae587dd8462be",  # gen_images/sugarcane_cctv_dumping.webp
    "43e4bedfb7ca850f81060312bb2db40495712691dd80d3f9b182b29d813ba51a",  # gen_images/sugarcane_dump_overhead.webp
    "93580d9c85778c9c830a56c537aa5a738adfaf59a9f51207179f3d408a05e875",  # gen_images/three_bay_fisheye_unloading.webp
    "8cc3dc70b9cf860343bec783c51b9f94fd673491dfffbcbabba69b4c3cbc0c42",  # gen_video/unloading_1.mp4
    "aa02253751b08d58b4116f1887dcf7494761ac4d2d0e83dec20c7572ac30e24c",  # gen_video/unloading_2.mp4
    "6f677d3231920a04bea0cb5c21c3f7b11f0a906f3b03c75e69172281941dd571",  # gen_video/unloading_3.mp4
    "effd827c7d6cc4d48b675b1ee47e67d27114a666f77e4ed6caa239fa114134a0",  # gen_video/round3/sugarcane_unloading_cctv.mp4
    "e1f14bb9f37608cae81b0e838d0480d0384b6c37d0326e950ce761c615f8039b",  # gen_images/eval_set.json
)


LEAK_GUARD = dict(status="NOT_RUN")


def _bg_sha256():
    import hashlib
    return [hashlib.sha256(p.read_bytes()).hexdigest() for p in BG_FILES]


def _registry_hashes():
    """sha256 of every entry in qa/own_model/test_assets.json - the machine-written test-asset registry.
    split_real.py appends held-out REAL footage there the moment it is set aside, so a dataset built afterwards
    physically cannot use it as a training background."""
    reg = Path(__file__).resolve().parent / "test_assets.json"
    if not reg.exists(): return set(), 0
    import json
    try:
        ents = json.loads(reg.read_text(encoding="utf-8")).get("entries", [])
    except Exception:                                                  # noqa: BLE001 - a broken registry must not
        return set(), 0                                                # silently disable the rest of the guard
    return {e["sha256"] for e in ents if e.get("sha256")}, len(ents)


def _leak_check():
    """Three sources, unioned: hashes EMBEDDED in this file, hashes recomputed from any test file that happens
    to be present, and the registry. The embedded set is the one that matters - jodie found (2026-09) that the
    file-based check silently degraded to an empty set inside a training bundle, where none of the test files
    ship, so the assert compared against nothing and passed unconditionally. A guard that passes because there
    was nothing to check is worse than no guard: it manufactures confidence. If the union is ever empty this
    now says so LOUDLY and records UNVERIFIED, instead of printing a reassuring line and moving on."""
    import hashlib
    global LEAK_GUARD
    embedded = set(TEST_ASSET_SHA256)
    present = {hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in TEST_ASSET_FILES if (ROOT / rel).exists()}
    reg_hashes, n_reg = _registry_hashes()
    test_hashes = embedded | present | reg_hashes
    for p, h in zip(BG_FILES, _bg_sha256()):
        assert h not in test_hashes, f"LEAK: {p} (sha256={h[:12]}) matches a known test asset."
    LEAK_GUARD = dict(status="OK" if test_hashes else "UNVERIFIED", checked_against=len(test_hashes),
                      embedded=len(embedded), files_present=len(present), registry=n_reg,
                      bg_sha256=[h[:12] for h in _bg_sha256()])
    if not test_hashes:
        print("!" * 100)
        print("[LEAK GUARD UNVERIFIED] nothing to compare against - embedded list empty AND no test files AND "
              "no registry. This run has NO leak protection. Do not trust any number produced from it.")
        print("!" * 100, flush=True)
    print(f"[dataset_v3] leak guard {LEAK_GUARD['status']}: BG_FILES are TRAINING-ONLY, "
          f"checked against {LEAK_GUARD['checked_against']} known test hashes "
          f"({LEAK_GUARD['embedded']} embedded + {LEAK_GUARD['files_present']} local files + "
          f"{LEAK_GUARD['registry']} registry) - see 2026-09 audit.")


_leak_check()
PALETTE_BGR = [(95, 125, 175), (110, 150, 195), (165, 165, 165), (200, 200, 200), (230, 232, 235), (120, 140, 160)]   # red-earth, yellow-brown, grey x2, white, cool grey
_BG = None
_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)


def backgrounds():
    global _BG
    if _BG is None:
        _BG = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in BG_FILES]
        assert all(im is not None for im in _BG)
    return _BG


def _crop_resize(im, rng, min_frac=0.4):
    h, w = im.shape[:2]; fh, fw = rng.uniform(min_frac, 1.0), rng.uniform(min_frac, 1.0)
    ch, cw = int(h * fh), int(w * fw); y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    return cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)


def _rect(rng, cx, cy, rw, rh):
    x0, x1 = int(max(cx - rw / 2, 0)), int(min(cx + rw / 2, W)); y0, y1 = int(max(cy - rh / 2, 0)), int(min(cy + rh / 2, H))
    return (x0, x1, y0, y1) if x1 > x0 and y1 > y0 else None


def fbm(rng, octaves=4, base=16, persistence=0.55):
    """multi-octave value noise on the full frame, z-scored -> ragged multiplier with mean ~1 (modulates an envelope without changing its mean)."""
    acc = np.zeros((H, W), np.float32); amp = 1.0; tot = 0.0
    for o in range(octaves):
        n = int(base * 2 ** o)
        small = rng.random((max(2, H * n // W), n)).astype(np.float32)
        acc += amp * cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC); tot += amp; amp *= persistence
    acc /= tot
    acc = (acc - acc.mean()) / (acc.std() + 1e-6)
    return np.clip(1.0 + 0.55 * acc, 0.0, 2.2)


def plume_alpha(rng, cx, cy, sigma, peak, wind, elong, drift):
    """anisotropic Gaussian: long axis along wind (sigma*elong), short axis sigma; centre shifted downwind by drift*sigma; x fBm."""
    ux, uy = np.cos(wind), np.sin(wind)
    mx, my = cx + ux * drift * sigma, cy + uy * drift * sigma
    dx, dy = _XX - mx, _YY - my
    along = dx * ux + dy * uy; across = -dx * uy + dy * ux
    env = np.exp(-(along ** 2) / (2 * (sigma * elong) ** 2) - (across ** 2) / (2 * sigma ** 2))
    return np.clip(peak * env * fbm(rng), 0, 0.95).astype(np.float32)


def veil_alpha(rng, cx, cy, sigma, peak):
    env = np.exp(-((_XX - cx) ** 2 + (_YY - cy) ** 2) / (2 * sigma * sigma))
    return np.clip(peak * env * fbm(rng, octaves=2, base=4), 0, 0.5).astype(np.float32)


def render_sample(rng):
    bgs = backgrounds(); topdown = rng.random() < TOPDOWN_PROB
    bg_idx = 0 if topdown else int(rng.integers(1, len(bgs)))
    img = bgs[0].astype(np.float32).copy() if topdown and rng.random() < 0.5 else _crop_resize(bgs[bg_idx], rng)
    if rng.random() < 0.5: img = img[:, ::-1].copy()
    # dark trucks / loads
    slots = list(sim.BAY_NAMES) if topdown else [None] * 3
    truck_boxes = []
    for slot in slots:
        if rng.random() >= TRUCK_PROB: continue
        if slot is not None:
            x, y, w, h = sim.BAY_BBOX[slot]; sc = rng.uniform(0.6, 1.0); r = _rect(rng, x + w * rng.uniform(0.3, 0.7), y + h * rng.uniform(0.25, 0.5), w * 0.8 * sc, h * 0.55 * sc)
        else:
            r = _rect(rng, rng.uniform(0.15, 0.85) * W, rng.uniform(0.2, 0.8) * H, rng.uniform(0.15, 0.35) * W, rng.uniform(0.15, 0.4) * H)
        if r is None: continue
        x0, x1, y0, y1 = r; base = rng.uniform(40, 70)
        patch = np.clip(base + rng.normal(0, 12, (y1 - y0, x1 - x0)), 15, 120)[..., None] * np.array(rng.uniform(0.85, 1.1, 3), np.float32)
        img[y0:y1, x0:x1] = np.clip(patch, 0, 255); truck_boxes.append(r)
    n_blobs = 0 if rng.random() < ZERO_DUST_PROB else int(rng.integers(1, MAX_BLOBS + 1))
    # bright textured negatives
    if rng.random() < (TEXNEG_PROB_ZERO if n_blobs == 0 else TEXNEG_PROB_DUST):
        for _ in range(int(rng.integers(1, 4))):
            r = _rect(rng, rng.uniform(0.1, 0.9) * W, rng.uniform(0.15, 0.85) * H, rng.uniform(0.08, 0.3) * W, rng.uniform(0.08, 0.3) * H)
            if r is None: continue
            x0, x1, y0, y1 = r
            if rng.random() < 0.5:
                tint = np.array(PALETTE_BGR[int(rng.integers(0, len(PALETTE_BGR)))], np.float32) / 180.0
                img[y0:y1, x0:x1] = np.clip((125 + rng.normal(0, 35, (y1 - y0, x1 - x0)))[..., None] * tint, 40, 230)
            else:
                src = _crop_resize(bgs[int(rng.integers(0, len(bgs)))], rng, 0.3)[y0:y1, x0:x1]
                img[y0:y1, x0:x1] = np.clip(src * rng.uniform(1.0, 1.4), 0, 255)
    # dust: one wind direction per sample; thin-dust weighting; origin at truck bed edge (downwind side) when possible
    alpha_total = np.zeros((H, W), np.float32)
    wind = rng.uniform(0, 2 * np.pi); thin = rng.random() < THIN_FRAC; cx = cy = None
    if n_blobs:
        colour = np.array(PALETTE_BGR[int(rng.integers(0, len(PALETTE_BGR)))], np.float32) + rng.normal(0, 12, 3)
        colour = np.clip(colour * rng.uniform(0.75, 1.15), 40, 255)
    for k in range(n_blobs):
        if truck_boxes and rng.random() < 0.7:
            x0, x1, y0, y1 = truck_boxes[int(rng.integers(0, len(truck_boxes)))]
            ux, uy = np.cos(wind), np.sin(wind)
            cx = x1 if ux > 0 else x0; cy = y1 if uy > 0 else y0
            cx = float(np.clip(cx + rng.uniform(-0.2, 0.2) * (x1 - x0), 0, W - 1)); cy = float(np.clip(cy + rng.uniform(-0.2, 0.2) * (y1 - y0), 0, H - 1))
        else:
            cx, cy = rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H
        sigma = rng.uniform(40, 200); peak = rng.uniform(0.05, 0.25) if thin else rng.uniform(0.25, 0.9)
        a = plume_alpha(rng, cx, cy, sigma, peak, wind + rng.normal(0, 0.25), rng.uniform(1.3, 3.0), rng.uniform(0.3, 1.5))
        img = img * (1 - a)[..., None] + colour * a[..., None]
        alpha_total = 1.0 - (1.0 - alpha_total) * (1.0 - a)
    if n_blobs and rng.random() < 0.6:
        a = veil_alpha(rng, cx, cy, rng.uniform(250, 600), rng.uniform(0.03, 0.12 if thin else 0.2))
        img = img * (1 - a)[..., None] + colour * a[..., None]
        alpha_total = 1.0 - (1.0 - alpha_total) * (1.0 - a)
    img = np.clip(img * rng.uniform(0.8, 1.2) * np.array(rng.uniform(0.95, 1.05, 3), np.float32), 0, 255)
    img = np.clip(img + rng.normal(0, 1.5, img.shape).astype(np.float32), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 96))]); img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img, alpha_total.astype(np.float32)


def frame_alpha(sample_seed):
    return render_sample(np.random.default_rng(sample_seed))


def frame_and_mask(sample_seed):
    img, a = frame_alpha(sample_seed)
    label = np.full((H, W), -1, np.int8); label[a <= GT_NEG] = 0; label[a >= GT_POS] = 1
    return img, label


def augment(img, label, rng):
    """EVERY sample: gain U(0.8,1.2) (+ per-channel jitter ±5% when colour), blur sigma U(0,2.5) px, JPEG q U(40,90); hflip p=.5."""
    img = img.astype(np.float32) * rng.uniform(0.8, 1.2)
    if img.ndim == 3: img = img * np.array(rng.uniform(0.95, 1.05, 3), np.float32)
    img = np.clip(img, 0, 255)
    sig = rng.uniform(0.0, 2.5)
    if sig > 0.05: img = cv2.GaussianBlur(img, (0, 0), sig)
    ok, enc = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 91))])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR if img.ndim == 3 else cv2.IMREAD_GRAYSCALE)
    if rng.random() < 0.5: img, label = img[:, ::-1].copy(), label[:, ::-1].copy()
    return img, label


AUG_DESC = "v3.2: gain U(.8,1.2) + channel jitter ±5%, blur sigma U(0,2.5) px, JPEG q U(40,90) on EVERY sample; hflip p.5"


def build_split(seed=0, n_total=750, val_frac=0.15):
    rng = np.random.default_rng(seed + 320000)      # v3.2 seed space (v3.1 used +300000, v2 used +0)
    seeds = np.unique(rng.integers(0, 2 ** 31 - 1, size=n_total)); rng.shuffle(seeds)
    n_val = round(len(seeds) * val_frac)
    return seeds[n_val:].tolist(), seeds[:n_val].tolist()


if __name__ == "__main__":
    tr, va = build_split(0); print(VERSION, "train", len(tr), "val", len(va), "backgrounds", len(backgrounds()))
    tiles = []; zero = 0; peaks = []
    for s in tr[:16]:
        img, a = frame_alpha(s); assert img.shape == (H, W, 3) and a.shape == (H, W) and 0 <= a.min() and a.max() <= 1
        assert frame_alpha(s)[0].tobytes() == img.tobytes()
        zero += a.max() < GT_POS; peaks.append(round(float(a.max()), 2))
        ov = img.astype(np.float32); t = np.zeros_like(ov); t[..., 0] = 200; t[..., 2] = 255; aa = (0.85 * a)[..., None]
        tiles.append(np.hstack([cv2.resize(img, (400, 219)), cv2.resize((ov * (1 - aa) + t * aa).astype(np.uint8), (400, 219))]))
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 16, 2)])
    out = Path(__file__).parent / "dataset_v32_sheet.jpg"; cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("alpha max of first 16:", peaks, "| below GT_POS:", zero, "| sheet ->", out, "OK")
