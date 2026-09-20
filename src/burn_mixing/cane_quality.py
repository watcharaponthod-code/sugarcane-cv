# -*- coding: utf-8 -*-
"""Model B v0 - per-truck cane quality from ONE snapshot, plain computer vision, nothing trained.

Answers the factory's tree (memory: cane-quality-taxonomy) from the pile that sits on the table after the dump:
    pile mask   = pixels inside the bay that changed against the latched clean reference (same reference the dust veto
                  keeps), minus the red truck body, cleaned with open/close
    black %     = share of pile pixels that are dark AND unsaturated  -> สดแท้ / ไฟไหม้ปน / ตัดไฟไหม้
    green %     = share of pile pixels with leaf hue                  -> ใบ น้อย / กลาง / มาก   (burnt cane has no green)
    stalk lines = LSD line segments on the pile: median length in px  -> ลำ (คนตัด, long) vs ท่อน (รถตัด, short)
Every number is printed before it is bucketed, and every bucket edge is a CLI knob - set them by eye on the first real
footage, replace with a logistic regression once weighbridge tickets give a few hundred labelled trucks.

Usage (logic test on the AI clips):
    python cane_quality.py --src ../../gen_video/burnt/clip7_burnt_nodust.mp4 --ref-t 0 --t 5,8,9.5 --bay bay2
    python cane_quality.py --src ../../gen_video/round3/sugarcane_unloading_cctv.mp4 --ref-t 0 --t 3,9.5 --bay bay2
Writes qa/own_model/cane_quality/<clip>_<t>.jpg (pile outline, black=red tint, green=green tint, line segments) and prints
one JSON line per snapshot."""
import sys, json, argparse
from pathlib import Path
import numpy as np, cv2

HERE = Path(__file__).parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from eval_seg_lib import truck_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--ref-t", type=float, default=0.0, help="time of the clean reference frame (before the truck)")
ap.add_argument("--t", default="9.5", help="snapshot times, comma separated")
ap.add_argument("--bay", default="bay2")
ap.add_argument("--zones", default=str(ROOT / "gen_images" / "eval_set.json"))
ap.add_argument("--zone-key", default="sugarcane_unloading_cctv.mp4")
ap.add_argument("--diff-min", type=float, default=25.0, help="pile = |cur-ref| gray above this")
ap.add_argument("--black-v", type=int, default=70); ap.add_argument("--black-s", type=int, default=90)
ap.add_argument("--tex-min", type=float, default=4.0, help="black counts only where the surface has texture (box|Lap| above this): a charred pile is dark AND streaky, the shadowed truck-bed floor is dark AND flat")
ap.add_argument("--green-h", default="35,85"); ap.add_argument("--green-s", type=int, default=60); ap.add_argument("--green-v", type=int, default=50)
ap.add_argument("--bright-v", type=int, default=110, help="a stalk pixel brighter than this = unburnt cane surface")
ap.add_argument("--burn-edges", default="8,15", help="BRIGHT%% edges (share of pile brighter than --bright-v): <a ตัดไฟไหม้, a-b ไฟไหม้ปน, >b สดแท้. Chosen over black%% because a fresh load in the truck-bed shadow is as dark as char, but a charred pile has NO bright stalks anywhere (AI clips: fresh 30-40%%, burnt 0-5%%)")
ap.add_argument("--leaf-edges", default="5,20", help="green%% edges: น้อย/กลาง/มาก")
ap.add_argument("--stalk-len", type=float, default=0.0, help="median line length (px, at 1280 wide) above which = ลำ; 0 = print only")
ap.add_argument("--out", default=str(HERE / "cane_quality"))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(exist_ok=True)
E = json.load(open(a.zones, encoding="utf-8"))
clip = E["clips"][a.zone_key]
gh0, gh1 = (int(x) for x in a.green_h.split(",")); b0, b1 = (float(x) for x in a.burn_edges.split(",")); l0, l1 = (float(x) for x in a.leaf_edges.split(","))


def grab(cap, t):
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
    if not ok: sys.exit(f"cannot read t={t}")
    return f


def lsd(gray):
    try:
        d = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = d.detect(gray)[0]
        return np.zeros((0, 4), np.float32) if lines is None else lines.reshape(-1, 4)
    except Exception:                                                 # LSD missing in this cv2 build -> probabilistic Hough
        e = cv2.Canny(gray, 60, 160)
        h = cv2.HoughLinesP(e, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=4)
        return np.zeros((0, 4), np.float32) if h is None else h.reshape(-1, 4).astype(np.float32)


cap = cv2.VideoCapture(a.src)
ref = grab(cap, a.ref_t); H, W = ref.shape[:2]
bay = cv2.fillPoly(np.zeros((H, W), np.uint8), [np.array(clip["roi_polygons"][a.bay], np.int32)], 1).astype(bool)
ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
name = Path(a.src).stem
for t in (float(x) for x in a.t.split(",")):
    f = grab(cap, t)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    changed = (np.abs(g - ref_g) > a.diff_min) & bay
    if clip.get("truck_mask"): changed &= ~truck_mask(f, clip["truck_mask"])
    m = changed.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:                                                         # keep the largest blob = the pile
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); pile = lab == k
    else:
        pile = np.zeros((H, W), bool)
    npx = int(pile.sum()); area_pct = 100.0 * npx / max(int(bay.sum()), 1)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    tex = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), -1, (9, 9))
    black = pile & (V < a.black_v) & (S < a.black_s) & (tex > a.tex_min)
    flat_dark = pile & (V < a.black_v) & (S < a.black_s) & (tex <= a.tex_min)          # shadow / bed floor, reported, not counted
    green = pile & (Hh >= gh0) & (Hh <= gh1) & (S >= a.green_s) & (V >= a.green_v)
    yellow = pile & (Hh >= 15) & (Hh < 35) & (S >= 50) & (V >= 80)
    bright = pile & (V > a.bright_v)
    black_pct = 100.0 * black.sum() / max(npx, 1); shadow_pct = 100.0 * flat_dark.sum() / max(npx, 1); green_pct = 100.0 * green.sum() / max(npx, 1); yellow_pct = 100.0 * yellow.sum() / max(npx, 1); bright_pct = 100.0 * bright.sum() / max(npx, 1)
    # stalk geometry: line segments whose midpoint lies on the pile
    L = lsd(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    if len(L):
        mx, my = ((L[:, 0] + L[:, 2]) / 2).astype(int).clip(0, W - 1), ((L[:, 1] + L[:, 3]) / 2).astype(int).clip(0, H - 1)
        on = pile[my, mx]; L = L[on]
    lens = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]) * (1280.0 / W) if len(L) else np.zeros(0)
    ang = np.degrees(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0])) % 180 if len(L) else np.zeros(0)
    med_len = float(np.median(lens)) if len(lens) else 0.0; p90_len = float(np.percentile(lens, 90)) if len(lens) else 0.0
    # orientation coherence: long parallel stalks -> one dominant angle; billets -> flat histogram
    coh = 0.0
    if len(ang) >= 10:
        hist = np.histogram(ang, bins=18, range=(0, 180), weights=lens)[0]; coh = float(hist.max() / max(hist.sum(), 1e-6))
    burn = "no pile" if npx < 2000 else ("ตัดไฟไหม้" if bright_pct < b0 else "ไฟไหม้ปน" if bright_pct < b1 else "สดแท้")
    leaf = "n/a" if npx < 2000 else ("น้อย" if green_pct < l0 else "กลาง" if green_pct < l1 else "มาก")
    cut = "n/a" if (npx < 2000 or not a.stalk_len) else ("ลำ (คนตัด)" if med_len >= a.stalk_len else "ท่อน (รถตัด)")
    rec = dict(clip=name, t=t, bay=a.bay, pile_px=npx, pile_area_pct=round(area_pct, 1), bright_pct=round(bright_pct, 1), black_pct=round(black_pct, 1), shadow_pct=round(shadow_pct, 1), green_pct=round(green_pct, 1),
               yellow_pct=round(yellow_pct, 1), lines=int(len(lens)), line_len_median=round(med_len, 1), line_len_p90=round(p90_len, 1),
               orient_coherence=round(coh, 2), burn=burn, leaf=leaf, cut=cut)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    # picture
    v = f.copy()
    v[black] = (0.4 * v[black] + [0, 0, 150]).astype(np.uint8); v[flat_dark] = (0.5 * v[flat_dark] + [120, 0, 120]).astype(np.uint8); v[green] = (0.4 * v[green] + [0, 150, 0]).astype(np.uint8); v[bright] = (0.5 * v[bright] + [0, 120, 120]).astype(np.uint8)
    cnts, _ = cv2.findContours(pile.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v, cnts, -1, (0, 255, 255), 2)
    for x0, y0, x1, y1 in L[:400].astype(int): cv2.line(v, (x0, y0), (x1, y1), (255, 200, 0), 1)
    txt = f"{name} t={t}s  pile {area_pct:.0f}% of bay | bright {bright_pct:.0f}% black {black_pct:.0f}% -> {burn} | green {green_pct:.0f}% -> leaf {leaf} | lines {len(lens)} med {med_len:.0f}px coh {coh:.2f}"
    cv2.rectangle(v, (0, 0), (W, 34), (0, 0, 0), -1); cv2.putText(v, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"{name}_{t}.jpg"), v, [cv2.IMWRITE_JPEG_QUALITY, 85])
