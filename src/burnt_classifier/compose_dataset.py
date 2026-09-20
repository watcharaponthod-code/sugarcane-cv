# -*- coding: utf-8 -*-
"""compositor — ประกอบชุดจำลอง `synth_c1` ที่ ground truth แม่นระดับพิกเซล ไม่ต้องพึ่ง labeler

**วัตถุดิบมาจาก g1 กับภาพจริงเท่านั้น — ไม่แตะ g2 (ต้องคง blind ถาวร)**

หลักการ: สุ่มพื้นหลัง → วางลำ cutout ทีละชิ้นในกรอบช่องเท (สุ่ม scale/หมุน/พลิก/ตำแหน่ง)
label เกิดจาก**ลำดับการวาด**: ชิ้นที่วาดทีหลังทับชิ้นก่อน ทั้งภาพและ label จึงตรงกันเสมอ
พิกเซลที่ยังเห็นของแต่ละชิ้นคือคลาสของชิ้นนั้น → ไม่มีพิกเซล "ไม่ชัด" เลยในชุดนี้

label PNG: **0 = พื้น/ผนัง · 1 = ลำสด · 2 = ลำไหม้**  (ไม่มี 255)
นี่คือจุดที่ synth_c1 เหนือกว่า label จากกฎสี: ลำไหม้ถูก label ครบทุกพิกเซล
**รวมไฮไลต์มันวาวบนลำดำ** ซึ่ง labeler เดิมทิ้งเป็น 255 และทำให้โมเดลเรียน "ไหม้" แบบอนุรักษ์นิยม
(ทีม A วัดบน g2 blind: truth 60 → ทาย 34, 55 → 39, 65 → 54 คือทายต่ำเป็นระบบที่ปลายสูง)

การกระจายที่บังคับไว้:
  - burnt_frac ต่อภาพสุ่ม uniform 0-100%
  - **บังคับอย่างน้อย N_MID ใบอยู่ในย่าน 40-80%** เพราะโมเดลปัจจุบัน bias ทายต่ำในย่านนี้
  - วางซ้อนหลายชั้นให้เกิดเคส "ลำสดพาดทับลำไหม้" เยอะ ๆ (เคสที่ถูกทิ้งไป 195 ชิ้นในชุด YOLO)

manifest บันทึก burnt_frac สองแบบ:
  - `burnt_frac_pixel` = พิกเซลคลาส 2 / (คลาส 1 + คลาส 2)  ← สิ่งที่ตาเห็นบนผิวกอง
  - `burnt_frac_stalks` = จำนวนลำไหม้ / จำนวนลำทั้งหมดที่วาง (นับชิ้น ไม่สนว่าถูกบังแค่ไหน)
ทั้งสองต่างกันเพราะการบัง — เก็บไว้ทั้งคู่จะได้รู้ว่าโมเดลเรียนอันไหน

รัน: python compose_dataset.py [จำนวนภาพ]      (ค่าเริ่มต้น 300)
     python compose_dataset.py 36 --preview    ทำชุดตัวอย่าง + montage + histogram + overlay ตรวจ label
"""
import sys, json, random, math
from pathlib import Path
import numpy as np, cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
GI = ROOT / "gen_images"
CUT = GI / "cutouts"
BGS = GI / "backgrounds"
OUT = GI / "synth_c1"

N_DEFAULT = 300
N_MID_FRAC = 0.20            # >=20% ของชุด (60 ใบจาก 300) ต้องอยู่ย่าน burnt 40-80%
MID_RANGE = (40.0, 80.0)
STALKS_MAX = 1100            # เพดานจำนวนชิ้นต่อภาพ (วางจนความหนาแน่นถึงเป้าแล้วหยุด)
FILL_TARGET = 0.85           # ความหนาแน่นของกองในกรอบช่องเท ต้องถึงเท่านี้ก่อนหยุดวาง
INPAINT_MIN_FILL = 0.70      # พื้นหลัง inpaint ใช้ได้เฉพาะเมื่อกองคลุมกรอบ >= เท่านี้ (ปิดรอยเบลอ)
NIGHT_MAX_FRAC = 0.35        # กลางคืนไม่เกิน 35% ของชุด
SHADOW = dict(off=(2, 4), alpha=(0.30, 0.50), blur=3)   # เงาอ่อนใต้ชิ้น
PIECE_FRAC = 0.42            # ด้านยาวของชิ้น = สัดส่วนนี้ของด้านสั้นของช่องเท
                             # (เดิม 1.6 เท่าของ span/ด้านยาว → ชิ้นเดียวกินครึ่งช่อง ภาพออกมาเป็นแผ่นปะติดปะต่อ)
HEAP_SIGMA = 0.30            # วางแบบกระจุกกลางช่อง (gaussian) ไม่ใช่โปรยทั่วกรอบ ให้เป็น "กอง" ไม่ใช่สะเก็ด
SCALE = (0.35, 0.80)         # เดิม 0.6-1.4 ชิ้นใหญ่เกิน ภาพดูเป็นแผ่นแปะ
BURNT_DEPTH_BIAS = 0.0       # 0 = สลับชั้นแบบสุ่มล้วน
                             # วัดแล้ว: ที่ค่า 0.30 พิกเซลไหม้ที่มองเห็นเหลือแค่ ~0.6 เท่าของสัดส่วนลำ
                             # (สั่ง 73% ได้พิกเซล 28%) เพราะลำไหม้ถูกดันลงชั้นล่างแล้วโดนลำสดบัง
                             # ขนาด cutout สองคลาสพอกัน (median 2448 vs 2440 px) จึงไม่ใช่เหตุจากขนาด
                             # สุ่มล้วนก็ยังได้เคส "ลำสดพาดทับลำไหม้" ราวครึ่งหนึ่งของการซ้อนอยู่แล้ว
SEED = 20260910


def _curate(path, cls):
    """คัดชิ้นที่ไม่สะอาดพอออกจากคลัง — คืน (ผ่านไหม, เหตุผล)

    ตรวจจาก preview grid แล้วพบว่าคลังดิบมีสามอย่างปนมา:
      - เส้นบางเล็ก (เศษขอบ ไม่ใช่ลำ)
      - ชิ้น burnt ที่มีใบเขียว/เหลืองสดติดมา  → ทำให้ label ผิดตรงพิกเซลนั้น
      - ชิ้น burnt ที่มีพื้นเหล็ก/กระบะสว่างปนมา
      - ชิ้น fresh ที่มีลำดำติดมา
    เกณฑ์ตั้งให้เข้ม ยอมเหลือชิ้นน้อยแล้วใช้ซ้ำด้วยการแปลงต่อชิ้น ดีกว่าปล่อยชิ้นสกปรกเข้าไป
    เพราะ ground truth ที่แม่นคือเหตุผลทั้งหมดของชุดนี้
    """
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None or im.shape[2] != 4:
        return False, "อ่านไม่ได้"
    a = im[..., 3] > 127
    n = int(a.sum())
    if n < 1500:
        return False, "เล็กเกิน"
    cnts, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "ไม่มีรูปทรง"
    (_, _), (w, h), _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    if min(w, h) < 6:
        return False, "เส้นบาง"
    hsv = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[..., 0][a], hsv[..., 1][a], hsv[..., 2][a]
    green = float((((hh >= 20) & (hh <= 90)) & (s >= 70) & (v >= 90)).mean())
    bright = float((v > 110).mean())
    dark = float(((v <= 70) & (s <= 90)).mean())
    if cls == "burnt":
        if green > 0.05:
            return False, "มีเขียว/เหลืองสดปน"
        if bright > 0.30:
            return False, "มีพิกเซลสว่างปน (เหล็ก/กระบะ)"
    else:
        if dark > 0.05:
            return False, "มีลำดำปน"
    return True, ""


def load_lib(verbose=True):
    lib, stats = {}, {}
    for c in ("fresh", "burnt"):
        allp = sorted((CUT / c).glob("*.png"))
        ok, drop = [], {}
        for q in allp:
            good, why = _curate(q, c)
            (ok.append(q) if good else drop.setdefault(why, []).append(q.name))
        lib[c] = ok
        stats[c] = dict(total=len(allp), kept=len(ok),
                        dropped={k: len(v) for k, v in sorted(drop.items(), key=lambda kv: -len(kv[1]))})
        if verbose:
            print(f"คลัง {c}: {len(allp)} → เหลือ {len(ok)} · ทิ้ง " +
                  ", ".join(f"{k} {len(v)}" for k, v in drop.items()))
        if not ok:
            raise SystemExit(f"คลัง {c} ว่างหลังคัด — เกณฑ์เข้มเกินไปหรือ cutout ยังไม่พอ")
    return lib, stats


def load_bgs():
    mf = json.load(open(BGS / "backgrounds_manifest.json", encoding="utf-8"))
    out = []
    for it in mf["items"]:
        p = BGS / it["file"]
        if p.exists():
            out.append((p, it["zone"], it["kind"]))
    if not out:
        raise SystemExit("ไม่มีพื้นหลัง — ต้องรัน bg_backgrounds.py ก่อน")
    return out


def jitter_stalk(rgba, rng):
    """เกรดสีระดับ "ต่อชิ้น" — ชิ้นเดิมที่ถูกหยิบซ้ำจะไม่เหมือนเดิม
    กันไม่ให้หัวโมเดลจำรูปร่าง+สีของ cutout ชิ้นใดชิ้นหนึ่งได้ (คลังมีจำกัด ชิ้นซ้ำแน่นอน)"""
    im = rgba.copy()
    f = im[..., :3].astype(np.float32)
    f *= np.array([rng.uniform(0.88, 1.12), rng.uniform(0.90, 1.10), rng.uniform(0.88, 1.12)], np.float32)
    f = np.clip((f / 255.0) ** rng.uniform(0.85, 1.18) * 255.0, 0, 255)
    im[..., :3] = f.astype(np.uint8)
    return im


def place(canvas, lab, rgba, cls, cx, cy, scale, angle, flip, zone_box=None, rng=None):
    """วาดลำหนึ่งชิ้นลงบน canvas + label ตามลำดับวาด คืนจำนวนพิกเซลที่วาดจริง

    zone_box = bool mask ขอบเขตกอง — พิกเซลนอกขอบเขตถูกตัดทิ้ง
    (ใช้ mask รูปทรงไม่สม่ำเสมอ ไม่ใช่สี่เหลี่ยม ไม่งั้นกองจะมีขอบตรงเป๊ะซึ่งเป็นร่องรอยสังเคราะห์ชัด)
    rng ไม่ None = วาดเงาอ่อนใต้ชิ้นก่อน
    """
    im = rgba
    if flip:
        im = im[:, ::-1]
    h, w = im.shape[:2]
    nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    d = int(math.hypot(nw, nh)) + 2                       # ขยายผืนก่อนหมุน กันมุมโดนตัด
    pad = np.zeros((d, d, 4), np.uint8)
    oy, ox = (d - nh) // 2, (d - nw) // 2
    pad[oy:oy + nh, ox:ox + nw] = im
    M = cv2.getRotationMatrix2D((d / 2, d / 2), angle, 1.0)
    rot = cv2.warpAffine(pad, M, (d, d), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0, 0))

    H, W = lab.shape
    x0, y0 = int(cx - d / 2), int(cy - d / 2)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x0 + d - sx0), min(H, y0 + d - sy0)
    if x1 <= x0 or y1 <= y0:
        return 0
    sub = rot[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    a = sub[..., 3] > 127                                  # alpha แข็ง: พิกเซลเป็นของลำหรือไม่เป็น
    if zone_box is not None:                               # ตัดส่วนที่ล้นออกนอกขอบเขตกอง
        a &= zone_box[y0:y1, x0:x1]                        # zone_box = bool mask ขอบเขตกอง
    if not a.any():
        return 0
    if rng is not None:                                    # เงาอ่อนใต้ชิ้น ให้ดูวางบนกองจริง ไม่ใช่แผ่นแปะ
        dx = rng.randint(*SHADOW["off"]); dy = rng.randint(*SHADOW["off"])
        sh = cv2.GaussianBlur(a.astype(np.float32), (0, 0), SHADOW["blur"])
        sh = np.roll(np.roll(sh, dy, axis=0), dx, axis=1)[..., None] * rng.uniform(*SHADOW["alpha"])
        reg = canvas[y0:y1, x0:x1].astype(np.float32)
        canvas[y0:y1, x0:x1] = np.clip(reg * (1 - sh), 0, 255).astype(np.uint8)
    canvas[y0:y1, x0:x1][a] = sub[..., :3][a]
    lab[y0:y1, x0:x1][a] = cls                             # ทับ label ด้วย = label ตรงภาพเสมอ
    return int(a.sum())


def grade(img, cond, rng, bg_is_night=False):
    """เกรดสีทั้งภาพ + ฝุ่น + เบลอ + JPEG

    bg_is_night: พื้นหลังใบนี้ถ่ายกลางคืนอยู่แล้ว (ติดไฟส้มมาตั้งแต่ต้น)
    ถ้าเป็นเช่นนั้นจะ **ไม่เกรดส้มซ้ำ** เพราะการคูณสองรอบทำให้ภาพส้มจัดเกินจริง
    และทำให้ชุดข้อมูลเอียงไปทางกลางคืนมากกว่าสัดส่วนที่ตั้งใจ
    """
    f = img.astype(np.float32)
    if bg_is_night:
        f = np.clip(f * rng.uniform(0.92, 1.08), 0, 255)   # ปรับความสว่างเล็กน้อยพอ
    elif cond == "sun":
        f *= np.array([0.98, 1.00, 1.06], np.float32)      # BGR: แดงขึ้นเล็กน้อย
        f = np.clip((f / 255.0) ** 0.92 * 255.0, 0, 255)
    elif cond == "cloud":
        f *= np.array([1.04, 1.00, 0.97], np.float32)      # ฟ้าขึ้น
        f = np.clip(((f / 255.0) ** 1.05) * 235.0, 0, 255)  # แบน คอนทราสต์ต่ำ
    else:                                                   # night: ไฟส้ม พื้นติดส้มทั้งเฟรม
        f *= np.array([0.55, 0.80, 1.25], np.float32)
        f = np.clip((f / 255.0) ** 1.35 * 255.0, 0, 255)
    if rng.random() < 0.45:                                 # ฝุ่นบาง: ชั้นน้ำตาลโปร่ง ไล่เข้ม
        H, W = f.shape[:2]
        g = cv2.GaussianBlur(rng.random((H // 16, W // 16)).astype(np.float32), (0, 0), 3)
        g = cv2.resize(g, (W, H))
        g = (g - g.min()) / max(g.max() - g.min(), 1e-6)
        al = (0.10 + 0.28 * rng.random()) * g[..., None]
        f = f * (1 - al) + np.array([150, 175, 200], np.float32) * al
    f += rng.normal(0, 2.0 + 5.0 * rng.random(), f.shape)   # noise
    out = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), 0.4 + rng.random() * 1.1)
    q = int(55 + rng.random() * 40)
    out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return out


def compose_one(i, lib, bgs, want_burnt, want_night, rng, pyrng):
    # เลือกพื้นหลังให้ตรงกับสภาพแสงที่ต้องการ (คุมสัดส่วนกลางคืนทั้งชุด)
    night_bgs = [b for b in bgs if "night" in b[0].stem]
    day_bgs = [b for b in bgs if "night" not in b[0].stem]
    pool = (night_bgs or bgs) if want_night else (day_bgs or bgs)
    bp, zone, kind = pool[pyrng.randrange(len(pool))]
    bg = cv2.imread(str(bp))
    H, W = bg.shape[:2]
    canvas = bg.copy()
    lab = np.zeros((H, W), np.uint8)                        # 0 = พื้น/ผนัง ทุกที่ที่ไม่ถูกทับ
    zx0, zx1, zy0, zy1 = zone
    zx0, zx1, zy0, zy1 = zx0 * W, zx1 * W, zy0 * H, zy1 * H
    span = min(zx1 - zx0, zy1 - zy0)
    # ขอบเขตกอง: วงรีในกรอบช่องเท บิดขอบด้วย noise เบลอ → กองมีขอบขรุขระเหมือนกองจริง
    # (ถ้าใช้กรอบสี่เหลี่ยมตรง ๆ กองจะมีขอบตรงเป๊ะ ซึ่งโมเดลจับเป็นฟีเจอร์ปลอมได้ทันที)
    zw, zh = int(zx1 - zx0), int(zy1 - zy0)
    yy, xx = np.mgrid[0:zh, 0:zw].astype(np.float32)
    ax = zw * pyrng.uniform(0.42, 0.50); ay = zh * pyrng.uniform(0.42, 0.50)
    ell = (((xx - zw / 2) / ax) ** 2 + ((yy - zh / 2) / ay) ** 2)
    nz = cv2.GaussianBlur(rng.random((max(2, zh // 24), max(2, zw // 24))).astype(np.float32), (0, 0), 1.2)
    nz = cv2.resize(nz, (zw, zh)); nz = (nz - nz.mean()) * 1.6
    region = np.zeros((H, W), bool)
    region[int(zy0):int(zy0) + zh, int(zx0):int(zx0) + zw] = (ell + nz) < 1.0
    if not region.any():
        region[int(zy0):int(zy0) + zh, int(zx0):int(zx0) + zw] = True
    zbox = region
    region_area = max(int(region.sum()), 1)

    placed = {"fresh": 0, "burnt": 0}
    used = []
    # วางไปเรื่อย ๆ จนความหนาแน่นในกรอบถึงเป้า (หรือชนเพดาน) — คลาสของแต่ละชิ้นสุ่มตามสัดส่วนที่สั่ง
    # ไม่แยกชั้น: สุ่มคลาสทีละชิ้นตามลำดับวาด จึงสลับชั้นกันเองและเกิดเคสลำสดทับลำไหม้ราวครึ่งหนึ่ง
    for _ in range(STALKS_MAX):
        if float((lab[region] > 0).mean()) >= FILL_TARGET:
            break
        cls_name = "burnt" if pyrng.random() * 100.0 < want_burnt else "fresh"
        f = lib[cls_name][pyrng.randrange(len(lib[cls_name]))]
        rgba = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            continue
        rgba = jitter_stalk(rgba, pyrng)                    # เกรดสีต่อชิ้น ไม่ใช่แค่ต่อภาพ
        s = (span * PIECE_FRAC / max(rgba.shape[:2])) * pyrng.uniform(*SCALE)
        cx = min(max(pyrng.gauss((zx0 + zx1) / 2, (zx1 - zx0) * HEAP_SIGMA), zx0), zx1)
        cy = min(max(pyrng.gauss((zy0 + zy1) / 2, (zy1 - zy0) * HEAP_SIGMA), zy0), zy1)
        if place(canvas, lab, rgba, 1 if cls_name == "fresh" else 2,
                 cx, cy, s, pyrng.uniform(0, 360), pyrng.random() < 0.5, zbox, pyrng):
            placed[cls_name] += 1
            used.append(f"{cls_name[0]}:{f.stem}")          # cutout id ที่ใช้จริงในภาพนี้

    fill = float((lab[region] > 0).mean())
    bg_is_night = "night" in bp.stem
    cond = "night" if (bg_is_night or want_night) else pyrng.choice(["sun", "sun", "cloud"])
    img = grade(canvas, cond, rng, bg_is_night)
    n1 = int((lab == 1).sum()); n2 = int((lab == 2).sum())
    rec = dict(file=f"synth_{i:04d}.png", bg=bp.name, bg_kind=kind, condition=cond,
               bg_already_night=bool(bg_is_night),
               zone_fill=round(fill, 3),
               n_stalks=placed["fresh"] + placed["burnt"],
               n_fresh=placed["fresh"], n_burnt=placed["burnt"],
               requested_burnt_pct=round(want_burnt, 1),
               burnt_frac_pixel=round(100.0 * n2 / max(n1 + n2, 1), 2),
               burnt_frac_stalks=round(100.0 * placed["burnt"] / max(placed["fresh"] + placed["burnt"], 1), 2),
               pile_px=n1 + n2,
               cutout_ids=used,
               n_unique_cutouts=len(set(used)))
    return img, lab, rec


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else N_DEFAULT
    preview = "--preview" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(exist_ok=True)
    lib, lib_stats = load_lib()
    bgs = load_bgs()
    rng = np.random.default_rng(SEED); pyrng = random.Random(SEED)

    n_mid = max(1, int(round(n * N_MID_FRAC)))
    wants = [pyrng.uniform(*MID_RANGE) for _ in range(n_mid)] + \
            [pyrng.uniform(0, 100) for _ in range(n - n_mid)]
    pyrng.shuffle(wants)

    # คุมสัดส่วนกลางคืนทั้งชุดให้ไม่เกิน NIGHT_MAX_FRAC (สุ่มว่าใบไหนเป็นกลางคืน)
    n_night = int(round(n * NIGHT_MAX_FRAC))
    nights = [True] * n_night + [False] * (n - n_night)
    pyrng.shuffle(nights)

    # พื้นหลัง inpaint ยังเห็นรอยเบลอตรงกลาง → ใช้ได้เฉพาะเมื่อกองคลุมกรอบพอที่จะปิดรอย
    inpaint_retry = 0
    rows = []
    for i, w in enumerate(wants):
        img, lab, rec = compose_one(i, lib, bgs, w, nights[i], rng, pyrng)
        if rec["bg_kind"] == "inpaint" and rec["zone_fill"] < INPAINT_MIN_FILL:
            empt = [b for b in bgs if b[2] == "empty"]      # สลับไปใช้ลานว่างจริงแทน
            if empt:
                inpaint_retry += 1
                img, lab, rec = compose_one(i, lib, empt, w, nights[i], rng, pyrng)
        cv2.imwrite(str(OUT / rec["file"]), img)
        cv2.imwrite(str(OUT / "labels" / rec["file"]), lab)
        rows.append(rec)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"{i + 1:4d}/{n}  {rec['condition']:5s} สั่ง {rec['requested_burnt_pct']:5.1f}%  "
                  f"ได้พิกเซล {rec['burnt_frac_pixel']:5.1f}%  ลำ {rec['burnt_frac_stalks']:5.1f}%  ({rec['n_stalks']} ลำ)")

    px = [r["burnt_frac_pixel"] for r in rows]
    json.dump(dict(
        note="ประกอบจาก cutout ของ g1 + พื้นหลัง g1/ภาพจริง — **ไม่ใช้ g2 เลย** (g2 คง blind)",
        limitation="ทุกชิ้นมาจาก generator g1 → สไตล์ลำยังผูกกับ g1 "
                   "สิ่งที่ใหม่คือ geometry (ตำแหน่ง/มุม/สเกล/การซ้อน) แสง พื้นหลัง และ label ที่แม่น 100%",
        classes={"0": "พื้น/ผนัง", "1": "ลำสด", "2": "ลำไหม้", "255": "ไม่มีในชุดนี้"},
        label_note="label มาจากลำดับวาด จึงตรงกับภาพเสมอ และลำไหม้ถูก label ครบทุกพิกเซลรวมไฮไลต์มันวาว",
        reuse_note="คลัง cutout มีจำกัด ชิ้นเดิมถูกหยิบซ้ำ — ลดการจำรูปร่างด้วยการสุ่ม scale/หมุน/พลิก "
                   "และเกรดสีระดับต่อชิ้น (jitter_stalk) ทุกครั้งที่วาง · cutout_ids ในแต่ละภาพบันทึกไว้ให้ตรวจย้อนได้",
        seed=SEED, n=len(rows), n_forced_mid=n_mid, mid_range=list(MID_RANGE),
        cutout_curation=lib_stats,
        density=dict(fill_target=FILL_TARGET,
                     zone_fill_mean=round(float(np.mean([r["zone_fill"] for r in rows])), 3),
                     zone_fill_min=round(float(min(r["zone_fill"] for r in rows)), 3)),
        night_frac=round(sum(1 for r in rows if r["condition"] == "night") / max(len(rows), 1), 3),
        inpaint_swapped_to_empty=inpaint_retry,
        burnt_pixel_stats=dict(min=min(px), max=max(px), mean=round(float(np.mean(px)), 2)),
        items=rows), open(OUT / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{len(rows)} ใบ → {OUT}")
    fills = [r["zone_fill"] for r in rows]
    print(f"burnt_frac_pixel: min {min(px):.1f} max {max(px):.1f} mean {np.mean(px):.1f} · "
          f"ในย่าน 40-80%: {sum(1 for v in px if 40 <= v <= 80)} ใบ")
    print(f"ความหนาแน่นกอง: mean {np.mean(fills):.2f} min {min(fills):.2f} (เป้า {FILL_TARGET}) · "
          f"กลางคืน {sum(1 for r in rows if r['condition'] == 'night')}/{len(rows)} · "
          f"สลับ inpaint→ลานว่าง {inpaint_retry} ใบ")
    if preview:
        sanity(rows)


def sanity(rows):
    """montage + histogram + overlay ตรวจว่า label ตรงภาพ"""
    sel = rows[:36]
    w, h, cols = 320, 180, 6
    sheet = np.zeros(((len(sel) + cols - 1) // cols * (h + 16), cols * w, 3), np.uint8)
    for i, r in enumerate(sel):
        im = cv2.resize(cv2.imread(str(OUT / r["file"])), (w, h))
        y, x = divmod(i, cols)
        sheet[y * (h + 16):y * (h + 16) + h, x * w:(x + 1) * w] = im
        cv2.putText(sheet, f"{r['file'][6:10]} px{r['burnt_frac_pixel']:.0f} st{r['burnt_frac_stalks']:.0f}",
                    (x * w + 3, y * (h + 16) + h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(str(OUT / "montage.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])

    px = [r["burnt_frac_pixel"] for r in rows]
    hh, _ = np.histogram(px, bins=10, range=(0, 100))
    H0, W0 = 300, 620
    hist = np.full((H0, W0, 3), 30, np.uint8)
    for b, c in enumerate(hh):
        x0 = 40 + b * 56; y1 = H0 - 40
        y0 = y1 - int(240 * c / max(hh.max(), 1))
        cv2.rectangle(hist, (x0, y0), (x0 + 48, y1), (80, 200, 255), -1)
        cv2.putText(hist, str(c), (x0 + 8, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(hist, f"{b * 10}", (x0 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(hist, "burnt_frac_pixel distribution", (40, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(OUT / "hist_burnt_frac.jpg"), hist, [cv2.IMWRITE_JPEG_QUALITY, 90])

    ov = []
    for r in rows[:10]:
        im = cv2.imread(str(OUT / r["file"]))
        lb = cv2.imread(str(OUT / "labels" / r["file"]), cv2.IMREAD_UNCHANGED)
        v = im.copy()
        v[lb == 1] = (0.45 * v[lb == 1] + np.array([0, 140, 0])).astype(np.uint8)
        v[lb == 2] = (0.45 * v[lb == 2] + np.array([0, 0, 150])).astype(np.uint8)
        ov.append(cv2.resize(np.hstack([im, v]), (760, 214)))
    cv2.imwrite(str(OUT / "label_overlay_check.jpg"), np.vstack(ov), [cv2.IMWRITE_JPEG_QUALITY, 84])
    print("sanity: montage.jpg · hist_burnt_frac.jpg · label_overlay_check.jpg (เขียว=สด แดง=ไหม้)")


if __name__ == "__main__":
    main()
