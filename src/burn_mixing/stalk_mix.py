# -*- coding: utf-8 -*-
"""วัด "สัดส่วนลำไหม้บนผิวกอง" ต่อภาพ (CV ล้วน ไม่เทรน) แล้วเทียบกับ % ที่สั่ง generator ในชื่อไฟล์ mix{N}_*.png
วิธี: ช่องกลางระหว่างราวเหลืองคู่ใน → กอง = พิกเซลที่มี texture และไม่ใช่พื้นเหล็กเทาเรียบ → ในกอง แยก
    ลำสว่าง/สด  = มีสี (S ≥ 45) และ V > 90  (พื้นเหล็กเทาสว่างแต่ S ต่ำ จึงไม่ติด)
    ลำดำ/ไหม้   = V < dark_v  และ S < dark_s (ดำ/เทาเข้ม ทั้งมันเงาและด้าน)
    อื่น (เงา ฝุ่น กลาง ๆ) ไม่นับ
    burnt_frac = ดำ / (ดำ + สว่าง)    ← สัดส่วนลำที่ตัดสินได้ ไม่ใช่สัดส่วนพิกเซลทั้งกอง
รัน: python stalk_mix.py [โฟลเดอร์ภาพ]  → ตาราง + stalk_mix.json + stalk_mix.png (plot สั่ง vs วัด) + ภาพ annotate ใน out/
ค่าเกณฑ์เป็น placeholder จากภาพ AI ใบแรก ต้องตั้งใหม่บนภาพจริง"""
import sys, re, json
from pathlib import Path
import numpy as np, cv2

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent.parent / "gen_images" / "burnt_mixed"
OUT = HERE / "stalk_mix_out"; OUT.mkdir(exist_ok=True)
BRIGHT_V, BRIGHT_S, DARK_V, DARK_S, TEX_MIN = 90, 45, 60, 70, 20   # grid-search บนชุด 30 ใบ 2026-09-10: corr .82 MAE 12.7
BAY = (0.42, 0.60, 0.30, 0.88)          # x0,x1,y0,y1 สัดส่วนของภาพ = ช่องกลาง (มุมบนเดิม) ปรับถ้า generator เปลี่ยนกรอบ


def measure(path):
    f = cv2.imread(str(path)); H, W = f.shape[:2]
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV); Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F, 3)), -1, (9, 9))
    bay = np.zeros((H, W), bool); bay[int(BAY[2] * H):int(BAY[3] * H), int(BAY[0] * W):int(BAY[1] * W)] = True
    cane = ((S >= BRIGHT_S) & (V > BRIGHT_V)) | ((V < DARK_V) & (S < DARK_S) & (tex > TEX_MIN))      # ลำสด (มีสี) หรือ ลำดำ — พื้นเหล็กเทา S ต่ำ V กลาง ไม่เข้าเงื่อนไข
    pile = bay & cane
    pile = cv2.morphologyEx(pile.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    pile = cv2.morphologyEx(pile, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool)
    bright = pile & (S >= BRIGHT_S) & (V > BRIGHT_V)
    dark = pile & (V < DARK_V) & (S < DARK_S) & (tex > TEX_MIN) & ~bright   # tex gate ตัดเงาพื้นเรียบ ลำดำมีลายเส้น
    nb, nd = int(bright.sum()), int(dark.sum())
    frac = 100.0 * nd / max(nb + nd, 1)
    m = re.search(r"mix(\d+)", path.stem); ordered = int(m.group(1)) if m else None
    v = f.copy(); v[dark] = (0.4 * v[dark] + [0, 0, 150]).astype(np.uint8); v[bright] = (0.5 * v[bright] + [0, 120, 120]).astype(np.uint8)
    cnts, _ = cv2.findContours(pile.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v, cnts, -1, (0, 255, 255), 2)
    cv2.rectangle(v, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(v, f"{path.name}  ordered {ordered}%  measured burnt {frac:.0f}%  (dark {nd} / bright {nb} / pile {int(pile.sum())})", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / (path.stem + "_annot.jpg")), cv2.resize(v, (W // 2, H // 2)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return dict(file=path.name, ordered_pct=ordered, measured_burnt_pct=round(frac, 1), dark_px=nd, bright_px=nb, pile_px=int(pile.sum()),
                undecided_pct=round(100.0 * (pile.sum() - nb - nd) / max(pile.sum(), 1), 1))


rows = [measure(p) for p in sorted(SRC.glob("*.png")) + sorted(SRC.glob("*.jpg"))]
for r in rows: print(f"{r['file']:40s} ordered {str(r['ordered_pct']):>4}%  measured {r['measured_burnt_pct']:5.1f}%  undecided {r['undecided_pct']:4.1f}%")
json.dump(rows, open(HERE / "stalk_mix.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pts = [(r["ordered_pct"], r["measured_burnt_pct"]) for r in rows if r["ordered_pct"] is not None]
if len(pts) >= 3:
    x, y = np.array(pts, float).T; a, b = np.polyfit(x, y, 1); rho = float(np.corrcoef(x, y)[0, 1])
    print(f"\n{len(pts)} ภาพ: measured ≈ {a:.2f}·ordered + {b:.1f}   corr {rho:.3f}   MAE {np.mean(np.abs(y - x)):.1f} จุด")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(4.5, 4.5)); plt.scatter(x, y); plt.plot([0, 100], [0, 100], "k--", lw=1); plt.plot([0, 100], [b, 100 * a + b], "r-", lw=1)
        plt.xlabel("ordered burnt %"); plt.ylabel("measured burnt % (surface)"); plt.title(f"n={len(pts)} corr={rho:.2f}"); plt.tight_layout(); plt.savefig(HERE / "stalk_mix.png", dpi=110)
    except Exception as e: print("plot skipped:", e)
