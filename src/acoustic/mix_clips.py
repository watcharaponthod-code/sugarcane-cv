# -*- coding: utf-8 -*-
"""ผสมคลิปทดสอบจากเสียงชิ้นเดี่ยว: วางเหตุการณ์ตามเวลาที่กำหนดเอง → ได้เฉลยแม่นระดับมิลลิวินาที

ทำไมต้องผสมเอง ไม่สั่งให้ AI สร้างคลิปยาวที่ฝังเสียงไว้: AI ไม่วางตามเวลาที่สั่ง เราจะไม่รู้เฉลย
และคุมอัตราส่วนเสียงเหตุการณ์ต่อเสียงพื้นหลัง (SNR) ไม่ได้ ซึ่งเป็นตัวแปรที่ต้องกวาดทั้งหมด

  python mix_clips.py                  # สร้างชุดทดสอบทั้งหมด + clips.json (เฉลย)

ชุดที่สร้าง ต่อพื้นหลัง B1..B4 × SNR -12..+12 dB:
  - clip ปกติ: เหตุการณ์กลุ่ม A 4-6 จุด เวลาแบบสุ่มด้วย seed คงที่
  - clip ควบคุม: ไม่มีเหตุการณ์เลย (วัดการแจ้งเตือนผิด)
  - clip หลอก: มีเฉพาะกลุ่ม C (ต้องไม่แจ้งเตือน)
SNR นิยามที่นี่ = 20*log10(rms ของเสียงเหตุการณ์ / rms ของพื้นหลังช่วงเดียวกัน)
"""
import json, random
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "gen_audio" / "raw"
OUT = ROOT / "gen_audio" / "clips"
SR = 44100
DUR = 30.0
SNRS = [12, 6, 0, -6, -12]
GAP = 2.0                                   # เว้นระหว่างเหตุการณ์ อย่างน้อยเท่านี้ (วินาที)


def load(p):
    x, sr = sf.read(p, always_2d=True)
    x = x.mean(1).astype(np.float32)
    if sr != SR:                             # ทุกไฟล์จาก Stable Audio เป็น 44.1k อยู่แล้ว กันเหนียว
        import math
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def trim(x, thr=0.02):
    """ตัดความเงียบหัวท้าย เพื่อให้เวลาที่เราวาง = เวลาที่ได้ยินจริง"""
    e = np.abs(x)
    idx = np.where(e > thr * e.max())[0]
    if len(idx) == 0: return x
    a = max(0, idx[0] - int(0.005 * SR)); b = min(len(x), idx[-1] + int(0.05 * SR))
    return x[a:b]


def bg_track(files, rng):
    """ต่อพื้นหลังให้ยาวพอ โดยวนไฟล์ที่มี"""
    out = []
    total = 0
    while total < int(DUR * SR):
        x = load(rng.choice(files))
        out.append(x); total += len(x)
    return np.concatenate(out)[:int(DUR * SR)]


def rms(x):
    return float(np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-12)


def real_bg(rng):
    """พื้นหลังจริงจากคลิปเทอ้อย YouTube — สุ่มช่วงในคลิป (ย่าน 4-10 kHz หายไปแล้วโดย codec: เคสแย่สุด)"""
    p = ROOT / "gen_audio" / "real" / "yt_dump_Lm5D0NSmjRA.wav"
    x = load(p)
    n = int(DUR * SR)
    i = rng.randrange(0, max(1, len(x) - n))
    return x[i:i + n]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok = {r["file"]: r["ok"] for r in json.loads((ROOT / "gen_audio" / "sfx_check.json").read_text(encoding="utf-8"))}
    groups = {}
    skipped = 0
    for p in sorted(RAW.glob("*.wav")):
        if not ok.get(p.name, True):                 # ตัดไฟล์ที่ไม่ผ่านเกณฑ์คุณภาพทิ้ง
            skipped += 1; continue
        groups.setdefault(p.stem.split("_")[0], []).append(p)
    print(f"ใช้ {sum(len(v) for v in groups.values())} ไฟล์ · ตัดที่ไม่ผ่าน {skipped} ไฟล์", flush=True)
    A = [k for k in groups if k.startswith("A")]
    D = [k for k in groups if k.startswith("D")]
    B = [k for k in groups if k.startswith("B")] + ["REAL"]     # REAL = เสียงเทอ้อยจริงจากคลิป
    C = [k for k in groups if k.startswith("C")]
    if not A or not B:
        raise SystemExit("ต้องมีทั้งกลุ่ม A และ B ใน gen_audio/raw ก่อน")

    index = []
    for b in sorted(B):
        for snr in SNRS:
            for kind in ("event", "control", "decoy", "sand"):
                rng = random.Random(f"{b}|{snr}|{kind}|20260918")
                bg = real_bg(rng) if b == "REAL" else bg_track(groups[b], rng)
                bg = bg / (np.abs(bg).max() + 1e-9) * 0.25
                mix = bg.copy()
                evs = []
                if kind != "control":
                    pool = {"decoy": C, "sand": D}.get(kind, A)
                    if kind == "sand" and not D: continue
                    t = rng.uniform(1.0, 3.0)
                    while t < DUR - 3.0:
                        key = rng.choice(sorted(pool))
                        src = trim(load(rng.choice(groups[key])))
                        i = int(t * SR); j = min(len(mix), i + len(src))
                        seg = src[:j - i]
                        # ปรับความดังให้ได้ SNR ตามต้องการ เทียบพื้นหลัง "ช่วงเดียวกัน"
                        g = rms(bg[i:j]) * (10 ** (snr / 20)) / rms(seg)
                        mix[i:j] += seg * g
                        evs.append(dict(t=round(t, 3), dur=round(len(seg) / SR, 3), cls=key,
                                        target=kind in ("event", "sand")))
                        t += rng.uniform(GAP, GAP + 4.0)
                peak = np.abs(mix).max()
                if peak > 0.99: mix = mix / peak * 0.99
                name = f"{b}_snr{snr:+d}_{kind}.wav"
                sf.write(OUT / name, mix, SR)
                index.append(dict(file=name, bg=b, snr=snr, kind=kind, dur=DUR, events=evs))
                print(f"{name}  {len(evs)} เหตุการณ์", flush=True)
    (ROOT / "gen_audio" / "clips.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ev = sum(len(c["events"]) for c in index)
    print(f"\nรวม {len(index)} คลิป · {n_ev} เหตุการณ์ · เฉลย {ROOT / 'gen_audio' / 'clips.json'}")


if __name__ == "__main__":
    main()
