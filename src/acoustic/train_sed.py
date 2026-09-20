# -*- coding: utf-8 -*-
"""PoC: โมเดล AI จำแนกเสียง (CNN บน log-mel) แทนกฎ threshold — ตอบคำถามว่า "โมเดลจับเสียงแบบนี้ได้ไหม"

วัดแบบที่ตรงกับคำถามจริง: เทรนบนพื้นหลังสังเคราะห์ (B2/B3) เท่านั้น แล้ว "ทดสอบกับพื้นหลังเสียงเทอ้อยจริง"
ที่โมเดลไม่เคยเห็น (REAL) — ถ้าตกฮวบ แปลว่าเสียงสังเคราะห์สอนอะไรที่ใช้กับของจริงไม่ได้ (บทเรียนเดียวกับภาพอ้อยไหม้)

3 คลาสต่อหน้าต่าง 0.5 วินาที: 0 = ไม่มีอะไร · 1 = ของแข็งกระแทก (กลุ่ม A) · 2 = ทรายไหล (กลุ่ม D)
เสียงหลอกกลุ่ม C อยู่ในคลิป decoy และถูกกำกับเป็นคลาส 0 → โมเดลต้องเรียนว่า "อ้อยกระแทก/โซ่/ค้อน ไม่ใช่เป้าหมาย"

  python train_sed.py            # เทรน + วัดผล → gen_audio/sed_result.json
"""
import argparse, json, time
from pathlib import Path
import math
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
GEN = ROOT / "gen_audio"
SR = 44100
WIN, HOP = 0.25, 0.125               # หน้าต่าง 0.25 วิ (เสียงน็อตสั้นกว่า 100 ms — 0.5 วิ เจือจางเกินไป)
HF_CUT = None                        # ถ้าตั้งค่า = ตัด mel bin ที่สูงกว่านี้ (Hz) ทั้ง train/test
N_FFT, N_MELS, MEL_HOP = 1024, 64, 256
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_BG = ("B2", "B3")              # เทรนเฉพาะพื้นหลังสังเคราะห์
TEST_BG = ("REAL",)                  # ทดสอบกับเสียงจริงที่ไม่เคยเห็น
torch.manual_seed(0); np.random.seed(0)

_mel_fb = None
def logmel(x):
    global _mel_fb
    X = torch.stft(torch.from_numpy(x).float(), N_FFT, MEL_HOP, window=torch.hann_window(N_FFT), return_complex=True)
    P = (X.abs() ** 2)
    if _mel_fb is None:
        f = torch.linspace(0, SR / 2, P.shape[0])
        m = 2595 * torch.log10(1 + f / 700)
        edges = torch.linspace(float(m.min()), float(m.max()), N_MELS + 2)
        fb = torch.zeros(N_MELS, P.shape[0])
        for i in range(N_MELS):
            lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
            fb[i] = torch.clamp(torch.minimum((m - lo) / (c - lo + 1e-9), (hi - m) / (hi - c + 1e-9)), min=0)
        _mel_fb = fb
    M = torch.log(_mel_fb @ P + 1e-8)
    if HF_CUT:                                    # จำลองไมค์/โคเดคจริงที่ย่านสูงหายไป -> โมเดลห้ามพึ่งย่านที่ของจริงไม่มี
        keep = int(N_MELS * (2595 * math.log10(1 + HF_CUT / 700)) / (2595 * math.log10(1 + (SR / 2) / 700)))
        M[keep:] = M[:keep].mean()
    return M


def windows(clip):
    """คืน (X, y) ของคลิปหนึ่ง: y=1 ถ้ามีเหตุการณ์กลุ่ม A ทับหน้าต่าง, 2 ถ้ากลุ่ม D, 0 ถ้าไม่มี"""
    x, sr = sf.read(GEN / "clips" / clip["file"], always_2d=True)
    x = x.mean(1).astype(np.float32)
    M = logmel(x).numpy()
    fps = SR / MEL_HOP
    w, h = int(WIN * fps), int(HOP * fps)
    ev = [(e["t"], e["t"] + e["dur"], 1 if e["cls"].startswith("A") else 2) for e in clip["events"] if e["target"]]
    X, y = [], []
    for i in range(0, M.shape[1] - w, h):
        t0, t1 = i / fps, (i + w) / fps
        lab = 0
        for a, b, c in ev:
            if c == 1 and a >= t0 and a < t1: lab = 1; break        # กระแทก: นับที่จังหวะเริ่ม
            if c == 2 and min(b, t1) - max(a, t0) > 0.5 * WIN: lab = 2   # ทราย: ต้องคลุมครึ่งหน้าต่าง
        X.append(M[:, i:i + w]); y.append(lab)
    return np.stack(X), np.array(y)


class Net(nn.Module):
    def __init__(s, nc=3):
        super().__init__()
        c = [1, 16, 32, 64]
        s.b = nn.ModuleList([nn.Sequential(nn.Conv2d(c[i], c[i + 1], 3, padding=1), nn.BatchNorm2d(c[i + 1]),
                                           nn.ReLU(), nn.MaxPool2d(2)) for i in range(3)])
        s.fc = nn.Linear(c[-1], nc)
    def forward(s, x):
        for b in s.b: x = b(x)
        return s.fc(x.mean((2, 3)))


def main():
    global HF_CUT, WIN, HOP
    ap = argparse.ArgumentParser()
    ap.add_argument("--lowpass", type=float, default=0, help="ตัดย่านสูงกว่านี้ (Hz) เช่น 4000")
    ap.add_argument("--win", type=float, default=WIN)
    a = ap.parse_args()
    HF_CUT = a.lowpass or None
    WIN, HOP = a.win, a.win / 2
    print(f"หน้าต่าง {WIN}s · ตัดย่านสูงกว่า {HF_CUT or 'ไม่ตัด'}", flush=True)
    clips = json.loads((GEN / "clips.json").read_text(encoding="utf-8"))
    data = {}
    t0 = time.perf_counter()
    for c in clips:
        if c["bg"] not in TRAIN_BG + TEST_BG: continue
        data[c["file"]] = (windows(c), c)
    print(f"เตรียมข้อมูล {len(data)} คลิป {time.perf_counter() - t0:.0f} วิ", flush=True)

    def pack(bgs):
        X = np.concatenate([v[0][0] for v in data.values() if v[1]["bg"] in bgs])
        y = np.concatenate([v[0][1] for v in data.values() if v[1]["bg"] in bgs])
        return torch.from_numpy(X)[:, None].float(), torch.from_numpy(y).long()
    Xtr, ytr = pack(TRAIN_BG); Xte, yte = pack(TEST_BG)
    mu, sd = Xtr.mean(), Xtr.std()
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    print(f"train {tuple(Xtr.shape)} · คลาส {np.bincount(ytr.numpy())} | test {tuple(Xte.shape)} · คลาส {np.bincount(yte.numpy())}", flush=True)

    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr.numpy(), minlength=3) + 1e-9), dtype=torch.float32, device=DEV)
    for ep in range(12):
        net.train(); perm = torch.randperm(len(Xtr)); tot = 0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb, yb = Xtr[idx].to(DEV), ytr[idx].to(DEV)
            loss = F.cross_entropy(net(xb), yb, weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        print(f"  ep{ep + 1} loss {tot / len(perm):.3f}", flush=True)

    @torch.no_grad()
    def pred(X):
        net.eval()
        return torch.cat([net(X[i:i + 256].to(DEV)).softmax(1).cpu() for i in range(0, len(X), 256)])

    out = {}
    for name, (X, y) in (("train(สังเคราะห์)", (Xtr, ytr)), ("test(เสียงจริง)", (Xte, yte))):
        p = pred(X).argmax(1).numpy(); t = y.numpy()
        r = {}
        for c, lab in ((1, "กระแทก"), (2, "ทราย")):
            tp = int(((p == c) & (t == c)).sum()); fp = int(((p == c) & (t != c)).sum()); fn = int(((p != c) & (t == c)).sum())
            r[lab] = dict(recall=round(tp / max(tp + fn, 1), 3), precision=round(tp / max(tp + fp, 1), 3), n=tp + fn)
        r["แจ้งผิดต่อนาที"] = round(float(((p != 0) & (t == 0)).sum()) * (60 / (len(t) * HOP)), 1)
        out[name] = r
        print(name, json.dumps(r, ensure_ascii=False), flush=True)
    (GEN / f"sed_result{('_lp%d' % a.lowpass) if a.lowpass else ''}_w{int(WIN*1000)}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save({"state": net.state_dict(), "mu": float(mu), "sd": float(sd), "win": WIN, "lowpass": a.lowpass}, GEN / "sed_cnn.pt")


if __name__ == "__main__":
    main()
