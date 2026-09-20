# -*- coding: utf-8 -*-
"""รัน leafflow_seg.pt บนวิดีโอ → คลิปผลซ้อน mask (เขียว = ลำอ้อยที่ไหล, ส้ม = ใบแห้ง) + วัด FPS ทั้ง pipeline

  python leafflow_video.py --model leafflow_seg.pt --src yt_Lm5D0NSmjRA.mp4 --out leafflow_overlay.mp4 [--start 0 --dur 0]

FPS ที่รายงาน = decode + preprocess + model + upscale mask + วาดซ้อน (ไม่รวมการเขียนไฟล์) ต่อเฟรม
ค่าที่มุมจอ: % พื้นที่ใบแห้งเทียบกับพื้นที่อ้อยทั้งหมด (ใบ / (ลำ+ใบ)) เฉลี่ยเคลื่อนที่ 1 วินาที
"""
import argparse, time
from collections import deque

import cv2, numpy as np, torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, deeplabv3_resnet50

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--src", required=True); ap.add_argument("--out", default="")
ap.add_argument("--start", type=float, default=0); ap.add_argument("--dur", type=float, default=0)
ap.add_argument("--half", action="store_true")
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(a.model, map_location="cpu")
H, W = ck["size"]
net = (deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=ck["nc"], aux_loss=False) if ck.get("arch") == "dlv3r50"
       else lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=ck["nc"]))
net.load_state_dict(ck["state"]); net = net.to(dev).eval()
if a.half and dev.type == "cuda":
    net = net.half()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
lut = torch.tensor([[0, 0, 0], [0, 200, 0], [0, 140, 255]], dtype=torch.uint8, device=dev)   # BGR

cap = cv2.VideoCapture(a.src); fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
ok, frame = cap.read()
vh, vw = frame.shape[:2]
wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (vw, vh)) if a.out else None
hist = deque(maxlen=int(fps_src)); times = []
n_max = int(a.dur * fps_src) if a.dur else 10 ** 9
n = 0
with torch.no_grad():
    while ok and n < n_max:
        t0 = time.perf_counter()
        f = torch.from_numpy(frame).to(dev, non_blocking=True)
        x = f[None].permute(0, 3, 1, 2)[:, [2, 1, 0]].float() / 255.0
        x = (torch.nn.functional.interpolate(x, (H, W), mode="bilinear", align_corners=False) - mean) / std
        out = net(x.half() if a.half and dev.type == "cuda" else x)["out"]
        p = torch.nn.functional.interpolate(out.float(), (vh, vw), mode="bilinear", align_corners=False).argmax(1)[0]
        col = lut[p]
        blend = torch.where((p > 0)[..., None], (0.55 * f + 0.45 * col).to(torch.uint8), f)
        stalk = int((p == 1).sum()); leaf = int((p == 2).sum())
        vis = blend.cpu().numpy()
        if dev.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        hist.append(leaf / max(1, stalk + leaf))
        fps_now = 1.0 / np.mean(times[-30:])
        cv2.rectangle(vis, (0, 0), (430, 70), (0, 0, 0), -1)
        cv2.putText(vis, f"leaf {100*np.mean(hist):4.1f}% of cane area", (10, 28), 0, 0.8, (0, 200, 255), 2)
        cv2.putText(vis, f"{fps_now:5.1f} fps (pipeline)", (10, 60), 0, 0.7, (255, 255, 255), 2)
        if wr: wr.write(vis)
        n += 1
        ok, frame = cap.read()
if wr: wr.release()
t = np.array(times[10:]) if len(times) > 20 else np.array(times)
print(f"frames {n} · pipeline {1/t.mean():.1f} fps (p95 frame {1000*np.percentile(t, 95):.1f} ms) · model input {W}x{H} · video {vw}x{vh} · half={a.half}")
