# -*- coding: utf-8 -*-
"""Kaggle: segmentation อ้อยที่กำลังไหล + ใบแห้ง ที่ช่องเทแท่นยกข้าง (มุมข้าง) — แทนการตีกรอบ

คลาส: 0 พื้นหลัง · 1 ลำอ้อยที่ไหล (stalk) · 2 ใบแห้ง/ฟาง (leaf)  · mask จาก SAM2 + จุดที่ Claude วาง (ฉบับร่าง ไม่ใช่คน)
โมเดล: LR-ASPP MobileNetV3-Large (ตัวเดียวกับโมเดลฝุ่น → เร็วพอ 30 FPS บน GTX 1060)
แบ่ง: test = เฟรม t > 100 s (รถคันหลัง) · val = เฟรม train ช่วง 88–100 s (ใช้เลือก epoch) · train = ที่เหลือ
เกณฑ์ผ่าน (ตั้งก่อนเห็นผล 2026-09-17): test IoU stalk >= 0.60 และ leaf >= 0.50 · FPS วัดบนเครื่องแยก (>= 30 ที่ 720p ทั้ง pipeline)
อินพุต DATA_URL = zip {frames/*.jpg, masks/*.png, index.json} · ออก /kaggle/working/leafflow_seg.pt + leafflow_eval.json + test_vis/*.jpg
ห้ามเรียก MLflow บน Kaggle (kernel ถูก CANCEL)
"""
import io, json, os, random, urllib.request, zipfile
from pathlib import Path

import cv2, numpy as np, torch, torch.nn.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, deeplabv3_resnet50
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights

W = Path("/kaggle/working"); T = Path("/tmp/flow"); NC = 3
H_IN, W_IN = int(os.environ.get("SEG_H", 360)), int(os.environ.get("SEG_W", 640))
EPOCHS = int(os.environ.get("SEG_EPOCHS", 120))
ARCH = os.environ.get("SEG_ARCH", "lraspp")   # lraspp | dlv3r50
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(0); np.random.seed(0); torch.manual_seed(0)

if not T.exists():
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(os.environ["DATA_URL"]).read())).extractall(T)
idx = json.loads((T / "index.json").read_text())
idx = [x for x in idx if (T / "masks" / (x["id"] + ".png")).exists()]
test = [x for x in idx if x["split"] == "test"]
val = [x for x in idx if x["split"] == "train" and 88 <= x["t"] <= 100]
train = [x for x in idx if x["split"] == "train" and not (88 <= x["t"] <= 100)]
print(f"train {len(train)} · val {len(val)} · test {len(test)} · {dev}", flush=True)


def load(x):
    im = cv2.imread(str(T / "frames" / (x["id"] + ".jpg")))[..., ::-1]
    m = cv2.imread(str(T / "masks" / (x["id"] + ".png")), cv2.IMREAD_GRAYSCALE)
    return im, m


def to_t(im):
    im = cv2.resize(im, (W_IN, H_IN), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(((im - MEAN) / STD).transpose(2, 0, 1))


def augment(im, m):
    if random.random() < 0.5:
        im, m = im[:, ::-1], m[:, ::-1]
    s = random.uniform(0.8, 1.25)                       # สุ่มย่อ/ขยาย + ครอป (กล้องมือถือแพน/ซูม)
    h, w = m.shape
    im = cv2.resize(np.ascontiguousarray(im), (int(w * s), int(h * s))); m = cv2.resize(np.ascontiguousarray(m), (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST)
    if s > 1:
        y = random.randint(0, im.shape[0] - h); x0 = random.randint(0, im.shape[1] - w)
        im, m = im[y:y + h, x0:x0 + w], m[y:y + h, x0:x0 + w]
    else:
        im = cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, w - im.shape[1], cv2.BORDER_REFLECT)
        m = cv2.copyMakeBorder(m, 0, h - m.shape[0], 0, w - m.shape[1], cv2.BORDER_CONSTANT, value=0)
    hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] *= random.uniform(0.6, 1.4); hsv[..., 1] *= random.uniform(0.7, 1.3)
    im = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    if random.random() < 0.3:                           # หมอกฝุ่นเทียมทับ (ฝุ่นไม่ใช่คลาส)
        a = random.uniform(0.1, 0.4); im = (im * (1 - a) + 200 * a).astype(np.uint8)
    return im, m


def batch(items, aug):
    xs, ys = [], []
    for x in items:
        im, m = load(x)
        if aug: im, m = augment(im, m)
        xs.append(to_t(im)); ys.append(torch.from_numpy(cv2.resize(np.ascontiguousarray(m), (W_IN, H_IN), interpolation=cv2.INTER_NEAREST).astype(np.int64)))
    return torch.stack(xs).to(dev), torch.stack(ys).to(dev)


def ious(model, items):
    inter = np.zeros(NC); union = np.zeros(NC); tp = np.zeros(NC); fp = np.zeros(NC); fn = np.zeros(NC)
    model.eval()
    with torch.no_grad():
        for x in items:
            im, m = load(x)
            p = model(to_t(im)[None].to(dev))["out"].argmax(1)[0].cpu().numpy()
            p = cv2.resize(p.astype(np.uint8), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
            for c in range(NC):
                a, b = p == c, m == c
                inter[c] += (a & b).sum(); union[c] += (a | b).sum(); tp[c] += (a & b).sum(); fp[c] += (a & ~b).sum(); fn[c] += (~a & b).sum()
    model.train()
    names = ["bg", "stalk", "leaf"]
    return {names[c]: dict(iou=float(inter[c] / max(1, union[c])), precision=float(tp[c] / max(1, tp[c] + fp[c])),
                           recall=float(tp[c] / max(1, tp[c] + fn[c]))) for c in range(NC)}


model = (deeplabv3_resnet50(weights_backbone=ResNet50_Weights.IMAGENET1K_V2, num_classes=NC, aux_loss=False) if ARCH == "dlv3r50"
         else lraspp_mobilenet_v3_large(weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1, num_classes=NC)).to(dev)
print("arch", ARCH, "params", sum(q.numel() for q in model.parameters()), flush=True)
cnt = np.zeros(NC)
for x in train:
    cnt += np.bincount(load(x)[1].ravel(), minlength=NC)[:NC]
cw = torch.tensor((cnt.sum() / (NC * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32, device=dev)
print("class px", cnt.astype(int).tolist(), "weights", [round(float(v), 2) for v in cw], flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
best, best_score = None, -1
for ep in range(EPOCHS):
    random.shuffle(train)
    for i in range(0, len(train), 4):
        xb, yb = batch(train[i:i + 4], True)
        loss = F.cross_entropy(model(xb)["out"], yb, weight=cw)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    if (ep + 1) % 5 == 0:
        v = ious(model, val) if val else ious(model, train[:6])
        score = (v["stalk"]["iou"] + v["leaf"]["iou"]) / 2
        print(f"ep {ep+1} loss {loss.item():.3f} val stalk {v['stalk']['iou']:.3f} leaf {v['leaf']['iou']:.3f}", flush=True)
        if score > best_score:
            best_score = score; best = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

model.load_state_dict(best)
res = ious(model, test)
passed = res["stalk"]["iou"] >= 0.60 and res["leaf"]["iou"] >= 0.50
torch.save(dict(state=best, nc=NC, size=(H_IN, W_IN), classes=["bg", "stalk", "leaf"], arch=ARCH), W / "leafflow_seg.pt")
out = dict(arch=ARCH, test=res, val_best=best_score, passed_iou=passed, n_train=len(train), n_val=len(val), n_test=len(test), size=[H_IN, W_IN], epochs=EPOCHS)
(W / "leafflow_eval.json").write_text(json.dumps(out, indent=1))
(W / "test_vis").mkdir(exist_ok=True)
model.eval()
with torch.no_grad():
    for x in test:
        im, m = load(x)
        p = model(to_t(im)[None].to(dev))["out"].argmax(1)[0].cpu().numpy().astype(np.uint8)
        p = cv2.resize(p, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        def paint(mask):
            v = im[..., ::-1].copy(); col = np.zeros_like(v); col[mask == 1] = (0, 200, 0); col[mask == 2] = (0, 140, 255)
            return np.where(mask[..., None] > 0, (0.5 * v + 0.5 * col).astype(np.uint8), v)
        cv2.imwrite(str(W / "test_vis" / (x["id"] + ".jpg")), np.hstack([paint(m), paint(p)]))
print(json.dumps(out), flush=True)
print("PASS(IoU)" if passed else "NOT PASS(IoU)", flush=True)
