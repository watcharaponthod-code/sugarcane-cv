# -*- coding: utf-8 -*-
"""Shared model loaders for fp_drift.py / fps_bench.py. Every predictor: bgr uint8 -> dust probability map (H,W) float32.
seg(tag): my torchvision models (gray->3ch at train_hw). clipseg_full / clipseg_tiles: zero-shot CIDAS baseline (prompt set B).
c6 CLIP-decoder: added when c6 delivers a loader (see c6_decoder())."""
import sys, json
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
PROMPTS_B = ["a cloud of dust in the air", "hazy dusty air", "brown dust haze"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def seg(tag, device, in_hw=None):
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large, deeplabv3_mobilenet_v3_large
    meta = json.load(open(HERE / f"{tag}_train.json"))
    dens = bool(meta.get("density")); m = (lraspp_mobilenet_v3_large if meta["model"] == "lraspp" else deeplabv3_mobilenet_v3_large)(weights=None, weights_backbone=None, num_classes=1 if dens else 2)
    sd = torch.load(HERE / f"{tag}.pt", map_location="cpu", weights_only=True)
    # Colab-trained heads wrap each 1x1 classifier conv in nn.Sequential(Dropout, Conv2d): keys read
    # "classifier.low_classifier.1.weight" instead of "classifier.low_classifier.weight". Dropout carries no parameters
    # and is a no-op at eval, so dropping that index restores the torchvision head exactly (verified: all 319 keys match).
    fix = {}
    for k, v in sd.items():
        parts = k.split(".")
        if len(parts) > 2 and parts[-3].endswith("_classifier") and parts[-2].isdigit():
            k = ".".join(parts[:-2] + parts[-1:])
        fix[k] = v
    sd = fix
    m.load_state_dict(sd); m.eval().to(device)
    hw = tuple(in_hw or meta["train_hw"]); mean, std = MEAN.to(device), STD.to(device)

    colour = bool(meta.get("color"))
    temporal = bool(meta.get("temporal")); st = {"bg": None}                       # v9: [cur, bg, diff]; call f.reset() at each clip start
    ema_k, ema_slow = 1.0 / (24.0 * float(meta.get("ema_t_s", 4.0))), float(meta.get("ema_dust_slow", 20.0))

    @torch.no_grad()
    def f(bgr):
        gs = None
        if temporal:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            gs = cv2.resize(g, (hw[1], hw[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
            if st["bg"] is None: st["bg"] = gs.copy()
            s3 = np.stack([gs, st["bg"], np.clip(128.0 + (gs - st["bg"]) * 0.5, 0, 255)], -1).astype(np.uint8)
            x = torch.from_numpy(np.ascontiguousarray(s3[..., ::-1])).to(device).float().div_(255).permute(2, 0, 1)[None].contiguous()
        elif colour:
            src = bgr if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR); g = src
            x = torch.from_numpy(np.ascontiguousarray(cv2.resize(src, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)[..., ::-1])).to(device).float().div_(255).permute(2, 0, 1)[None].contiguous()
        else:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            x = torch.from_numpy(cv2.resize(g, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)).to(device).float().div_(255)[None, None].repeat(1, 3, 1, 1)
        out = m((x - mean) / std)["out"]
        p = (torch.sigmoid(out)[0, 0] if dens else torch.softmax(out, 1)[0, 1]).cpu().numpy()
        if gs is not None:
            k = np.where(p > 0.15, ema_k / ema_slow, ema_k).astype(np.float32); st["bg"] += k * (gs - st["bg"])
        return cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    f.reset = lambda: st.update(bg=None)
    f.native_hw = hw; f.name = tag; f.density = dens; f.colour = colour; f.thr = meta.get("thr")   # per-model operating point, written by training
    return f


def _clipseg(device):
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(device)

    @torch.no_grad()
    def heat(tile):
        pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        inp = proc(text=PROMPTS_B, images=[pil] * 3, padding=True, return_tensors="pt").to(device)
        p = torch.sigmoid(model(**inp).logits).cpu().numpy()
        h, w = tile.shape[:2]
        return np.max([cv2.resize(p[i], (w, h)) for i in range(3)], axis=0)
    return heat


def clipseg_full(device):
    heat = _clipseg(device)
    f = lambda bgr: heat(bgr); f.native_hw = (352, 352); f.name = "clipseg_full"
    return f


def clipseg_tiles(device):
    heat = _clipseg(device)

    def f(bgr):
        H, W = bgr.shape[:2]; out = np.zeros((H, W), np.float32)
        for a, b in ((0, H // 2), (H // 2, H)):
            for c, d in ((0, W // 2), (W // 2, W)):
                out[a:b, c:d] = heat(bgr[a:b, c:d])
        return out
    f.native_hw = (352, 352); f.name = "clipseg_tiles2x2"
    return f


def clipdec_v3(device, tag="clipdec_v3_seed0"):
    """QA round-2 CLIP-decoder (train_clipdec_v3.py): dropout-wrapped up-blocks, density output."""
    sys.path.insert(0, str(ROOT / "gen_video" / "own_model")); sys.path.insert(0, str(HERE))
    import clipdec_model as cm; from train_clipdec_v3 import build_decoder
    meta = json.load(open(HERE / f"{tag}_train.json")); dec = build_decoder(meta["dropout"]); dec.load_state_dict(torch.load(HERE / f"{tag}.pt", map_location="cpu")); dec.eval().to(device)
    txt = cm.text_condition_vector(device); colour = bool(meta.get("color"))
    from train_clipdec_v3 import preprocess_any

    @torch.no_grad()
    def f(bgr):
        H0, W0 = bgr.shape[:2]
        p = torch.sigmoid(dec(cm.patch_tokens(preprocess_any(bgr, device, colour), device), txt))[0, 0].cpu().numpy()
        return cv2.resize(p, (W0, H0), interpolation=cv2.INTER_LINEAR)
    f.native_hw = (cm.IMG_SIZE, cm.IMG_SIZE); f.name = tag; f.density = True
    return f


def c6_decoder(device, ckpt="clipdec_ckpt.pt"):
    """c6's CLIP-decoder (frozen openai/clip-vit-base-patch16 + trained FiLM/upconv head). Frame -> gray -> squash to 224x224
    (as in train_clipdec.py; CLIPProcessor alone would centre-crop a 16:9 frame and drop bays 1/3) -> sigmoid heat -> resize back."""
    sys.path.insert(0, str(ROOT / "gen_video" / "own_model"))
    import clipdec_model as cm
    dec = cm.ClipDecoder(); dec.load_state_dict(torch.load(ROOT / "gen_video" / "own_model" / ckpt, map_location="cpu")["decoder"]); dec.eval().to(device)
    txt = cm.text_condition_vector(device)

    @torch.no_grad()
    def f(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        px = cm.preprocess_gray(cv2.resize(g, (cm.IMG_SIZE, cm.IMG_SIZE), interpolation=cv2.INTER_AREA), device)
        p = torch.sigmoid(dec(cm.patch_tokens(px, device), txt))[0, 0].cpu().numpy()
        return cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    f.native_hw = (cm.IMG_SIZE, cm.IMG_SIZE); f.name = "c6_clipdec"
    return f


def clipseg_v2(device, variant="sliding", use_neg=True):
    """c6's clipseg_v2 config (sliding 512/768 50% overlap tent-blend, 6 pos + 5 neg prompts, NEGW 0.5, THR 0.4) run on `device`:
    reuses c6's raw_maps/combine unchanged, only seg_batch_fast is re-implemented with device placement (c6's runs CPU-only)."""
    sys.path.insert(0, str(ROOT / "gen_video" / "clipseg_v2"))
    import clipseg_v2 as cv2mod
    proc, model = cv2mod.get_model(); model.to(device)

    def seg_batch_fast(pil, prompts):
        img_inputs = proc(images=[pil], return_tensors="pt").to(device)
        txt_inputs = proc(text=prompts, padding=True, return_tensors="pt").to(device)
        n = len(prompts)
        with torch.no_grad():
            vo = model.clip.get_image_features(pixel_values=img_inputs["pixel_values"], output_hidden_states=True)
            acts = [vo.hidden_states[i + 1].expand(n, -1, -1) for i in model.extract_layers]
            cond = model.get_conditional_embeddings(batch_size=n, input_ids=txt_inputs["input_ids"], attention_mask=txt_inputs.get("attention_mask"))
            return torch.sigmoid(model.decoder(acts, cond).logits).cpu().numpy()
    cv2mod.seg_batch_fast = seg_batch_fast          # raw_maps -> _window_pos_neg -> module-global lookup

    def f(bgr):
        sp, sn, fp, fn_ = cv2mod.raw_maps(bgr)
        return cv2mod.combine(sp, sn, fp, fn_, variant, use_neg).astype(np.float32)
    f.native_hw = (352, 352); f.name = f"clipseg_v2_{variant}{'_neg' if use_neg else ''}"; f.thr = cv2mod.THR; f.video_roi_mask = cv2mod.video_roi_mask
    return f


def seg_veto(tag, device, veto_spec="on", fps=24.0, in_hw=None):
    """seg(tag) + veto.Veto on top: same call signature (bgr -> alpha), same reset() at each clip start.
    The wrapped alpha is what eval_seg.py thresholds, so the round-3 tables measure v3.1+veto end to end."""
    import veto as _veto
    base = seg(tag, device, in_hw=in_hw)
    cfg = _veto.parse(veto_spec)
    if cfg is None: return base
    V = _veto.Veto(fps=fps, device=device, **cfg)

    def f(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        a = base(bgr)
        out, info = V(g.astype(np.float32), a)
        f.last_info = info
        return out
    f.reset = lambda: (base.reset(), V.reset())
    f.native_hw = base.native_hw; f.name = tag + "+veto"; f.density = base.density; f.colour = base.colour; f.thr = base.thr
    f.veto = V; f.last_info = None
    return f
