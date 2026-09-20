# -*- coding: utf-8 -*-
"""Post-hoc veto for the frozen v3.1 density head: the weights stay untouched, the alpha map gets pixels ZEROED where
cheap frame-vs-background cues say "this is not haze". Chosen by measurement (veto_probe.py -> veto_search.py), not by
theory; the thresholds below are the defaults that won the cross-camera search and every one of them is a CLI knob.

Per frame, on the gray image and an EMA background of the recent past (alpha-agnostic - see WHY below):
    diff  = |cur - bg|                 a pixel that has not changed against the background is a parked truck / pile /
                                       floor, never dust: keep only diff >= diff_min
    Ecur  = box(|Lap(cur)|, k)         texture energy of the current frame around the pixel: haze is smooth, a truck
                                       bed, cane streaks and a charred pile are not: keep only Ecur <= e_max
    R     = (Ecur+1) / (Ebg+1)         texture ratio to the background: R >> 1 = new edges arrived (object), R ~ 0 =
                                       the background texture was wiped out by an opaque cover: keep r_min <= R <= r_max
    lens  = median R over the whole frame where the background has texture: when EVERYTHING lost its texture at once
            (median R < blur_r) while the intensities barely moved (median |diff| <= blur_d, default 8 gray levels; a frame
            FULL of dust also collapses texture but shifts intensity by tens of levels) the lens defocused / fogged / got
            dirty - v3.1 paints 47-55 % of a sigma>=1 px blurred clean frame as dust (qa/bench_results.md T7). Such a frame
            is vetoed whole and flagged `lens_blur`.

Reference = plain EMA, default 60 s: on a 10 s clip that is effectively the first frame (the probe's `frozen` variant, the
best of ema 2/4/8/16/frozen on the install camera); on a live stream it follows lighting drift within a minute and absorbs
a parked truck or a dumped pile in the same time. That is the runtime stand-in for the per-truck latched reference of
technology_and_method_decision.md until the trigger exists.

WHY the background must NOT slow down under alpha (unlike the v9/v10 temporal input): with a 20x slower update where
the model fires, anything the model mistakes for dust never enters the background, so `diff` stays high forever on a
static pile - the veto then agrees with the model's own mistake. The probe measured it: burnt7's black pile kept
diff ~20 for the whole clip with slow=20. With a plain EMA the pile fades into the background in ~ema_t seconds and
diff falls to 0. The price: dust that lingers unchanged for >> ema_t seconds fades too - the alarm is raised at the
onset, which is what the gate measures; a lingering plume that has stopped changing is the trigger/decision-window's
job (technology_and_method_decision.md), not this filter's.

Usage:  v = Veto(fps=24, **kw); v.reset() at each clip start; alpha2, info = v(gray_float32, alpha)
        gray / alpha / masks may be numpy arrays or torch tensors; all cues run in torch on `device` (default: cuda if
        available). Ported from cv2/numpy 2026-09-10 for speed - same rule, same knobs, same defaults: Laplacian ksize 3 =
        [[2,0,2],[0,-8,0],[2,0,2]] (Sobel-derived, NOT the 4-neighbour ksize-1 stencil) with reflect-101 border, boxFilter = avg_pool2d after reflect pad, np.median = mean of
        the two middle values on an even count. Checked event-for-event against the numpy version on clips 1/2/3/4/6/7."""
import numpy as np, cv2, torch
import torch.nn.functional as F

DEFAULTS = dict(ema_t=60.0, k=11, diff_min=3.0, e_max=12.0, r_min=0.4, r_max=3.0, blur_r=0.55, blur_d=8.0, f_ref=0.3, thr=0.29, r_norm=1.0, f_min=0.0, scale=1.0, sign=1.0)


def parse(spec):
    """'diff_min=5,e_max=12' -> dict; '' / None / 'on' -> defaults; 'off' -> None"""
    if spec is None or str(spec).lower() == "off": return None
    d = dict(DEFAULTS)
    if str(spec).lower() in ("", "on", "default"): return d
    for kv in str(spec).split(","):
        k, v = kv.split("="); d[k.strip()] = float(v)
    return d


class Veto:
    def __init__(self, fps=24.0, ema_t=60.0, k=11, diff_min=3.0, e_max=12.0, r_min=0.4, r_max=3.0, blur_r=0.55, blur_d=8.0, f_ref=0.3, thr=0.29, r_norm=1.0, f_min=0.0, scale=1.0, sign=1.0, device=None):
        self.sign = int(sign); self.r_norm = bool(r_norm); self.f_min = float(f_min); self.scale = float(scale); self._n = 0
        self.fps, self.ema_t, self.k, self.f_ref, self.thr = float(fps), float(ema_t), int(k), float(f_ref), float(thr)
        self.diff_min, self.e_max, self.r_min, self.r_max, self.blur_r, self.blur_d = diff_min, e_max, r_min, r_max, blur_r, float(blur_d)
        self.dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lap = torch.tensor([[2.0, 0.0, 2.0], [0.0, -8.0, 0.0], [2.0, 0.0, 2.0]], device=self.dev).view(1, 1, 3, 3)   # cv2.Laplacian ksize=3 (Sobel-derived); the 4-neighbour [0 1 0;1 -4 1;0 1 0] is ksize=1
        self.reset()

    def reset(self):
        self.bg = None; self.Ebg = None; self.keep = None; self.hot = None; self.lens_blur = False

    def config(self):
        return dict(ema_t=self.ema_t, k=self.k, diff_min=self.diff_min, e_max=self.e_max, r_min=self.r_min, r_max=self.r_max, blur_r=self.blur_r, blur_d=self.blur_d, f_ref=self.f_ref, thr=self.thr, r_norm=float(self.r_norm), f_min=self.f_min, scale=self.scale, sign=self.sign)

    def _t(self, x):
        """numpy / tensor -> tensor on self.dev; bool stays bool, anything else becomes float32."""
        t = x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))
        if t.dtype != torch.bool: t = t.to(torch.float32)
        return t.to(self.dev)

    def tex(self, g):
        """box_k(|Lap3(g)|) = cv2.boxFilter(np.abs(cv2.Laplacian(g, CV_32F, ksize=3)), (k, k)) in torch; both borders reflect-101."""
        x = F.conv2d(F.pad(g[None, None], (1, 1, 1, 1), mode="reflect"), self._lap).abs_()
        p = self.k // 2
        return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="reflect"), self.k, stride=1)[0, 0]

    @staticmethod
    def _median(v):
        """np.median semantics (mean of the two middle values when n is even) as a 0-d tensor; v is 1-D, non-empty."""
        n = v.numel(); s = torch.sort(v).values
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    @torch.no_grad()
    def __call__(self, gray, alpha):
        """gray: HxW (0-255). alpha: HxW in [0,1] at the same size; numpy or tensor. -> (alpha (untouched, as given), info).
        Cues are computed at `scale` of the frame (1.0 = full; 0.5 was measured to cost clip4 recall@5 .81->.53, see
        veto_v3_summary.md 4.5) and the keep map is upsampled; the background's texture map is refreshed every 12 frames
        because a 60 s EMA moves nowhere in half a second."""
        g = self._t(gray)
        if self.scale != 1.0:
            g = F.interpolate(g[None, None], scale_factor=self.scale, mode="area")[0, 0]
        if self.bg is None or self.bg.shape != g.shape:               # no reference yet (first frame, a still, a new stream size):
            self.bg = g.clone(); self.Ebg = self.tex(self.bg); self._n = 0   # nothing to compare against -> pass the model through
            self.keep = None; self.hot = None; self.lens_blur = False
            return alpha, dict(lens_blur=False, r_median=1.0, vetoed_frac=0.0, first_frame=True)
        Ecur = self.tex(g)
        R = (Ecur + 1.0) / (self.Ebg + 1.0)
        sdiff = g - self.bg
        diff = sdiff.abs()
        textured = self.Ebg[::3, ::3] > 2.0
        if bool(textured.any()):
            r_med = self._median(R[::3, ::3][textured]); d_med = self._median(diff[::3, ::3][textured])
        else:
            r_med = torch.ones((), device=self.dev); d_med = torch.zeros((), device=self.dev)
        # lens defocus/fog: texture collapses everywhere while intensities barely move. A frame FULL of dust also collapses
        # texture, but it shifts intensities by tens of gray levels (clip1 at 5-8 s did exactly that and was wrongly vetoed
        # whole until this second condition existed), so both must hold.
        lens_blur = (r_med < self.blur_r) & (d_med <= self.blur_d)   # blur_d 8 (was a hard-coded `< 6.0`: sigma-5 blur on the bench still lands on exactly 6.00 and escaped)
        Rn = (R / r_med.clamp_min(1e-3)) if self.r_norm else R        # gain-invariant: a global exposure step scales every |Lap| alike
        # sign=+1: only pixels BRIGHTER than the reference count as haze - scattering dust over the dark steel floor
        # brightens; a charred pile / truck bed darkens. Kills the static-pile FP outright (burnt7 170 -> 0 frame-zones) at the
        # cost of backlit/absorbing haze (soot, dust in front of the bright doorway), which v3.1 barely sees anyway.
        # sign=0 restores |diff| (both directions); sign=-1 darker-only (a future soot model).
        dd = sdiff if self.sign > 0 else (-sdiff if self.sign < 0 else diff)
        keep = (dd >= self.diff_min) & (Ecur <= self.e_max) & (Rn >= self.r_min) & (Rn <= self.r_max)
        keep &= ~lens_blur
        a_t = self._t(alpha)
        if keep.shape != a_t.shape:
            keep = F.interpolate(keep[None, None].float(), size=a_t.shape, mode="nearest")[0, 0] > 0.5
        hot = a_t > self.thr
        n_hot = hot.sum()
        vfrac = torch.where(n_hot > 0, 1.0 - (keep & hot).sum() / n_hot.clamp_min(1), torch.zeros((), device=self.dev))
        r_med_f, d_med_f, vfrac_f, lb = torch.stack([r_med, d_med, vfrac, lens_blur.float()]).tolist()   # one GPU sync per frame
        self.keep, self.hot, self.lens_blur = keep, hot, bool(lb)
        info = dict(lens_blur=self.lens_blur, r_median=round(r_med_f, 3), d_median=round(d_med_f, 2), vetoed_frac=round(vfrac_f, 4))
        # background update AFTER the decision, plain EMA, alpha-agnostic (see module docstring)
        kk = 1.0 / (self.fps * self.ema_t)
        self.bg += kk * (g - self.bg)
        self._n += 1
        if self._n % 12 == 0: self.Ebg = self.tex(self.bg)
        return alpha, info

    @torch.no_grad()
    def factors(self, masks, min_px=50):
        """Batched factor(): masks BxHxW bool (numpy or tensor) -> list of B floats, one GPU sync for all bays."""
        B = int(masks.shape[0])
        if self.lens_blur: return [0.0] * B
        if self.keep is None or self.hot is None: return [1.0] * B
        M = self._t(masks)
        hm = self.hot[None] & M
        n = hm.sum((1, 2)); c = (self.keep[None] & hm).sum((1, 2))
        out = []
        for ni, ci in zip(n.tolist(), c.tolist()):
            if ni < min_px: out.append(1.0); continue
            frac = ci / ni
            out.append(0.0 if frac < self.f_min else min(1.0, frac / self.f_ref))   # f_min = optional hard floor (default off; see veto_v3_summary.md)
        return out

    def factor(self, mask, min_px=50):
        """Zone evidence weight in [0,1] for the fired pixels inside `mask`: clean_frac = share of fired pixels that pass the
        pixel rule; factor = min(1, clean_frac / f_ref). Multiply the zone's detected% by it.
        WHY a per-zone weight and not a per-pixel cut: zeroing pixels shaved TRUE dust to 35 % of its detected% (thick haze
        also wipes out background texture, R < r_min), so tau* could not be compared and the thin/mid/thick band lost its
        meaning. The probe showed the SHARE of rule-passing pixels separates a fired region that is haze (clip1 0.35,
        clip4 0.43, burnt6 0.30) from one that is a truck/pile (clip1 pre-dust 0.03, burnt7 0.10, clip4 pre-dust 0.18), so
        the weight is taken from that share and the magnitude of the haze itself is left alone. 1.0 when nothing fired or
        no reference yet; 0.0 on a lens-blur frame."""
        return self.factors(mask[None], min_px)[0]


if __name__ == "__main__":                                            # self-test: the lens-blur guard on the bench stills
    import sys
    from pathlib import Path
    HERE = Path(__file__).resolve().parent
    clean = cv2.imread(str(HERE.parent / "T0_clean_base.png"), cv2.IMREAD_GRAYSCALE)
    if clean is None: sys.exit("T0_clean_base.png not found next to qa/")
    fails = []
    a = np.ones(clean.shape, np.float32)
    m = np.ones(clean.shape, bool)
    print(f"lens-blur guard on T0_clean_base.png {clean.shape[1]}x{clean.shape[0]} (blur_r {DEFAULTS['blur_r']}, blur_d {DEFAULTS['blur_d']}):")
    for s in (0.0, 0.7, 1.0, 1.5, 2.0, 5.0):
        b = cv2.GaussianBlur(clean, (0, 0), s) if s > 0 else clean
        w = Veto(fps=24); w(clean.astype(np.float32), a)              # frame 0 = reference
        out, info = w(b.astype(np.float32), a)
        print(f"blur sigma {s:3.1f}: median R {info['r_median']:.3f} median diff {info['d_median']:.2f}  lens_blur={info['lens_blur']}  zone factor {w.factor(m):.3f}")
        if info["lens_blur"] != (s >= 1.0): fails.append(s)
    print("OK" if not fails else f"FAIL at sigma {fails}"); sys.exit(1 if fails else 0)
