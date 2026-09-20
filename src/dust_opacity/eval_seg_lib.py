# -*- coding: utf-8 -*-
"""Bits of the eval harness shared by eval_seg.py and visuals.py (eval_seg.py itself parses argv at import time)."""
import numpy as np, cv2


def truck_mask(bgr, cfg):
    """Rigid mask of the (red) truck cab+bed so its body never counts as dust. cfg = clip["truck_mask"] of eval_set.json.
    HSV red on both hue wraps -> close -> keep components >= min_area (bbox-filled: cab + bed are one rigid body) -> dilate by pad."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, tuple(cfg["hsv_lo1"]), tuple(cfg["hsv_hi1"])) | cv2.inRange(hsv, tuple(cfg["hsv_lo2"]), tuple(cfg["hsv_hi2"]))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= cfg["min_area"]:
            x, y, w, h, _ = stats[i]
            keep[y:y + h, x:x + w] |= (lab[y:y + h, x:x + w] == i).astype(np.uint8)
    if not keep.any():
        return np.zeros(bgr.shape[:2], bool)
    k = int(cfg["pad"]) * 2 + 1
    return cv2.dilate(keep, np.ones((k, k), np.uint8)).astype(bool)
