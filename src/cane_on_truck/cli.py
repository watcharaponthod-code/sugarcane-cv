# -*- coding: utf-8 -*-
"""CLI: cane-detect <image_or_folder> [--out annotated_dir] [--config site.json]"""
import os, glob, argparse, json
import cv2
from .core import CaneDetector, DetectorConfig

def main():
    ap = argparse.ArgumentParser(description="Cane load detection (CANE / COVERED / EMPTY)")
    ap.add_argument("path", help="image file or folder")
    ap.add_argument("--out", default=None, help="folder for annotated output")
    ap.add_argument("--config", default=None, help="per-site config JSON")
    args = ap.parse_args()

    cfg = DetectorConfig.from_json(args.config) if args.config else None
    det = CaneDetector(cfg)

    files = ([args.path] if os.path.isfile(args.path)
             else sorted(sum([glob.glob(os.path.join(args.path, e))
                              for e in ("*.jpg", "*.jpeg", "*.png")], [])))
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    counts = {}
    for f in files:
        try:
            res = det.detect(f)
        except ValueError:
            print(f"{f}: read error"); continue
        counts[res.label] = counts.get(res.label, 0) + 1
        print(f"{os.path.basename(f):40s} {res.label:8s} ctex={res.cane_tex} top={res.top_med}")
        if args.out:
            cv2.imwrite(os.path.join(args.out, os.path.basename(f)), det.annotate(f, res))
    print("---"); print(json.dumps(counts, ensure_ascii=False))

if __name__ == "__main__":
    main()
