# -*- coding: utf-8 -*-
"""เลือกภาพ 150 ใบให้คนวาดกรอบ (ยอดยาว / กระจุกใบแห้ง) — ไม่วาดเอง ไม่เทรน

  python leaf_label_set.py            # เขียน gen_images/leaf_label_set/ + manifest.json

กติกา: หนึ่งกลุ่มรถหนึ่งใบ (กันรถคันเดียวกันซ้ำ) · กัน near-duplicate ด้วย dHash <= 4
· กระจายทุกวันทุกช่วงเวลา · seed คงที่ · test = วันสุดท้ายที่มีทั้งสองกล้อง กำหนดก่อนใครวาดสักกรอบ
"""
import json, re, shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2, numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SRC = ROOT / "gen_images" / "roboflow_cane"
DST = ROOT / "gen_images" / "leaf_label_set"
WANT = {"cane_fresh": 90, "cane_cut": 60}
SEED = 0
DHASH_MAX = 4          # ใกล้กันกว่านี้ถือว่าเป็นภาพซ้ำ
BANDS = [(0, 6), (6, 12), (12, 18), (18, 24)]   # ช่วงเวลาในวัน


def dhash(path, size=8):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, (size + 1, size), interpolation=cv2.INTER_AREA)
    return int("".join("1" if b else "0" for b in (im[:, 1:] > im[:, :-1]).flatten()), 2)


def band_of(t):
    h = int(t[:2])
    for i, (a, b) in enumerate(BANDS):
        if a <= h < b:
            return i
    return 0


def main():
    groups = json.loads((HERE / "rf_audit_out" / "truck_groups.json").read_text(encoding="utf-8"))["image_to_group"]
    rng = np.random.RandomState(SEED)

    pool = defaultdict(list)      # คลาส -> รายการภาพ
    for sp in ("train", "valid", "test"):
        for cls in WANT:
            for p in sorted((SRC / sp / cls).glob("*.jpg")):
                m = re.match(r"(\d{8})-(\d{6})_([A-Za-z0-9]+)", p.name)
                if not m:
                    continue
                day, tm, cam = m.groups()
                pool[cls].append(dict(path=p, file=p.name, cls=cls, day=day, time=tm, cam=cam,
                                      band=band_of(tm), group=groups.get(p.name, "ungrouped:" + p.name)))

    print("ภาพทั้งหมดในคลัง:", {c: len(v) for c, v in pool.items()})
    for cls, rows in pool.items():
        print(f"  {cls}: กล้อง {dict(Counter(r['cam'] for r in rows))} · วัน {dict(Counter(r['day'] for r in rows))}")

    picked, seen_groups, hashes = [], set(), []
    for cls, n_want in WANT.items():
        rows = pool[cls]
        # ถังละ (วัน, ช่วงเวลา) แล้ววนหยิบทีละถัง — ได้การกระจายทั้งวันและเวลาโดยไม่ต้องถ่วงน้ำหนักเอง
        buckets = defaultdict(list)
        for r in rows:
            buckets[(r["day"], r["band"])].append(r)
        for k in buckets:
            rng.shuffle(buckets[k])
        keys = sorted(buckets)
        got, i, guard = 0, 0, 0
        while got < n_want and guard < 100000:
            guard += 1
            k = keys[i % len(keys)]
            i += 1
            if not buckets[k]:
                continue
            r = buckets[k].pop()
            if r["group"] in seen_groups:          # หนึ่งกลุ่มรถหนึ่งใบ
                continue
            h = dhash(r["path"])
            if h is None:
                continue
            if any(bin(h ^ g).count("1") <= DHASH_MAX for g in hashes):   # ภาพซ้ำระดับพิกเซล
                continue
            seen_groups.add(r["group"])
            hashes.append(h)
            picked.append(r)
            got += 1
        print(f"  เลือก {cls}: {got}/{n_want}")

    # test = วันสุดท้ายที่มี "ทั้งสองกล้อง" (แก้ 2026-09-11 จากกฎเดิม "วันสุดท้าย")
    # เหตุผล: 20230119 มีแต่ MPDC00 -> test วัดข้ามกล้องไม่ได้ ซึ่งเป็นคำถามหลักที่เราจะถูกถาม
    by_day_cam = defaultdict(set)
    for r in picked:
        by_day_cam[r["day"]].add(r["cam"])
    both = [d for d in sorted(by_day_cam) if len(by_day_cam[d]) >= 2]
    test_day = both[-1] if both else sorted(by_day_cam)[-1]
    DST.mkdir(parents=True, exist_ok=True)
    man = []
    for r in picked:
        shutil.copy2(r["path"], DST / r["file"])
        man.append(dict(file=r["file"], mill_class=r["cls"], camera=r["cam"], day=r["day"],
                        time=r["time"], truck_group=r["group"],
                        split="test" if r["day"] == test_day else "train"))

    (DST / "manifest.json").write_text(json.dumps({
        "note": "ชุดให้คนวาดกรอบ 2 คลาส (ยอดยาว / กระจุกใบแห้ง) — ดู leaf_label_GUIDE.md",
        "source": "Roboflow Universe aimlsugarcane/cane-classification-classify v6 (CC BY 4.0)",
        "seed": SEED, "dhash_max": DHASH_MAX, "one_image_per_truck_group": True,
        "test_day": test_day, "test_rule": "วันสุดท้ายที่มีทั้งสองกล้องเป็น test ทั้งหมด · กำหนดก่อนวาดกรอบแรก · ล็อกตาย ห้ามเปลี่ยนย้อนหลัง",
        "test_rule_history": [
            {"rule": "วันสุดท้าย", "test_day": "20230119", "status": "ยกเลิก"},
            {"rule": "วันสุดท้ายที่มีทั้งสองกล้อง", "test_day": "20230118",
             "changed_at": "2026-09-11", "changed_before_any_label": True,
             "why": "20230119 มีแต่กล้อง MPDC00 -> test วัดข้ามกล้องไม่ได้ ซึ่งเป็นข้อที่ทีมพลาดซ้ำทุกรอบ · 20230118 มีทั้ง MPDC00 และ MPK00"}
        ],
        "n_images": len(man),
        "by_class": dict(Counter(r["mill_class"] for r in man)),
        "by_camera": dict(Counter(r["camera"] for r in man)),
        "by_day": dict(Counter(r["day"] for r in man)),
        "by_split": dict(Counter(r["split"] for r in man)),
        "by_camera_day": {f"{c}|{d}": n for (c, d), n in
                          sorted(Counter((r["camera"], r["day"]) for r in man).items())},
        "images": man,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nเขียน {DST} · {len(man)} ใบ")
    print("  คลาส:", dict(Counter(r["mill_class"] for r in man)))
    print("  กล้อง:", dict(Counter(r["camera"] for r in man)))
    print("  วัน:", dict(sorted(Counter(r["day"] for r in man).items())))
    print("  split:", dict(Counter(r["split"] for r in man)), f"(test = {test_day})")
    print("  กล้อง x วัน:", {f"{c}|{d}": n for (c, d), n in sorted(Counter((r["camera"], r["day"]) for r in man).items())})


if __name__ == "__main__":
    main()
