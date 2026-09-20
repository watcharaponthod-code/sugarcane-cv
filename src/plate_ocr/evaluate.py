#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""วัดความแม่นของผลอ่าน (CSV จาก main.py) เทียบกับ ground truth

วิธีใช้:  python evaluate.py results.csv
แก้ GROUND_TRUTH ด้านล่างให้ตรงกับชุดทดสอบของตัวเอง
"""
import csv
import sys
import re

# เลขทะเบียนจริง (เฉพาะภาพที่อยู่ในขอบเขตระบบ: รถยนต์/รถบรรทุกป้ายไทย)
GROUND_TRUTH = {
    "truck_cane.png":  ("83-6237", None),              # รถบรรทุกอ้อย (ป้าย ~70px)
    "pd_thai_car.jpg": ("1กส 9659", "กรุงเทพมหานคร"),
    "tc_car2.jpg":     ("1กภ 4437", "กรุงเทพมหานคร"),
    "tc_car3.jpg":     ("ชล 5500", "กรุงเทพมหานคร"),
    "tc_car4.jpg":     ("7กญ 3603", "กรุงเทพมหานคร"),
    "tc_0.jpg":        ("1กภ 4437", None),             # ป้าย crop เบลอ
    "tc_1.jpg":        ("1กภ 4437", "กรุงเทพมหานคร"),  # ป้าย crop ชัด
}
# ภาพนอกขอบเขต (ไม่นับคะแนน): มอเตอร์ไซค์, ป้ายอังกฤษสังเคราะห์,
# ภาพกลางคืนที่ไม่ทราบเลขจริง, ภาพไกลเกินอ่าน


def char_overlap(a, b):
    """สัดส่วนตัวอักษร (ไทย+เลข) ของ b ที่อยู่ใน a ตามลำดับ"""
    strip = lambda s: re.sub(r"[^0-9ก-ฮ]", "", s or "")
    a, b = strip(a), strip(b)
    if not a or not b:
        return 0.0
    # longest common subsequence แบบง่าย
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] \
                else max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n] / max(m, n)


def main(csv_path):
    best = {}   # ไฟล์ -> แถวแรก (ผลอันดับ 1) ของไฟล์นั้น
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            src = row["แหล่งภาพ"].split("/")[-1]
            if src not in best:
                best[src] = row

    exact = partial = prov_ok = prov_total = 0
    print(f"{'ภาพ':<18}{'เลขจริง':<12}{'อ่านได้':<12}"
          f"{'ตรง':<7}{'จังหวัด':<16}ใกล้เคียง")
    print("-" * 75)
    for img, (gt_num, gt_prov) in GROUND_TRUTH.items():
        row = best.get(img)
        got = row["เลขทะเบียน"] if row else "-"
        got_prov = (row["จังหวัด"] or "-") if row else "-"
        ov = char_overlap(gt_num, got)
        is_exact = re.sub(r"\s", "", got) == re.sub(r"\s", "", gt_num)
        exact += is_exact
        partial += ov >= 0.6
        if gt_prov:
            prov_total += 1
            prov_ok += (got_prov == gt_prov)
        print(f"{img:<18}{gt_num:<12}{got:<12}"
              f"{'✔' if is_exact else '✘':<6}{got_prov:<16}{ov:.0%}")

    n = len(GROUND_TRUTH)
    print("-" * 75)
    print(f"เลขตรงเป๊ะ    : {exact}/{n}  ({exact / n:.0%})")
    print(f"ใกล้เคียง>=60%: {partial}/{n}  ({partial / n:.0%})")
    if prov_total:
        print(f"จังหวัดถูก    : {prov_ok}/{prov_total}  "
              f"({prov_ok / prov_total:.0%})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
