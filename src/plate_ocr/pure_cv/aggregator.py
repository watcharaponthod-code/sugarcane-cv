# -*- coding: utf-8 -*-
"""โหวตทะเบียนข้ามหลายเฟรม (สำหรับ live video / RTSP)

หลักการ: กล้องเห็นรถคันเดิมหลายเฟรม การอ่านแต่ละเฟรมอาจเพี้ยนต่างกัน
(เฟรมหนึ่งอ่าน 93-6227 อีกเฟรม 83-6237) — จับเฟรมที่อ่าน "คล้ายกัน"
มาเข้ากลุ่มเดียว แล้วโหวตรายตำแหน่งตัวอักษรถ่วงน้ำหนักด้วย confidence
ผลสุดท้ายจึงนิ่งและแม่นกว่าอ่านเฟรมเดียวมาก — เทคนิคเดียวกับระบบ LPR เชิงพาณิชย์
"""
import re
import time
import difflib
from collections import Counter, defaultdict

from . import config as C


def _norm(number):
    """เหลือเฉพาะตัวอักษรสาระ (ไทย+เลข) ไว้เทียบความคล้าย"""
    return re.sub(r"[^0-9ก-ฮ]", "", number or "")


class PlateVote:
    """สะสมผลอ่านรายเฟรม -> ยืนยันทะเบียนเมื่อหลักฐานพอ

    ตัวอย่างการใช้ (ดู main.py โหมดวิดีโอ):
        vote = PlateVote()
        ...ทุกเฟรม...
        confirmed = vote.add(result)   # คืน dict เมื่อ 'ยืนยันได้' ครั้งแรก
        if confirmed:
            print(confirmed["number"], confirmed["votes"])
    """

    def __init__(self, min_frames=None, dedup_seconds=None):
        self.min_frames = min_frames or C.VOTE_MIN_FRAMES
        self.dedup = dedup_seconds or C.DEDUP_SECONDS
        self.clusters = []   # [{reads:[(num,prov,conf)], last:t, emitted:bool}]

    # ------------------------------------------------------------ ภายใน
    def _find_cluster(self, number):
        key = _norm(number)
        for cl in self.clusters:
            rep = _norm(cl["reads"][0][0])
            if difflib.SequenceMatcher(None, key, rep).ratio() >= 0.6:
                return cl
        return None

    @staticmethod
    def _vote_number(reads):
        """โหวตรายตำแหน่ง: ใช้เฉพาะผลอ่านที่ความยาวเท่ากับความยาวส่วนใหญ่
        แต่ละตำแหน่งเลือกตัวอักษรที่ผลรวม confidence สูงสุด"""
        lens = Counter(len(_norm(n)) for n, _, _ in reads)
        target_len = lens.most_common(1)[0][0]
        rows = [(n, c) for n, _, c in reads if len(_norm(n)) == target_len]
        raw_rows = [(_norm(n), c) for n, c in rows]

        voted = []
        for i in range(target_len):
            weight = defaultdict(float)
            for s, c in raw_rows:
                weight[s[i]] += max(c, 0.05)
            voted.append(max(weight.items(), key=lambda kv: kv[1])[0])
        voted = "".join(voted)

        # คืนรูปแบบสวยงามจากผลอ่านต้นฉบับที่ตรงกับผลโหวตมากที่สุด
        best_fmt = max(rows, key=lambda r: difflib.SequenceMatcher(
            None, _norm(r[0]), voted).ratio())[0]
        if _norm(best_fmt) == voted:
            return best_fmt
        # ประกอบรูปแบบใหม่ตามโครงของ best_fmt (คงขีด/ช่องว่าง)
        out, j = [], 0
        for ch in best_fmt:
            if re.match(r"[0-9ก-ฮ]", ch):
                out.append(voted[j] if j < len(voted) else ch)
                j += 1
            else:
                out.append(ch)
        return "".join(out)

    # ------------------------------------------------------------ public
    def add(self, result):
        """ป้อนผลอ่าน 1 เฟรม (dict จาก process_image หรือ None)

        Returns
        -------
        dict {number, province, votes, avg_conf} เมื่อทะเบียนนี้ 'ยืนยันได้'
        เป็นครั้งแรก (สะสมครบ min_frames) — นอกนั้นคืน None
        """
        now = time.time()
        # ล้างกลุ่มเก่าที่เงียบไปนาน (รถคันเดิมกลับมาใหม่ = นับใหม่)
        self.clusters = [cl for cl in self.clusters
                         if now - cl["last"] < self.dedup * 3]
        if not result or not result.get("number"):
            return None

        num = result["number"]
        cl = self._find_cluster(num)
        if cl is None:
            cl = {"reads": [], "last": now, "emitted": False}
            self.clusters.append(cl)
        cl["reads"].append((num, result.get("province"),
                            float(result.get("confidence", 0))))
        cl["last"] = now

        if cl["emitted"] or len(cl["reads"]) < self.min_frames:
            return None

        cl["emitted"] = True
        provs = [p for _, p, _ in cl["reads"] if p]
        return {
            "number": self._vote_number(cl["reads"]),
            "province": Counter(provs).most_common(1)[0][0] if provs else None,
            "votes": len(cl["reads"]),
            "avg_conf": round(sum(c for _, _, c in cl["reads"])
                              / len(cl["reads"]), 3),
        }
