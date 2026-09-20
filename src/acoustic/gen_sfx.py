# -*- coding: utf-8 -*-
"""สร้างเสียงจำลองด้วย Stable Audio Open 1.0 บน GPU เครื่องนี้ (gated repo ต้องมี HF_TOKEN)

  python gen_sfx.py --only A1 --takes 1      # ลองตัวเดียวก่อนเช็คสเปกตรัม
  python gen_sfx.py                          # ครบทุกกลุ่ม

ทุกคลิป 44.1 kHz stereo (ตัดเป็น mono ตอนผสม) · ไฟล์ออกที่ gen_audio/raw/<id>_t<k>.wav
กลุ่ม A = สิ่งแปลกปลอมที่ต้องจับให้ได้ · B = พื้นหลัง · C = เสียงหลอกที่ต้องไม่แจ้งเตือน
มุมรับเสียง = contact mic แปะบนเหล็ก (structure-borne 1–10 kHz) ตาม research/cheap_sensor_ai_hypotheses.md H1
"""
import argparse, os, time
from pathlib import Path
import soundfile as sf
import torch
from diffusers import StableAudioPipeline

OUT = Path(__file__).resolve().parents[3] / "gen_audio" / "raw"
MIC = ("Recorded by a contact microphone bolted to the steel plate itself, structure-borne sound, "
       "sharp 1-10 kHz transient, bass-light, dry, no room reverb, no music.")
NEG = "music, speech, voice, reverb, echo, fade in, fade out, soft, smooth, lo-fi"

PROMPTS = {
    # A — ต้องจับให้ได้
    "A1": ("A single M12 steel hex nut dropped from 1.5 metres onto a 6 mm thick steel conveyor trough, "
           "bouncing twice then settling, bright metallic ping with short ring-out. " + MIC, 3),
    "A2": ("A handful of small steel bolts, nuts and washers scattering across a steel conveyor trough, "
           "six to ten separate sharp impacts over 1.5 seconds. " + MIC, 3),
    "A3": ("A heavy steel scrap object, a wrench or flat bar, clanging once onto a steel trough floor, "
           "low pitched with a long metallic ring. " + MIC, 3),
    "A4": ("A fist-sized granite stone falling onto a steel trough plate, dull heavy thud with a short "
           "rattle, no metallic ring. " + MIC, 3),
    "A5": ("A scoop of coarse gravel and small stones pattering onto a steel trough plate. " + MIC, 3),
    "A6": ("A steel nut landing on a 30 cm deep layer of sugarcane stalks lying on a steel trough, the "
           "impact absorbed by the cane, only a faint muffled thud reaches the steel plate below. " + MIC, 3),
    # B — พื้นหลัง (ยาว วนได้)
    "B1": ("Continuous sugarcane stalks sliding and tumbling along a steel conveyor trough, dry fibrous "
           "scraping and rustling, steady flow, no impacts, structure-borne through the steel. " + MIC, 20),
    "B2": ("Industrial conveyor drive, electric motor hum and gearbox whine under steady load, "
           "structure-borne through the steel frame, continuous. " + MIC, 20),
    "B3": ("Sugar mill yard machinery: diesel truck idling, hydraulic tippler running, distant cane "
           "crushing, transmitted through the steel structure. " + MIC, 20),
    "B4": ("Heavy sugarcane flow onto a steel trough at full throughput, dense continuous rumble and "
           "scraping that masks fine detail. " + MIC, 20),
    # D — ทรายไหล (ดินทราย/ทรายที่หลุดออกมาตอนเท) — เป็น "เสียงต่อเนื่อง" ไม่ใช่กระแทก จับคนละวิธีกับ A
    "D1": ("Dry sand pouring in a steady stream onto a bare concrete floor from one metre, continuous "
           "fine hiss, no impacts. Close mic on the concrete slab, structure-borne, dry, no reverb.", 10),
    "D2": ("Dry sand sliding down a steel chute and spilling onto a concrete floor, sustained "
           "high-frequency hiss with a grainy rattle against the steel. " + MIC, 10),
    "D3": ("A thin trickle of fine sand running onto concrete, very quiet continuous hiss, "
           "close mic on the concrete slab, dry, no reverb.", 10),
    "D4": ("A heavy load of sand and soil dumped onto a concrete floor all at once, then trailing "
           "grains scattering. Close mic on the concrete slab, dry, no reverb.", 10),
    "D5": ("Sugarcane stalks and loose sand tumbling together onto a concrete floor, woody knocks "
           "mixed with continuous sandy hiss. Close mic on the concrete slab, dry, no reverb.", 10),
    # C — เสียงหลอก ต้องไม่แจ้งเตือน
    "C1": ("The cut butt end of a sugarcane stalk striking a steel trough hard, a woody knock, "
           "not metallic, no ring. " + MIC, 3),
    "C2": ("A steel chain slapping against the side of a steel trough, several rattling links. " + MIC, 3),
    "C3": ("A worker hammering metal twenty metres away, muffled and distant, arriving through the "
           "steel structure. " + MIC, 3),
    "C4": ("Thermal expansion tick of a large steel structure, one sharp isolated click, then silence. " + MIC, 3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="เช่น A1 A6")
    ap.add_argument("--takes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args()

    tok = os.environ.get("HF_TOKEN")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0",
                                               torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
                                               token=tok).to(dev)
    pipe.set_progress_bar_config(disable=True)
    OUT.mkdir(parents=True, exist_ok=True)
    keys = a.only or list(PROMPTS)
    for k in keys:
        prompt, secs = PROMPTS[k]
        for t in range(a.takes):
            g = torch.Generator(dev).manual_seed(a.seed + 1000 * keys.index(k) + t)
            t0 = time.perf_counter()
            audio = pipe(prompt, negative_prompt=NEG, num_inference_steps=a.steps,
                         audio_end_in_s=float(secs), num_waveforms_per_prompt=1, generator=g).audios
            w = audio[0].T.float().cpu().numpy()                    # (samples, 2)
            p = OUT / f"{k}_t{t}.wav"
            sf.write(p, w, pipe.vae.sampling_rate)
            print(f"{p.name}  {secs}s  {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
