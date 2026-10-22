"""Turn the podcast-processed takes into Qwen3-TTS fine-tune input. Local, no GPU.

    uv run training/prepare.py                 # data/processed/podcast -> data/prepared
    uv run training/prepare.py --ref <id>      # pin the reference take
    uv run training/prepare.py --repick-ref    # choose a new reference even if the old one is still kept

Input: `scripts/prep_clean.py` output (48 kHz / 24-bit audiofolder + metadata.jsonl).
  train: data/processed/podcast/            (from data/takes/)
  test:  data/processed/test/podcast/       (from data/takes/test/; optional, listening test only)

Output (data/prepared/, gitignored):
  train/<id>.wav, test/<id>.wav   24 kHz mono 16-bit, silence trimmed to PAD_HEAD_S before, PAD_TAIL_S after
  train_raw.jsonl                 {"audio","text","ref_audio"}, paths RELATIVE to data/prepared/
                                  (training/modal_app.py makes them absolute on the volume)
  test.jsonl                      {"id","audio","text","set","style"}: listening-test lines, never trained on
  ref.wav, ref.json               the one fixed reference (every train row points at it)
  summary.json                    hours kept / dropped, by reason

Filters (docs/training-plan.md "Prep"): train split only for training; drop clipped_samples > 0,
browser_processing, transcript_status == "pending", empty text, and any take whose prompt is in the
test split. Reference: an unedited take of 6-10 s after trimming, from the `conversational` or
`arctic` set if one is that long, else `long` or `tech` (REF_SETS); the previous pick is kept while it survives the filters, so
growing the dataset does not change the speaker embedding between runs.
"""
import argparse
import json
import shutil
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
OUT_SR = 24000
PAD_HEAD_S = 0.15
# 0.5 s of the take's own trailing room tone (where it has that much): fine-tunes with tight tails
# learned to rush and run on (QwenLM/Qwen3-TTS#39 needed ~1 s; docs/data-guide.md).
PAD_TAIL_S = 0.5
FRAME_S = 0.02
MIN_S = 0.5
# Reference preference: conversational, then arctic. Their sentences are short (longest 4.8 / 5.1 s
# on 2026-10-21), so `long` and `tech` (plain English, no Hebrew words) are the fallback.
REF_SETS = ("conversational", "arctic", "long", "tech")
REF_STYLES = {"neutral", "conversational", "long"}


def trim(x, sr):
    """Cut to the first/last frame clearly above the take's own floor, plus PAD_HEAD_S / PAD_TAIL_S."""
    fr = int(sr * FRAME_S)
    n = len(x) // fr
    if n == 0:
        return x
    db = 20 * np.log10(np.sqrt(np.mean(x[: n * fr].reshape(n, fr) ** 2, axis=1)) + 1e-9)
    active = np.where(db > max(np.percentile(db, 10) + 10, -60))[0]
    if len(active) == 0:
        return x[:0]
    return x[max(0, active[0] * fr - int(PAD_HEAD_S * sr)): min(len(x), (active[-1] + 1) * fr + int(PAD_TAIL_S * sr))]


def to_24k(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = trim(x, sr)
    g = gcd(OUT_SR, sr)
    return resample_poly(x, OUT_SR // g, sr // g).astype("float32") if len(x) else x


def drop_reason(r, test_prompts):
    if (r.get("clipped_samples") or 0) > 0:
        return "clipped"
    if r.get("browser_processing"):
        return "browser_processing"
    if r.get("transcript_status") == "pending":
        return "transcript_pending"
    if not (r.get("text") or "").strip():
        return "empty_text"
    if r.get("prompt_id") and r["prompt_id"] in test_prompts:
        return "prompt_in_test_split"
    return None


def read_rows(d):
    p = d / "metadata.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else None


def pick_ref(kept, prev_id, forced_id, repick):
    by_id = {k["row"]["id"]: k for k in kept}
    if forced_id:
        if forced_id not in by_id:
            raise SystemExit(f"--ref {forced_id}: not among the kept train takes")
        return by_id[forced_id]
    if prev_id in by_id and not repick:
        return by_id[prev_id]
    cands = [k for k in kept if k["row"].get("set") in REF_SETS and k["row"].get("style") in REF_STYLES
             and not k["row"].get("edited") and 6 <= k["dur"] <= 10]
    if not cands:
        raise SystemExit(f"no reference candidate: need an unedited take of 6-10 s from {REF_SETS} "
                         f"with style in {sorted(REF_STYLES)}; pass --ref <id>")
    # earliest set in REF_SETS first (conversational = Irish cadence), then cleanest, then closest to 8 s
    cands.sort(key=lambda k: (REF_SETS.index(k["row"]["set"]), -(k["row"].get("snr_db") or 0), abs(k["dur"] - 8)))
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data/processed/podcast"))
    ap.add_argument("--test", default=str(ROOT / "data/processed/test/podcast"))
    ap.add_argument("--out", default=str(ROOT / "data/prepared"))
    ap.add_argument("--ref", help="take id to use as the fixed reference")
    ap.add_argument("--repick-ref", action="store_true")
    a = ap.parse_args()
    src, tsrc, out = Path(a.train), Path(a.test), Path(a.out)

    rows = read_rows(src)
    if rows is None:
        raise SystemExit(f"{src}/metadata.jsonl missing: run scripts/prep_clean.py first")
    test_rows = read_rows(tsrc) or []
    test_prompts = {r.get("prompt_id") for r in test_rows} - {None}

    prev_ref = None
    if (out / "ref.json").exists():
        prev_ref = json.loads((out / "ref.json").read_text()).get("id")

    for d in ("train", "test"):  # rebuilt from scratch, so a removed take cannot linger and get uploaded
        shutil.rmtree(out / d, ignore_errors=True)
    (out / "train").mkdir(parents=True, exist_ok=True)
    kept, dropped = [], {}
    for r in rows:
        why = drop_reason(r, test_prompts)
        y = None
        if not why:
            y = to_24k(src / r["file_name"])
            if len(y) < MIN_S * OUT_SR:
                why = "too_short_after_trim"
        if why:
            d = dropped.setdefault(why, {"takes": 0, "seconds": 0.0})
            d["takes"] += 1
            d["seconds"] += r.get("duration_s") or 0
            continue
        rel = f"train/{r['id']}.wav"
        sf.write(out / rel, y, OUT_SR, subtype="PCM_16")
        kept.append({"row": r, "rel": rel, "dur": len(y) / OUT_SR})

    ref = pick_ref(kept, prev_ref, a.ref, a.repick_ref)
    (out / "ref.wav").write_bytes((out / ref["rel"]).read_bytes())
    (out / "ref.json").write_text(json.dumps({
        "id": ref["row"]["id"], "text": ref["row"]["text"], "set": ref["row"].get("set"),
        "style": ref["row"].get("style"), "duration_s": round(ref["dur"], 2)}, indent=2, ensure_ascii=False) + "\n")
    with open(out / "train_raw.jsonl", "w") as f:
        for k in kept:
            f.write(json.dumps({"audio": k["rel"], "text": k["row"]["text"].strip(), "ref_audio": "ref.wav"},
                               ensure_ascii=False) + "\n")

    test_kept = []
    if test_rows:
        (out / "test").mkdir(exist_ok=True)
        for r in test_rows:
            if drop_reason(r, set()) or not (tsrc / r["file_name"]).exists():
                continue
            y = to_24k(tsrc / r["file_name"])
            if len(y) < MIN_S * OUT_SR:
                continue
            rel = f"test/{r['id']}.wav"
            sf.write(out / rel, y, OUT_SR, subtype="PCM_16")
            test_kept.append({"id": r["id"], "audio": rel, "text": r["text"].strip(),
                              "set": r.get("set"), "style": r.get("style")})
    with open(out / "test.jsonl", "w") as f:
        f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in test_kept)

    kept_s = sum(k["dur"] for k in kept)
    summary = {
        "train_takes_in": len(rows),
        "kept": {"takes": len(kept), "hours": round(kept_s / 3600, 3), "minutes": round(kept_s / 60, 1),
                 "mean_clip_s": round(kept_s / max(1, len(kept)), 2)},
        "dropped": {k: {"takes": v["takes"], "minutes": round(v["seconds"] / 60, 1)} for k, v in dropped.items()},
        "test": {"takes": len(test_kept), "source": str(tsrc) if test_rows else None},
        "ref": json.loads((out / "ref.json").read_text()),
        "probe_threshold_h": 1.0,
        "main_run_threshold_h": 2.5,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"train: kept {len(kept)}/{len(rows)} takes, {kept_s / 3600:.3f} h ({kept_s / 60:.1f} min) after trim")
    for k, v in summary["dropped"].items():
        print(f"  dropped {k}: {v['takes']} takes, {v['minutes']} min")
    if not test_rows:
        print(f"test: none ({tsrc} missing). Fetch data/test/ from the HF dataset into data/takes/test/ and run "
              "scripts/prep_clean.py --takes data/takes/test --out data/processed/test (training/run.sh does both)")
    else:
        print(f"test: {len(test_kept)} listening-test lines")
    print(f"ref: {summary['ref']['id']} ({summary['ref']['duration_s']} s, {summary['ref']['set']}/{summary['ref']['style']}): "
          f"{summary['ref']['text']}")
    print(f"probe needs 1 h: {'READY' if kept_s >= 3600 else f'{(3600 - kept_s) / 60:.0f} min short'}")


if __name__ == "__main__":
    main()
