"""Make the processed copy of the takes that training uses. Raw takes are read, never written.

    uv run scripts/prep_clean.py                      # variant podcast (the training default)
    uv run scripts/prep_clean.py --variant clean      # denoise + high-pass + loudness only
    uv run scripts/prep_clean.py --ab 6               # raw / clean / podcast trios to listen to

Variant `podcast` (decided 2026-10-21) runs the chain in config/podcast-chain.json:
stationary denoise -> Daniel's podcast voice EQ -> split-band de-esser -> normalise -> 2:1
compressor -> normalise to -20 LUFS with a -1 dBFS peak ceiling.

Output: data/processed/<variant>/{audio/*.wav, metadata.jsonl} (48 kHz / 24-bit, same file names,
an HF audiofolder like the source), each row with a `processing` record. data/processed/ is
gitignored. Decisions and measurements: docs/voice-profile.md.
"""
import argparse
import json
import subprocess
from pathlib import Path

import noisereduce as nr
import numpy as np
import pyloudnorm
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
FRAME, HOP = 2048, 512


def noise_sample(x):
    """The quietest frames of the take (within 3 dB of its 10th-percentile RMS), concatenated."""
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    rms = np.array([np.sqrt(np.mean(x[i * HOP:i * HOP + FRAME] ** 2)) + 1e-12 for i in range(n)])
    db = 20 * np.log10(rms)
    keep = np.where(db <= np.percentile(db, 10) + 3)[0]
    return np.concatenate([x[i * HOP:i * HOP + HOP] for i in keep]), float(np.percentile(db, 10))


def ffmpeg(x, sr, af):
    """Run a mono float signal through an ffmpeg filter chain (the podcast pipeline's syntax)."""
    cmd = ["ffmpeg", "-v", "error", "-f", "f64le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-af", af, "-f", "f64le", "-ar", str(sr), "-ac", "1", "pipe:1"]
    out = subprocess.run(cmd, input=x.astype("<f8").tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<f8").copy()


def envelope(x, sr, attack_ms, release_ms):
    a = np.exp(-1 / (sr * attack_ms / 1000))
    r = np.exp(-1 / (sr * release_ms / 1000))
    env, e = np.empty_like(x), 0.0
    for i, v in enumerate(np.abs(x)):
        c = a if v > e else r
        e = c * e + (1 - c) * v
        env[i] = e
    return env


def deess(x, sr, cfg):
    """Split-band de-esser: reduce only the sibilance band, only while it dominates the signal.
    The band is split zero-phase, so band + rest == x exactly when no reduction is applied."""
    lo, hi = cfg["band_hz"]
    band = signal.sosfiltfilt(signal.butter(4, [lo, hi], "bandpass", fs=sr, output="sos"), x)
    rest = x - band
    eb = envelope(band, sr, cfg["attack_ms"], cfg["release_ms"]) + 1e-9
    ex = envelope(x, sr, cfg["attack_ms"], cfg["release_ms"]) + 1e-9
    over = 20 * np.log10(eb / ex) - cfg["threshold_db"]
    red_db = np.clip(over * (1 - 1 / cfg["ratio"]), 0, cfg["max_reduction_db"])
    out = rest + band * 10 ** (-red_db / 20)
    return out, round(float(np.percentile(red_db, 99)), 1), round(float(np.mean(red_db > 1) * 100), 1)


def to_lufs(x, meter, target):
    return x * 10 ** ((target - meter.integrated_loudness(x)) / 20)


def process(x, sr, chain, variant, meter):
    steps = {"variant": variant}
    noise, floor_before = noise_sample(x)
    prop = chain["denoise"]["prop_decrease"]
    if prop > 0 and len(noise) >= sr // 4:
        x = nr.reduce_noise(y=x, sr=sr, y_noise=noise, stationary=True, prop_decrease=prop)
        steps["denoise"] = {"prop_decrease": prop, "approx_reduction_db": round(-20 * np.log10(1 - prop), 1)}
    if variant == "clean":
        hp = chain["eq"]["ffmpeg"][0]
        x = ffmpeg(x, sr, hp)
        steps["eq"] = [hp]
    else:
        x = ffmpeg(x, sr, ",".join(chain["eq"]["ffmpeg"]))
        steps["eq"] = chain["eq"]["ffmpeg"]
        x, p99, pct = deess(x, sr, chain["deesser"])
        steps["deesser"] = {"p99_reduction_db": p99, "pct_samples_over_1db": pct}
        x = ffmpeg(to_lufs(x, meter, chain["loudness_lufs"]), sr, chain["compressor"]["ffmpeg"])
        steps["compressor"] = chain["compressor"]["ffmpeg"]
    x = to_lufs(x, meter, chain["loudness_lufs"])
    peak, ceiling = np.max(np.abs(x)), 10 ** (chain["peak_ceiling_dbfs"] / 20)
    if peak > ceiling:  # plain gain, no limiter
        x = x * ceiling / peak
    steps["loudness_lufs"] = round(meter.integrated_loudness(x), 1)
    steps["peak_dbfs"] = round(20 * np.log10(np.max(np.abs(x))), 1)
    steps["floor_dbfs_before"] = round(floor_before, 1)
    steps["floor_dbfs_after"] = round(noise_sample(x)[1], 1)
    return x, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", default=str(ROOT / "data/takes"))
    ap.add_argument("--chain", default=str(ROOT / "config/podcast-chain.json"))
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    ap.add_argument("--variant", choices=["podcast", "clean"], default="podcast")
    ap.add_argument("--ab", type=int, default=0, help="write the last N takes as raw/clean/podcast trios to <out>/ab")
    a = ap.parse_args()

    src = Path(a.takes)
    chain = json.loads(Path(a.chain).read_text())
    rows = [json.loads(l) for l in open(src / "metadata.jsonl") if l.strip()]

    if a.ab:
        out = Path(a.out) / "ab"
        out.mkdir(parents=True, exist_ok=True)
        for r in sorted(rows, key=lambda r: r["id"])[-a.ab:]:
            x, sr = sf.read(src / r["file_name"], dtype="float64")
            meter = pyloudnorm.Meter(sr)
            # Raw is loudness-matched too, so the comparison is about tone and noise, not level.
            sf.write(out / f"{r['id']}_a_raw.wav", to_lufs(x, meter, chain["loudness_lufs"]), sr, subtype="PCM_24")
            for tag, v in (("b", "clean"), ("c", "podcast")):
                y, _ = process(x, sr, chain, v, meter)
                sf.write(out / f"{r['id']}_{tag}_{v}.wav", y, sr, subtype="PCM_24")
        print(f"A/B trios in {out}")
        return

    out = Path(a.out) / a.variant
    (out / "audio").mkdir(parents=True, exist_ok=True)
    meta = []
    for r in rows:
        x, sr = sf.read(src / r["file_name"], dtype="float64")
        y, st = process(x, sr, chain, a.variant, pyloudnorm.Meter(sr))
        sf.write(out / r["file_name"], y, sr, subtype="PCM_24")
        meta.append({**r, "processing": st})
    (out / "metadata.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta))
    med = lambda k: np.median([m["processing"][k] for m in meta])
    summary = {"takes": len(meta), "out": str(out), "floor_before": med("floor_dbfs_before"),
               "floor_after": med("floor_dbfs_after"), "peak_dbfs_median": med("peak_dbfs")}
    if a.variant == "podcast":
        summary["deess_p99_db_median"] = float(np.median([m["processing"]["deesser"]["p99_reduction_db"] for m in meta]))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
