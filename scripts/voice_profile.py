"""Profile Daniel's voice from the capture-app takes: pitch, formants, long-term spectrum, noise
floor, loudness, and a suggested personal EQ.

    uv run scripts/voice_profile.py                       # takes in data/takes, quiet-room only
    uv run scripts/voice_profile.py --max-floor-dbfs 0    # every take

Input: an HF "audiofolder" copy of the dataset (`metadata.jsonl` + `audio/*.wav`); fetch it with
`python3 scripts/fetch_dataset.py train data/takes` (docs/voice-profile.md). Output (committed; no audio): `docs/voice-profile/
profile.json` (machine-readable), `profile.md`, `ltas.png`. Its EQ suggestion is diagnostic;
training uses config/podcast-chain.json.

Method and the reasoning behind each threshold: docs/voice-profile.md.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import parselmouth
import pyloudnorm
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent

# ANSI S3.5-1997 Table 3: standard speech spectrum LEVEL (dB per Hz) for normal vocal effort, one-
# third-octave bands 160 Hz-8 kHz. Checked 2026-10-21 against MoSQITo's copy
# (mosqito/sq_metrics/speech_intelligibility/sii_ansi/_speech_data.py). The reference is an average
# talker in free field: deviations from it describe this voice + mic + room, not a fault.
ANSI_FC = [160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
ANSI_NORMAL = [32.41, 34.48, 34.75, 33.98, 34.59, 34.27, 32.06, 28.30, 25.01, 23.00, 20.15, 17.32, 13.18,
               11.55, 9.33, 5.31, 2.59, 1.13]
FC = [50, 63, 80, 100, 125] + ANSI_FC + [10000, 12500, 16000, 20000]

NFFT, HOP = 4096, 1024  # 85 ms frames at 48 kHz: ~12 Hz bins, enough for the 50-125 Hz bands


def frames(x, sr):
    win = np.hanning(NFFT)
    n = 1 + max(0, (len(x) - NFFT) // HOP)
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    seg = x[idx] * win
    power = np.abs(np.fft.rfft(seg, axis=1)) ** 2 / np.sum(win**2)
    rms_db = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-20)
    return power, rms_db


def bands(power_mean, sr):
    f = np.fft.rfftfreq(NFFT, 1 / sr)
    out = []
    for fc in FC:
        lo, hi = fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6)
        m = (f >= lo) & (f < hi)
        out.append(10 * np.log10(power_mean[m].sum() + 1e-20) if m.any() and hi < sr / 2 else None)
    return out


def classify(rms_db):
    """Speech = within 20 dB of the loud frames; noise = the quietest frames. Both per take."""
    p10, p90 = np.percentile(rms_db, [10, 90])
    speech = rms_db > max(p90 - 20, p10 + 12)
    noise = rms_db <= p10 + 3
    return speech, noise


def pitch_formants(x, sr):
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=350)
    f0 = pitch.selected_array["frequency"]
    t = pitch.xs()
    voiced = f0 > 0
    form = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5000)
    fm = [[form.get_value_at_time(k, ti) for ti in t[voiced]] for k in (1, 2, 3)]
    return f0[voiced], fm


def pct(a, q):
    a = np.asarray([v for v in a if v == v])  # drop NaN
    return round(float(np.percentile(a, q)), 1) if len(a) else None


def suggest_eq(dev, snr, f0_p5):
    """Gentle corrective EQ: half the smoothed deviation from ANSI speech, capped, at most 4 peaks.
    Rules are in docs/voice-profile.md; they are heuristics, to be confirmed by A/B listening."""
    eq = [{"type": "highpass", "f": int(np.clip(5 * round(0.8 * f0_p5 / 5), 60, 100)), "order": 4,
           "why": "below the lowest voiced pitch (0.8 x F0 5th percentile): rumble, AC, handling"}]
    ansi_i = [FC.index(f) for f in ANSI_FC]
    d = np.array([dev[i] for i in ansi_i])
    sm = np.convolve(np.pad(d, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    cands = []
    for j in range(len(sm)):
        left, right = sm[max(j - 1, 0)], sm[min(j + 1, len(sm) - 1)]
        is_ext = (sm[j] >= left and sm[j] >= right) or (sm[j] <= left and sm[j] <= right)
        if is_ext and abs(sm[j]) >= 2.0:
            cands.append((abs(sm[j]), j))
    picked = []
    for _, j in sorted(cands, reverse=True):
        if len(picked) == 4:
            break
        if all(abs(np.log2(ANSI_FC[j] / ANSI_FC[k])) >= 2 / 3 for k in picked):
            picked.append(j)
    for j in sorted(picked):
        gain = float(np.clip(-0.5 * sm[j], -4.0, 3.0))
        # Never boost a band where the room noise is within 30 dB of the voice: that boosts the noise.
        if gain > 0 and (snr[ansi_i[j]] or 0) < 30:
            continue
        # Nor in the sibilance region: a dip at 8 kHz next to a 5-6.3 kHz excess is not worth an "s" boost.
        if gain > 0 and 5000 <= ANSI_FC[j] <= 9000:
            continue
        eq.append({"type": "peak", "f": ANSI_FC[j], "gain_db": round(gain, 1), "q": 1.4,
                   "why": f"voice is {sm[j]:+.1f} dB vs ANSI S3.5 average speech here (smoothed)"})
    hi = [snr[FC.index(f)] for f in (10000, 12500) if snr[FC.index(f)] is not None]
    if hi and min(hi) >= 35:
        eq.append({"type": "highshelf", "f": 10000, "gain_db": 1.5, "q": 0.707,
                   "why": f"optional 'air' for podcast playback; SNR above 10 kHz is {min(hi):.0f} dB, so it lifts little noise"})
    return eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", default=str(ROOT / "data/takes"))
    ap.add_argument("--out", default=str(ROOT / "docs/voice-profile"))
    ap.add_argument("--max-floor-dbfs", type=float, default=-57.0,
                    help="only takes whose floor is at or below this (default: the quiet room from 2026-10-21 13:51)")
    a = ap.parse_args()
    src, out = Path(a.takes), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(src / "metadata.jsonl") if l.strip()]
    rows = [r for r in rows if (r.get("floor_dbfs") is None or r["floor_dbfs"] <= a.max_floor_dbfs)]
    if not rows:
        raise SystemExit("no takes pass the floor filter")

    sp_sum = nz_sum = None
    n_sp = n_nz = 0
    f0_all, fm_all, lufs, sr0 = [], [[], [], []], [], None
    meter = None
    for r in rows:
        x, sr = sf.read(src / r["file_name"], dtype="float64")
        x = x if x.ndim == 1 else x.mean(axis=1)
        sr0 = sr0 or sr
        assert sr == sr0, "mixed sample rates"
        power, rms_db = frames(x, sr)
        speech, noise = classify(rms_db)
        sp_sum = power[speech].sum(0) + (0 if sp_sum is None else sp_sum)
        nz_sum = power[noise].sum(0) + (0 if nz_sum is None else nz_sum)
        n_sp += speech.sum()
        n_nz += noise.sum()
        f0, fm = pitch_formants(x, sr)
        f0_all.extend(f0)
        for k in range(3):
            fm_all[k].extend(fm[k])
        meter = meter or pyloudnorm.Meter(sr)
        lufs.append(meter.integrated_loudness(x))

    sp_mean, nz_mean = sp_sum / n_sp, nz_sum / n_nz
    voice_b = bands(np.maximum(sp_mean - nz_mean, 1e-20), sr0)  # noise power subtracted
    noise_b = bands(nz_mean, sr0)
    snr = [None if v is None or n is None else round(v - n, 1) for v, n in zip(voice_b, noise_b)]

    # ANSI spectrum level -> one-third-octave band level, then align the two overall levels
    # over 160 Hz-8 kHz so only the SHAPE is compared.
    ansi_band = [s + 10 * np.log10(0.2316 * f) for s, f in zip(ANSI_NORMAL, ANSI_FC)]
    v_ansi = [voice_b[FC.index(f)] for f in ANSI_FC]
    tot = lambda db: 10 * np.log10(np.sum(10 ** (np.array(db) / 10)))
    shift = tot(ansi_band) - tot(v_ansi)
    aligned = [None if v is None else round(v + shift, 1) for v in voice_b]
    noise_al = [None if v is None else round(v + shift, 1) for v in noise_b]
    ref = [round(ansi_band[ANSI_FC.index(f)], 1) if f in ANSI_FC else None for f in FC]
    dev = [None if r is None or v is None else round(v - r, 1) for v, r in zip(aligned, ref)]

    f0 = np.array(f0_all)
    f0_p5, f0_med, f0_p95 = (float(np.percentile(f0, q)) for q in (5, 50, 95))
    f = np.fft.rfftfreq(NFFT, 1 / sr0)
    voice_spec = np.maximum(sp_mean - nz_mean, 1e-20)
    m = (f >= 80) & (f <= 12000)
    centroid = float(np.sum(f[m] * voice_spec[m]) / np.sum(voice_spec[m]))
    mt = (f >= 250) & (f <= 8000)
    tilt = float(np.polyfit(np.log2(f[mt]), 10 * np.log10(voice_spec[mt]), 1)[0])
    ms = (f >= 4000) & (f <= 10000)
    sib_peak = float(f[ms][np.argmax(voice_spec[ms])])
    band_e = lambda lo, hi: float(voice_spec[(f >= lo) & (f < hi)].sum())
    sib_ratio = 10 * np.log10(band_e(5000, 9000) / band_e(1000, 4000))
    hum = {}
    for h in (50, 100, 150):
        i = int(round(h / (sr0 / NFFT)))
        local = np.median(nz_mean[max(i - 8, 1):i + 9])
        hum[f"{h}_hz_over_local_db"] = round(float(10 * np.log10(nz_mean[i] / local)), 1)

    profile = {
        "generated": dt.date.today().isoformat(),
        "source": {"takes": len(rows), "speech_seconds": round(n_sp * HOP / sr0, 1),
                   "noise_seconds": round(n_nz * HOP / sr0, 1), "sample_rate": sr0,
                   "filter": f"floor_dbfs <= {a.max_floor_dbfs}", "take_ids": [r["id"] for r in rows],
                   "microphone": sorted({r.get("microphone") or "unset" for r in rows}),
                   "room": sorted({r.get("room") or "unset" for r in rows})},
        "pitch_hz": {"p5": round(f0_p5, 1), "median": round(f0_med, 1), "p95": round(f0_p95, 1),
                     "range_semitones_p5_p95": round(12 * np.log2(f0_p95 / f0_p5), 1)},
        "formants_hz_median": {"F1": pct(fm_all[0], 50), "F2": pct(fm_all[1], 50), "F3": pct(fm_all[2], 50),
                               "note": "long-term medians over voiced frames; depend on the vowel mix of these prompts"},
        "spectrum": {"centroid_hz": round(centroid), "tilt_db_per_octave_250_8k": round(tilt, 1),
                     "sibilance_peak_hz": round(sib_peak), "sibilance_5_9k_vs_1_4k_db": round(sib_ratio, 1)},
        "loudness_lufs": {"median": round(float(np.median(lufs)), 1), "min": round(min(lufs), 1), "max": round(max(lufs), 1)},
        "noise": {"hum": hum},
        "third_octave": [{"fc": fc, "voice_db": v, "ansi_db": r, "dev_db": d, "noise_db": n, "snr_db": s}
                         for fc, v, r, d, n, s in zip(FC, aligned, ref, dev, noise_al, snr)],
        "eq_suggestion": suggest_eq(dev, snr, f0_p5),
    }
    (out / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    write_md(profile, out / "profile.md")
    plot(profile, out / "ltas.png")
    print(json.dumps({k: profile[k] for k in ("pitch_hz", "formants_hz_median", "spectrum", "loudness_lufs", "noise", "eq_suggestion")}, indent=1))


def write_md(p, path):
    s = p["source"]
    L = [f"# Voice profile ({p['generated']})", "",
         f"Generated by `scripts/voice_profile.py` from {s['takes']} takes ({s['speech_seconds']} s of speech, "
         f"{s['noise_seconds']} s of room noise), filter `{s['filter']}`. Method: `docs/voice-profile.md`. "
         "Machine-readable copy: `profile.json`.", "",
         "| | |", "|---|---|",
         f"| Pitch (F0) | median {p['pitch_hz']['median']} Hz, 5th-95th percentile {p['pitch_hz']['p5']}-{p['pitch_hz']['p95']} Hz ({p['pitch_hz']['range_semitones_p5_p95']} semitones) |",
         f"| Formants (median) | F1 {p['formants_hz_median']['F1']}, F2 {p['formants_hz_median']['F2']}, F3 {p['formants_hz_median']['F3']} Hz |",
         f"| Spectral centroid | {p['spectrum']['centroid_hz']} Hz |",
         f"| Spectral tilt 250 Hz-8 kHz | {p['spectrum']['tilt_db_per_octave_250_8k']} dB/octave |",
         f"| Sibilance | peak at {p['spectrum']['sibilance_peak_hz']} Hz; 5-9 kHz is {p['spectrum']['sibilance_5_9k_vs_1_4k_db']} dB vs 1-4 kHz |",
         f"| Loudness per take | median {p['loudness_lufs']['median']} LUFS ({p['loudness_lufs']['min']} to {p['loudness_lufs']['max']}) |",
         f"| Mains hum in the noise floor | " + ", ".join(f"{k.split('_')[0]} Hz {v:+} dB" for k, v in p['noise']['hum'].items()) + " over neighbouring bins |",
         "", "## One-third-octave spectrum", "",
         "Voice level-aligned to ANSI S3.5-1997 normal-effort speech over 160 Hz-8 kHz; `dev` = voice minus reference.", "",
         "| Band (Hz) | Voice dB | ANSI dB | dev | Noise dB | SNR |", "|---:|---:|---:|---:|---:|---:|"]
    for b in p["third_octave"]:
        c = lambda v: "" if v is None else f"{v}"
        L.append(f"| {b['fc']} | {c(b['voice_db'])} | {c(b['ansi_db'])} | {c(b['dev_db'])} | {c(b['noise_db'])} | {c(b['snr_db'])} |")
    L += ["", "## Suggested EQ (heuristic; confirm by A/B)", "", "| Type | Freq | Gain | Q | Why |", "|---|---:|---:|---:|---|"]
    for e in p["eq_suggestion"]:
        L.append(f"| {e['type']} | {e['f']} Hz | {e.get('gain_db', '')} | {e.get('q', '')}{' (order %d)' % e['order'] if 'order' in e else ''} | {e['why']} |")
    L += ["", "![LTAS](ltas.png)", ""]
    path.write_text("\n".join(L))


def plot(p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = p["third_octave"]
    fc = [x["fc"] for x in b]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].semilogx(fc, [x["voice_db"] for x in b], "o-", label="Daniel (speech, noise subtracted)")
    ax[0].semilogx([x["fc"] for x in b if x["ansi_db"]], [x["ansi_db"] for x in b if x["ansi_db"]], "s--", label="ANSI S3.5 normal speech (shape)")
    ax[0].semilogx(fc, [x["noise_db"] for x in b], "^:", label="room noise floor")
    ax[0].set_ylabel("band level, dB (aligned)")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].set_title(f"Long-term average spectrum, {p['source']['takes']} takes, {p['generated']}")
    dv = [(x["fc"], x["dev_db"]) for x in b if x["dev_db"] is not None]
    ax[1].semilogx([d[0] for d in dv], [d[1] for d in dv], "o-")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("voice - ANSI, dB")
    ax[1].set_xlabel("Hz")
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
