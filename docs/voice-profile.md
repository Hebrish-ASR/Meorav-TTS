# Voice profile and cleanup at prep

Started 2026-10-21. Daniel asked to document his voice (pitch, frequency balance, noise floor),
derive a personal EQ for podcast and narration use, and train on a processed copy rather than the
raw takes. **Training uses the podcast chain** (below); raw takes stay untouched in the dataset. Current numbers: `docs/voice-profile/profile.md` (+ `profile.json`, `ltas.png`).

## Running it

```bash
# 1. Local copy of the HF dataset (gitignored; never commit audio): data/takes/{metadata.jsonl,audio/}
python3 scripts/fetch_dataset.py train data/takes   # or: training/run.sh fetch (train + test)
# 2. Profile (writes docs/voice-profile/), then the cleaned copies (data/processed/, gitignored)
uv run scripts/voice_profile.py                 # --max-floor-dbfs -57 by default: quiet-room takes only
uv run scripts/prep_clean.py                    # podcast chain (training default); --variant clean, --ab 6
```

Only takes already published to the HF dataset are in this copy.

## What the profile measures

| Figure | How |
|---|---|
| Pitch | Praat autocorrelation (parselmouth), 60–350 Hz search range, voiced frames only; 5th / 50th / 95th percentile |
| Formants F1–F3 | Praat Burg, 5 formants up to 5 kHz (male setting), medians over voiced frames. Depend on the prompts' vowel mix: track them over time, not against other people |
| Long-term spectrum (LTAS) | 4096-point frames (85 ms) at 48 kHz. Speech frames = within 20 dB of the take's 90th-percentile level; noise frames = within 3 dB of its 10th percentile. Mean noise power is subtracted from mean speech power, then summed into one-third-octave bands, 50 Hz–20 kHz |
| Reference | **ANSI S3.5-1997 Table 3**, normal-effort speech spectrum level, 160 Hz–8 kHz, converted to band levels (+10·log10(0.2316·fc)) and level-aligned to the voice over that range, so only the shape is compared. Values checked 2026-10-21 against MoSQITo's copy (`mosqito/sq_metrics/speech_intelligibility/sii_ansi/_speech_data.py`). It is an average talker in a free field, not a podcast target: a deviation describes voice + mic + room |
| SNR per band | voice band minus noise band. Decides whether a boost is safe |
| Hum | noise power at 50/100/150 Hz against neighbouring bins (Israeli mains is 50 Hz) |
| Loudness | BS.1770 integrated LUFS per take (pyloudnorm) |

## First profile (2026-10-21, 25 quiet-room takes, 85 s of speech)

- Pitch median **105 Hz**, 5th–95th percentile 78–185 Hz (15 semitones: a lively range, which is
  the Irish cadence the clone has to keep).
- Spectrum within ±3 dB of average speech from 160 Hz to 2.5 kHz. Distinctive: a **dip at
  3.15 kHz (−6 dB)** and **sibilance at 5–6.3 kHz (+2 to +3 dB, peak 5.3 kHz)**. Mic, room and
  voice cannot be separated from one profile; re-run after the studio mic arrives.
- Room: SNR 35–53 dB from 80 Hz up; a small 50 Hz hum (+7 dB over neighbouring bins, but the
  50 Hz band is still about 60 dB below the speech peak).
- Diagnostic EQ from the profile: high-pass 60 Hz, +1.4 dB at 1 kHz, +1.9 dB at 3.15 kHz, +1.5 dB
  shelf at 10 kHz. Not used for training (see the podcast chain).

## Training data: the podcast chain (decided 2026-10-21)

Daniel: train on the podcast-processed voice, "a little bit of de-essing, compressing,
normalization". The chain is `config/podcast-chain.json`, run by `scripts/prep_clean.py`
(default variant `podcast`):

1. **Denoise**: stationary spectral gating (`noisereduce`), the take's own quietest frames as the
   noise profile, `prop_decrease` 0.6 ≈ 8 dB. Partial on purpose: full gating leaves "musical
   noise" the model would learn.
2. **EQ**: Daniel's own voice-prompt EQ from the podcast pipeline, unchanged
   (the voice EQ preset of his podcast production pipeline): high-pass 80 Hz, −4 dB at
   150 Hz, −3 dB at 350 Hz, +3 dB at 3 kHz, +2 dB at 6 kHz. Same ffmpeg filter strings.
3. **De-esser**: split-band and dynamic, 4.5–9 kHz, threshold −12 dB band-over-full, 4:1, at most
   8 dB. Written here (not ffmpeg `deesser`, which has no frequency setting) so it can sit on the
   measured 5.3 kHz peak.
4. **Compressor**: the podcast's dialogue-bus setting, `acompressor` 2:1 at −20 dB, 8/120 ms,
   +1 dB makeup, applied after normalising each take so the threshold means the same everywhere.
5. **Loudness**: −20 LUFS per take, −1 dBFS sample-peak ceiling by gain (no limiter).

Output stays 48 kHz / 24-bit; the 24 kHz downsample is the training prep's job
(`docs/training-plan.md`). Variant `clean` (denoise + high-pass + loudness only) is kept as the
A/B baseline; `--ab 6` writes loudness-matched raw / clean / podcast trios.

### Measured effect (first run, 31 takes; `docs/voice-profile/podcast-output/`)

- Noise floor: median −62.5 → −66 dBFS. Higher than `clean` (−69.6) because compression and the
  presence boost lift quiet passages: expected, and still 40+ dB under the voice above 80 Hz.
- Peaks median −2.3 dBFS; every take at −20 LUFS.
- Spectrum vs average speech: 2–2.5 kHz +4 dB and 4–6.3 kHz +4.6 to +6.1 dB (raw was −1 to +3).
  This is the intended podcast brightness, but it is bright. The de-esser reaches its 8 dB cap
  on the most sibilant 1% of samples in a median take.
- **Alternative tried and not adopted:** replacing the +2 dB at 6 kHz with the dialogue bus's
  `highshelf=f=9000:g=1.5` lowers 5–6.3 kHz by only ~1 dB; most of the lift comes from the 3 kHz
  presence boost, the low cuts and compression together. Not worth departing from the trusted
  preset on measurement alone. If listening finds the clone hissy on "s", try that swap first,
  then a lower de-esser threshold.

## Diagnostic EQ rules (`suggest_eq` in `scripts/voice_profile.py`)

The profile still prints a heuristic EQ. It is **diagnostic only**: training uses the podcast
chain above. Rules, chosen 2026-10-21:

1. High-pass at 0.8 × the 5th-percentile F0, clamped to 60–100 Hz: below the voice, so it removes
   rumble and the air conditioner's low end without thinning the voice.
2. Deviation from ANSI, 3-point smoothed; peaks at local extremes of at least 2 dB, at least 2/3
   octave apart, at most 4. Gain = **half** the deviation, capped at −4 / +3 dB, Q 1.4.
3. No boost where band SNR is under 30 dB (it would boost the noise), and no boost at 5–9 kHz
   (next to the sibilance peak; the first run suggested +1.2 dB at 8 kHz, which this rule drops).
4. Air shelf (+1.5 dB at 10 kHz) only if SNR at 10 and 12.5 kHz is 35 dB or more.
