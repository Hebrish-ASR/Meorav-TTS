# Recording spec

Decided 2026-10-21 (Daniel: "record clean"). Sources and the reasoning are in
`docs/model-selection.md`; the research behind it was done the same day.

## Capture (enforced by the app)

| | Value | Where |
|---|---|---|
| Format | WAV, PCM, mono, **48 kHz, 24-bit** | `recorder.js` |
| Path | AudioWorklet raw-PCM tap. Never MediaRecorder: Chrome gives Opus, which is lossy, and the model would learn its artefacts. | `recorder.js` |
| Browser processing | `echoCancellation`, `noiseSuppression`, `autoGainControl` all requested **off**. Browsers treat these as preferences, so what was actually applied (`track.getSettings()`) is saved with every take (`input.*`, dataset column `browser_processing`), and Settings shows "processing ON (bad)" if a device ignores the request. | `recorder.js`, Settings |
| Context rate | `AudioContext({sampleRate: 48000})`; if refused, the device rate, resampled offline at stop (`native_rate` records which). | `recorder.js` |
| Max length | 26 s (the upload endpoint's 4.5 MB request-body limit) | `MAX_S` |
| Trim | Fixed guards (first 0.10 s, last 0.05 s: the tap transients), plus a 0.4 s tail hold after the stop tap, plus leading silence over 0.6 s cut to 0.3 s. Nothing inside a take is ever cut. Final padding to ~0.15 s each side happens in training prep. | `recorder.js` |

## Interim conditions (Daniel, 2026-10-21)

No quiet room yet: an air conditioner and a white noise machine run in the recording room, and
the home studio is not set up. Decision: **record now with what is available**, since steady
broadband noise is the most benign kind for TTS (the model learns voice, accent and rhythm from
speech that sits well above it; the worst likely effect is a faint hiss in the output). Intermittent
noise (voices, doors, the baby, traffic) still means a retake. When the studio exists, record a
cleaner batch and fine-tune again; nothing recorded now is wasted, and `room` / `floor_dbfs` per row
let training filter or weight by condition.

Measured on the first six takes (2026-10-21): peaks −1.8 to −7.1 dBFS, no clipping, 48 kHz native,
browser processing off. Floor −39 to −41 dBFS on takes 1–3 (SNR 20–23 dB), then −50 to −55 dBFS on
takes 4–6 (SNR 30–39 dB) after a change in the room.

## Per-take checks (shown after each take, never blocking)

Computed in `quality()` in `recorder.js` from 20 ms RMS windows and saved in the record:

| Figure | Target | Warning shown when |
|---|---|---|
| Peak | −6 to −3 dBFS (meter brackets: −18 to −6) | clipped (any sample ≥ 0.99), or below −30 dBFS |
| Noise floor (10th-percentile window) | below −60 dBFS ideal; −40 accepted for now | never warned since release 3; shown as a figure on the quality line |
| SNR estimate (90th − 10th percentile) | > 40 dB | shown, not judged |
| Edges (peak in first/last 120 ms) | quiet | above −20 dBFS and 15 dB over the floor: the take probably starts or ends mid-word (one warning at a time, release 3) |

## Session rules (Settings → Recording tips)

- **One mic, one room, one distance for the whole dataset.** Consistency matters more than the
  mic itself. The 2026-10-15 dataset mixed phone and imported takes; that is what to avoid.
  A USB mic on the desktop is preferred over the phone; whichever is chosen, keep it.
- Soft furnishings around the mic; fans and fridge off where possible.
- Speak in the normal conversational voice, not a "reading voice": Irish cadence is the whole point.
- 20–30 minute sessions. A tired voice is a different voice.

## Not done, deliberately

- **No ML enhancement** of takes, now or at prep (profile-based gentle reduction is the exception, below) (Daniel, 2026-10-21: "let's just record
  clean"). Speech enhancers (Resemble Enhance, AudioSR bandwidth extension, DeepFilterNet) are
  trained mostly on generic speech and can shift timbre and vowels, which a fine-tune would learn.
  If a take is noisy, retake it. The one place enhancement may be tried is the old 16 kHz ASR
  takes as optional warm-up data (`docs/training-plan.md`), never the final mix.
- Loudness normalisation, EQ, de-essing and compression are prep steps (the podcast chain,
  `docs/voice-profile.md`), never capture steps.

## Profile-based noise reduction at prep (Daniel asked 2026-10-21)

Implemented 2026-10-21 in `scripts/prep_clean.py`; method, first results and the personal EQ: `docs/voice-profile.md`.

Not the ML enhancers ruled out above. A **stationary spectral gate**: estimate the room's noise
spectrum from the non-speech parts of each session's takes (the ~0.3 s kept before speech, and a
planned 10 s "Room tone" capture per session), then subtract it per frequency bin across the clip
(`noisereduce` stationary mode, or Audacity-style spectral subtraction). Suited to the interim room:
an air conditioner and a white noise machine are steady, so the profile holds for the whole take.

Rules:
- **Raw takes are never modified**; reduction is a prep option, so it can be changed or dropped.
- **Partial reduction only** (about 6–10 dB, not full gating): heavy subtraction leaves "musical
  noise", an artefact the model would learn. Noise under the voice's own frequencies cannot be
  separated; white noise covers the whole band, so some always remains.
- **Decided by A/B listening** at the first probe run: same takes, raw vs. reduced, judged on
  voice and accent, not on how clean the output sounds.
