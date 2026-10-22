# How much data: milestones

Written 2026-10-21 from published reports (sources below), before any of this project's own runs. Minutes are
**recorded** audio (raw takes); training keeps ~90% after trimming, and at ~4 s per kept clip 1 h kept is
~900 clips. At the measured pace (0.33 min of audio per minute at the mic, `docs/estimates.md`) each
recorded minute costs ~3 minutes at the mic. The capture app shows progress against these milestones.

**The short version:** a fine-tune learns *who* is speaking fast (minutes) and *how they pronounce*
slowly. Nothing published shows a Qwen3-TTS fine-tune holding a non-US English accent at any amount of
data, so every accent threshold here is inference, and the ~1 h probe is the real test. The closest
direct evidence is non-Qwen: an Amazon study modelled an Irish accent from ~2,000 utterances of one
Irish speaker, rated by Irish listeners.

| Milestone | Recorded | Kept / clips | What to expect | Confidence |
|---|---|---|---|---|
| **Smoke test** | **15 min** (reached 2026-10-21) | ~14 min / ~210 | Pipeline runs, stops cleanly on familiar text, recognisably his voice | Voice identity: **high** (a fixed-script run on 12.6 min hit speaker similarity 0.98+). Unseen sentences: **coin-flip** (0–4 of 8 passed). Accent: not judgeable. 1–4 epochs only |
| **First listen** ("won't absolutely suck") | **38 min** | ~34 min / ~500 | Mostly intelligible, ends reliably, his timbre; may sound flat and US-leaning | **Moderate, ~60–70% (inference).** FastPitch quality plateaus at 30 min; 21- and 34-min Qwen runs were usable even with the buggy script |
| **Probe** (does the accent move?) | **68 min** | ~1 h / ~900 | Compare against zero-shot on Irish-marked words and intonation | Decent quality: **fairly strong** (a fixed-script run on 117 min Korean had 2% character error). Accent moving: **genuinely uncertain** |
| **Main run** ("good, accent held") | **165 min** (2.75 h) | ~2.5 h / ~2,200 | Where the Irish-accent study (2,000 utterances), ElevenLabs' "best" range (2–3 h) and the fixed Korean run converge | Identity and quality near-certain if the probe was clean; accent held is **unproven for LLM-style TTS**. Worth it only if the probe showed movement |
| **Ceiling** | **270 min** (4.5 h) | ~4 h | Diminishing returns; the Scottish Piper voice `alba` used ~4–5 h | **Low–moderate.** Qwen runs at 7–18 h kept a foreign accent anyway: past here change the mix or method, not the hours |

## What moves the odds besides minutes

- **Clip length** (biggest risk after the accent): the clips average ~4 s. Short-only training is linked to
  decay on long outputs and unstable openings (#39, #343). `training/prepare.py` now keeps 0.5 s of
  trailing room tone (was 0.15 s); still to try: joining consecutive takes into 10–20 s items, and more
  multi-sentence prompts (`long` set).
- **Free speech**: no ablation found; the "strongest lever on accent" claim in `docs/dataset.md` is
  plausible inference. Transcripts must be exact.
- **Variety over repetition** of prompts (#39); consistent mic, room and chain; one fixed reference clip.
- **Epochs at small data**: few (1–4 under ~30 min), a checkpoint every epoch, judged by ear; loss misleads
  (#39, #314). Overfitting shows as runaway length: listen to held-out sentences, not training ones.
- **Prove it beats zero-shot**: Baseten (1.5 h, LJSpeech) and #314 (7 h Japanese) did not. Blind A/B on the
  same sentences, ideally with an Irish listener.

## Evidence

"Buggy" = upstream `sft_12hz.py` (double label shift, no `text_projection`; `docs/training-plan.md`); those
results understate what works. "Fixed" = PR #278's fixes, which `training/sft_12hz_patched.py` carries.

| Model | Data | Outcome | Source |
|---|---|---|---|
| Qwen3-TTS 0.6B | 12.6 min Spanish | Fixed: target speaker on every take (cosine 0.979–0.995); unseen-text pass 0/8, 4/8, 1/8 at epochs 2/4/6, runaway by 6. Buggy: wrong voice | [PR #278](https://github.com/QwenLM/Qwen3-TTS/pull/278) |
| Qwen3-TTS | 117 min Korean, 880 utts, 5e-6 × 2 epochs | Fixed: all 44 test sentences end normally, CER 2.0%. Buggy: never stops | [PR #278](https://github.com/QwenLM/Qwen3-TTS/pull/278) |
| Qwen3-TTS | 9 min, 50 clips, 2e-6 | "Worked" on both sizes; needed 1 s trailing silence; likeness mediocre (buggy) | [#39](https://github.com/QwenLM/Qwen3-TTS/issues/39) |
| Qwen3-TTS LoRA | 21 min English, 20 epochs | Long-form "genuinely excellent"; ~15% of short utterances start in the wrong timbre (buggy) | [#343](https://github.com/QwenLM/Qwen3-TTS/issues/343) |
| Qwen3-TTS | 1.5 h LJSpeech | Did not beat zero-shot (a US voice the base already does) | [Baseten](https://www.baseten.co/blog/fine-tuning-qwen3-tts-for-high-quality-voice-cloning/) |
| Qwen3-TTS | 3 h African-accented English | Still American (buggy) | [#134](https://github.com/QwenLM/Qwen3-TTS/issues/134) |
| Qwen3-TTS | 7 h Bengali; 15–18 h Turkish | Hindi-accented; "like a Greek or American guy talking Turkish" (buggy) | [#323](https://github.com/QwenLM/Qwen3-TTS/issues/323), [#27](https://github.com/QwenLM/Qwen3-TTS/issues/27) |
| Glow-style TTS (Amazon) | ~2,000 utts, one Irish speaker | "Enough to model the accent", 24 Irish raters | [arXiv 2301.04606](https://arxiv.org/pdf/2301.04606) |
| XTTS-v2 vs CosyVoice2 | 0.8 h Irish | Irish accent MOS: zero-shot 2.09, fine-tuned 2.63, real 3.49 | [arXiv 2508.07426](https://arxiv.org/pdf/2508.07426) |
| FastPitch | 1 / 5 / 30 / 60 min | MOS 2.81 / 3.21 / 3.40 / 3.39 vs 3.44 at 27 h; ≤5 min speaks too fast | [arXiv 2110.05798](https://arxiv.org/pdf/2110.05798) |
| Piper `alba` | ~4–5 h Scottish, from a US checkpoint | Shipped as an official regional voice | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) |
| ElevenLabs PVC | 30 min minimum, 2–3 h best | Commercial guidance | ElevenLabs docs |

Reports are anecdotes unless a paper is cited. Replace this table's role with this project's own numbers as the
POC, probe and main run land (`docs/training-plan.md`, run log).
