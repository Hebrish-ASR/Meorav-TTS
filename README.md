# Meorav TTS

Meorav is a text-to-speech voice for code-switched English and Hebrew: Irish-accented English that
pronounces the Hebrew words in it correctly ("Yalla, tachles: Bituach Leumi wants the form by
Thursday"). It is trained on one speaker, Daniel Rosehill, and speaks in his voice.

This repository is the training pipeline: data prep, the fine-tuning code (Qwen3-TTS-12Hz-1.7B-Base,
full SFT on Modal) and the decisions behind it. It is the first version, shared as it was when the
first checkpoint was trained.

Meorav is part of [Hebrish](https://hebrish.app), Daniel's one-person not-for-profit project on
English–Hebrew code-switched speech. Its other model lines are speech-recognition models (Sabich,
Jachnun, Shakshuka); like them, Meorav is named after Israeli food. *Meorav* is Hebrew for "mixed", as in
*meorav Yerushalmi*, the Jerusalem mixed grill: a fitting name for speech that mixes two languages.

## Status

| Checkpoint | Date | Data | Published | Verdict |
| --- | --- | --- | --- | --- |
| **Meorav PoC-1** (run `first-test`) | 2026-10-22 | 657 read takes, 50 min after trimming | `Hebrish/TTS-V1` (removed from Hugging Face; 404 on 2026-10-27) | Ran end to end; loss 2.88 to about 1.6; all 27 listening-set samples stop cleanly at lengths near the real takes. Listening verdict pending |
| Meorav v1.0 | not yet | target 2.5–3 h, including free speech | will be published on Hugging Face under the Meorav name | |

PoC-1 settings: LR 7e-6, 3 epochs, batch 2 with gradient accumulation 4 (effective 8), one L40S on
Modal, final checkpoint only. Run log and settings for later runs: [`docs/training-plan.md`](docs/training-plan.md).

### Naming

Checkpoints before the first mature release are **Meorav PoC-1, PoC-2, …**, never a bare "v1"
(PoC-1's Hugging Face repo, `Hebrish/TTS-V1`, predates the name). The first mature release will be
**Meorav v1.0**, and the Hugging Face model takes the Meorav name then; later retrains are v1.1,
v1.2, and so on. The same convention is used for Hebrish's Jachnun and Shakshuka lines. "Meorav" is
provisional until v1.0.

## Links

| | |
| --- | --- |
| Dataset | [`Hebrish/Hebrish-TTS-Voice-Recordings`](https://huggingface.co/datasets/Hebrish/Hebrish-TTS-Voice-Recordings), CC BY-NC 4.0 |
| Model (PoC-1) | was `Hebrish/TTS-V1`, CC BY-NC 4.0; removed from Hugging Face (404 on 2026-10-27). Current versions: [`Hebrish/Meorav`](https://huggingface.co/Hebrish/Meorav) |
| Base model | [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base), Apache 2.0 |
| Project | [hebrish.app](https://hebrish.app) |

## Why this approach

Two earlier attempts at cloning this voice (an Orpheus fine-tune and a zero-shot clone) came out with
an American accent: zero-shot copies timbre but takes prosody from the base model. Meorav is a full
fine-tune of Qwen3-TTS, which reads raw text through its tokenizer with no phonemiser (espeak has no
Irish-English voice) and updates every weight, so the model can move off its US prior. Full reasoning,
the fallbacks (CosyVoice3, Chatterbox) and what was ruled out: [`docs/model-selection.md`](docs/model-selection.md).

No published source shows an open TTS fine-tune holding a non-US English accent, so the accent is the
open question this project is testing. Evaluation is by ear: held-out sentences and out-of-distribution
lines, base model zero-shot vs. fine-tune, with the real recordings as the anchor.

### Fixes to the upstream training script

Training runs [`training/sft_12hz_patched.py`](training/sft_12hz_patched.py), a modified copy of
Qwen3-TTS's `finetuning/sft_12hz.py` (Apache 2.0, Alibaba Qwen team; licence header kept). The fixes
are useful to anyone fine-tuning Qwen3-TTS, and none was merged upstream as of 2026-10-21:

1. Double label shift: the talker learned frame t+2 instead of t+1 (QwenLM/Qwen3-TTS#371, #179; as PR #278).
2. Sub-talker leak: codes were predicted from a hidden state that had already seen them (#337; PR #278).
3. Sub-talker loss shifted twice; computed here from the logits, so `qwen-tts` stays stock.
4. `text_projection` skipped in training but applied at inference (#120; PRs #188, #336).
5. bf16 weights with plain AdamW lose most updates at these learning rates: fp32 master weights and
   optimizer state with bf16 autocast (checkpoints still saved in bf16).
6. `--save_every N` for per-epoch checkpoints.
7. `Accelerator(log_with="tensorboard")` without a logging dir raises in accelerate 1.12; dropped.

Each change is marked `VC1026` in the code (the project's working name when they were written).
Details and the evidence for each: [`docs/training-plan.md`](docs/training-plan.md).

## Recordings

The takes come from a phone capture app that records raw 48 kHz / 24-bit PCM with browser voice
processing off. It also collects Hebrish's speech-recognition data, and the two need different
conditions. Speech recognition wants real ones: an air conditioner, a crying baby, people talking.
A voice clone wants a quiet room. So the app has an explicit **voice-clone mode**, and only takes
recorded in it, with no noise flags and passing the level checks (`tts_eligible`), go into Meorav's
training set. The rule is one-way: a quiet-room take is good speech-recognition data too, but a noisy
speech-recognition take never trains the voice.

## Pipeline

```
HF dataset (48 kHz / 24-bit WAV + metadata.jsonl)
  -> scripts/prep_clean.py      podcast chain: partial denoise, voice EQ, de-ess, 2:1 compression, -20 LUFS
  -> training/prepare.py        filter, trim (0.15 s head / 0.5 s tail), 48 -> 24 kHz, Qwen JSONL, one fixed reference take
  -> training/modal_app.py      upload + codec-tokenize (A10) -> train (L40S / A100-80GB) -> sample every checkpoint
  -> publish                    one checkpoint from the Modal volume to Hugging Face
```

`training/run.sh` runs it one stage per argument:

```bash
uv sync                       # local prep dependencies (needs ffmpeg on PATH)
training/run.sh local         # fetch + clean + prepare: no GPU, no cost
training/run.sh upload poc    # Modal: upload, tokenize, a short run, samples (~$1 on an L40S)
training/run.sh get           # samples -> data/samples/ to listen to
```

PoC-1 was trained with `modal run training/modal_app.py::poc --lr 7e-6 --epochs 3 --name first-test`.
Larger runs: `probe` at about 1 h of data, `sweep` (three learning rates) at 2.5–3 h. Publishing a
checkpoint: `HF_TOKEN=... modal run training/modal_app.py::publish --run <run> --epoch <n> --repo <org>/<name> [--public]`.

Requirements: a [Modal](https://modal.com) account with the `modal` CLI, [uv](https://docs.astral.sh/uv/),
and ffmpeg. The training image pins `qwen-tts==0.1.1`, torch 2.8.0 and Qwen3-TTS at commit `022e286b98`.
Costs per stage: [`docs/compute-and-cost.md`](docs/compute-and-cost.md).

| Variable | Default | Used by |
| --- | --- | --- |
| `MEORAV_DATASET` | `Hebrish/Hebrish-TTS-Voice-Recordings` | `run.sh fetch` |
| `MEORAV_MODAL_APP` | `meorav-tts` | Modal app and volume name |
| `MEORAV_SPEAKER` | `daniel` | custom-voice name stored in the checkpoint |
| `MODAL_PROFILE` | the CLI's active profile | Modal stages |
| `HF_TOKEN` | unset | `publish`; sent on fetch if set |

## Adapting it to your own voice

1. **Record.** Same format and layout as the Hebrish dataset: mono WAV at 48 kHz / 24-bit, no browser
   echo cancellation, noise suppression or auto-gain, one mic and one room throughout, conversational
   rather than reading voice. Numbers spelled out in the text as spoken. Spec:
   [`docs/recording-spec.md`](docs/recording-spec.md); schema and prompt sets: [`docs/dataset.md`](docs/dataset.md).
2. **Publish** the takes as a Hugging Face audiofolder (`data/train/`, `data/test/`, each with
   `audio/` and `metadata.jsonl`) and set `MEORAV_DATASET` to it, or place the files under
   `data/takes/` yourself.
3. **Replace the processing chain.** [`config/podcast-chain.json`](config/podcast-chain.json) is tuned to
   this voice (its EQ and de-esser band come from the measurements in
   [`docs/voice-profile.md`](docs/voice-profile.md)). Run `scripts/voice_profile.py` on your takes, then
   edit the chain, or use `scripts/prep_clean.py --variant clean` (denoise, high-pass, loudness only).
4. **Set your speaker name** (`MEORAV_SPEAKER`) and the out-of-distribution listening lines (`OOD` in
   `training/modal_app.py`), then run `local`, `upload` and `poc`.
5. **Judge by ear**, not loss. How much data buys what: [`docs/data-guide.md`](docs/data-guide.md).

### How the recordings are captured

Takes are recorded in a phone-friendly progressive web app that shows one prompt at a time, captures
raw PCM through an AudioWorklet (never MediaRecorder, whose Opus output is lossy), checks each take
for clipping, level and noise floor, and publishes accepted takes to the dataset. Recording is moving
to a unified recorder that records each take once at 48 kHz / 24-bit FLAC into one canonical corpus of
Daniel's read speech; the TTS dataset becomes a view of that corpus rather than a separate collection.
The capture app is not part of this repository.

## Repository

| Path | What |
| --- | --- |
| `training/modal_app.py` | Modal stages: upload, tokenize, train, synthesize, publish; run settings (`POC`, `PROBE`, `RUNS`) |
| `training/sft_12hz_patched.py` | Patched upstream training script (Apache 2.0) |
| `training/prepare.py` | Processed takes to Qwen3-TTS input |
| `training/run.sh` | The pipeline, one stage per argument |
| `scripts/prep_clean.py` | Processing chain (podcast or clean variant), A/B trios |
| `scripts/voice_profile.py` | Pitch, formants, long-term spectrum, noise, diagnostic EQ |
| `config/podcast-chain.json` | The chain training uses |
| `docs/` | Plan and run log, model choice, recording spec, dataset, data guide, voice profile, cost |

No audio, takes or reference clips are kept here; `data/` is gitignored.

## Licence

Code: MIT ([`LICENSE`](LICENSE)), except `training/sft_12hz_patched.py`, which is Apache 2.0 as a
modified copy of Qwen3-TTS code ([`NOTICE`](NOTICE)). The dataset and the published checkpoints are
CC BY-NC 4.0.
