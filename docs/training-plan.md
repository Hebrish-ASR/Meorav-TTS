# Training plan

Pipeline written 2026-10-21 (`training/`); first run on Modal 2026-10-22 (`first-test`, below). On Modal,
as for the Hebrish ASR fine-tunes.

## When to train

- **Proof of concept first** (Daniel, 2026-10-21: "target the minimum viable path in terms of training
  data and model… just to establish that it works"): `training/run.sh local upload poc` on whatever is
  recorded (14 min on 2026-10-21). 1.7B, not 0.6B: at this size the GPU cost differs by cents, and a
  failure on 0.6B would not say whether the 1.7B plan works. Success = the pipeline runs, the voice is
  his and the speech is intelligible; accent judgement waits for the probe. Reports of single-speaker
  fine-tunes on 9–21 min exist (QwenLM/Qwen3-TTS#39, #343), so it is not hopeless at 14 min.

- **Probe at ~1 h** of scripted takes: one short run, to prove the pipeline end to end and hear
  whether the accent moves at all. Exploratory runs are minutes-long probes; scale only on evidence.
- **Main run at 2.5–3 h**, including free speech (target ~30% of hours once the scripted base exists).

### Data on hand (2026-10-21, `training/run.sh local`)

| | Takes | Audio |
|---|---|---|
| HF `train` | 208 | 15.5 min recorded |
| Kept for training (after trim to 0.15 s each side) | 208 (none dropped) | **13.9 min = 0.23 h**, mean clip 4.0 s |
| HF `test` (listening test only) | 9 | 0.6 min |

Probe threshold 1 h: **46 min short** (~2.6 h at the mic; time and Modal cost per stage in `docs/estimates.md`). Main run: ~2.3 h short. `data/prepared/summary.json`
has the current figures after every `prepare`.

## Running it

`training/run.sh <stage>…` runs from the repo root, one stage per argument:

```bash
training/run.sh local     # = fetch clean prepare: no Modal, no cost
#   fetch    HF data/train -> data/takes/, data/test -> data/takes/test/ (new files only)
#   clean    scripts/prep_clean.py (podcast chain) on both -> data/processed/podcast/, data/processed/test/podcast/
#   prepare  training/prepare.py -> data/prepared/
training/run.sh upload    # data/prepared -> volume meorav-tts, fetch models, codec codes (A10)
training/run.sh probe     # at ~1 h: one run (7e-6 x 6, batch 4) + baseline/probe samples
training/run.sh sweep     # at 2.5-3 h: low / mid / high (A100-80GB, in parallel)
training/run.sh listen    # samples for every run on the volume (skips checkpoints already sampled)
training/run.sh get       # volume samples/ -> data/samples/ (gitignored)
```

Publishing one checkpoint, volume to HF with no local copy (private unless `--public`; needs `HF_TOKEN`
in the environment):

```bash
HF_TOKEN=... modal run training/modal_app.py::publish \
  --run first-test --epoch 2 --repo Hebrish/TTS-V1 --public
```

Checkpoints are numbered from 0: a 3-epoch run's last is `checkpoint-epoch-2`. Once a pushed
checkpoint is confirmed on HF (file list and sizes), delete it from the volume
(`modal volume rm -r meorav-tts runs/<run>/checkpoint-epoch-N`); `config.json` and
`train.log` stay.

The Modal stages are `modal run training/modal_app.py::<stage>` (set `MODAL_PROFILE` to pick a Modal profile);
`probe` takes `--lr --epochs --batch-size --name`, `listen` takes `--runs low,mid`. `modal` is the
CLI tool (client 1.5.2 at the time), not a project dependency. Upload is
`Volume.batch_upload` from the `upload` entrypoint, after deleting the volume's old `prepared/`, so a
take dropped locally cannot linger. `prepare.py` writes paths relative to `data/prepared/`;
`tokenize` rewrites them to `/vol/prepared/…` before upstream `prepare_data.py`.

## Prep (`training/prepare.py`)

1. Input is `scripts/prep_clean.py` output, variant `podcast` (decided 2026-10-21: denoise,
   Daniel's podcast EQ, de-ess, 2:1 compression, −20 LUFS; `docs/voice-profile.md`), never raw takes.
   Train split only for training; drop rows with `clipped_samples > 0`, `browser_processing = true`,
   `transcript_status = pending`, empty text, or a `prompt_id` that is in the test split. Each drop
   reason is printed with takes and minutes.
2. Trim to 0.15 s of silence before and 0.5 s after (tight tails taught rushing and run-on, #39) (first/last 20 ms frame 10 dB over the take's floor);
   downsample 48 → 24 kHz (`resample_poly`), 16-bit. Clips under 0.5 s after trim are dropped.
3. Write `train_raw.jsonl` `{"audio", "text", "ref_audio"}` with **one fixed `ref_audio`**, and
   `test.jsonl` (the held-out split, for `listen` only; real recordings as the anchor).
4. On Modal, `tokenize` runs upstream `prepare_data.py --tokenizer_model_path` (the
   `Qwen/Qwen3-TTS-Tokenizer-12Hz` snapshot on the volume) to add codec codes.

**Reference take.** Rule: unedited, 6–10 s after trim, set `conversational`, else `arctic`, else
`long`, else `tech` (style neutral / conversational / long), cleanest SNR, then nearest 8 s. No
conversational or arctic sentence is that long (longest 4.8 / 5.1 s), so on 2026-10-21 it fell
back to `long`: **`20261021T144348Z_dz7cvs`** (6.38 s), "The thing about running your own server at
home is that, when it breaks at eleven at night, there's nobody to ring except yourself." The pick
is sticky: later `prepare` runs keep it while it survives the filters (`--ref <id>` to force,
`--repick-ref` to choose again), because the speaker embedding comes from it.

## Run log

| Date | Run | Data | Settings | Result |
|---|---|---|---|---|
| 2026-10-22 | **Meorav PoC-1**: run `first-test` (`poc` entrypoint, `--lr 7e-6 --epochs 3 --name first-test`) | 657 takes, 50.0 min kept, mean 4.6 s | 7e-6 × 3 epochs, batch 2 (eff. 8), L40S, final checkpoint only | Ran end to end after fix 7. Loss 2.88 → ~1.6. All 27 samples stop cleanly, lengths near the real takes. Published **public** as [`Hebrish/TTS-V1`](https://huggingface.co/Hebrish/TTS-V1) (Daniel: an early look at what is possible, not the release); checkpoint then deleted from the volume. 8 comparison lines (fine-tuned / zero-shot / real) set aside for listening. Listening verdict pending |

## Upstream script bugs, patched (2026-10-21)

Training runs `training/sft_12hz_patched.py`, not upstream `finetuning/sft_12hz.py` at `022e286b98`.
Each change is marked `VC1026` in the file. Verified by reading upstream `sft_12hz.py`, `dataset.py`
and `modeling_qwen3_tts.py`; none of the fixes has been run yet.

1. **Double label shift.** `dataset.py` writes `codec_0_labels` aligned with the inputs (label at the
   position of the same frame). The script passed `inputs[:, :-1]` with `labels[:, 1:]`, and the
   talker's HF `ForCausalLMLoss` shifts once more, so it trained frame t+2. Measured in #371; likely
   the "speech gets faster each epoch" of #179. Fix as open PR #278.
2. **Sub-talker leak** (#337): codes 1–15 of frame k were predicted from the hidden state at frame
   k's own position, which already saw them. Now the previous position (PR #278).
3. **Sub-talker loss shift:** `forward_finetune`'s labels are aligned to its logits and the causal
   loss shifts them again. PR #278 patches the installed package; the copy here computes the cross
   entropy from the returned logits instead, so `qwen-tts` stays stock.
4. **`text_projection` skipped** in training but applied at inference (1.7B: same width, so it
   fails silently; 0.6B: crashes; #120, PRs #188/#336).
5. **bf16 weights + plain AdamW, no fp32 master.** At these LRs most Adam steps are under half a bf16
   ulp of the weight: on two real 1.7B layers, 95% / 90% / 62% of weights get no update at
   2e-6 / 7e-6 / 2e-5 (subagent measurement, 2026-10-21). Now fp32 weights and optimizer state with
   bf16 autocast; checkpoints are saved in bf16 (same size as before). Peak VRAM ~35 GB (estimate).

7. **`log_with="tensorboard"` without `logging_dir`**: accelerate 1.12 raises `ValueError: Logging with
   tensorboard requires a logging_dir` in `Accelerator()`, before any training (first Modal run,
   2026-10-22). No tracker is ever initialised, so the argument is dropped; loss goes to `train.log`.

Community code that already carries fixes 1 and 4: `instavar/qwen3-tts-lora-finetuning` (LoRA, plus an
experimental full mode) and `modelscope/ms-swift` (native `qwen3_tts` since PR #9638). Unsloth does
not support Qwen3-TTS (2026-10-21).

## Sweep (Qwen3-TTS-12Hz-1.7B-Base, full SFT, A100-80GB, fp32 master + bf16 autocast)

| Run | LR | Epochs | Batch (eff. ×4) | Updates at 2.5–3 h | Note |
|---|---|---|---|---|---|
| poc | 1e-5 | 4 (saves each) | 2 (8) | ~100 at 14 min | `poc` entrypoint, L40S; small data overfits by epoch ~6 |
| probe | 7e-6 | 6 | 4 (16) | ~340 at 1 h | `probe` entrypoint |
| low | 2e-6 | 10 | 8 (32) | ~700–850 | README's documented LR/epochs (it used batch 32) |
| mid | 7e-6 | 6 | 8 (32) | ~420–510 | |
| high | 2e-5 | 3 | 8 (32) | ~210–255 | README's script values; add warmup if unstable |

`sft_12hz.py` hard-codes gradient accumulation 4, so updates/epoch = clips / (4 × batch). One
batch size across the sweep keeps LR the only variable. Checkpoint every epoch (~4.5 GB each, 19
for the sweep ≈ 85 GB; prune after listening). Each run writes `runs/<name>/config.json` (LR,
batch, clips, updates/epoch, upstream commit, ref id) and `train.log`.

**Listening set** (`listen`): up to 20 test-split lines round-robin across prompt sets (9 exist
today) plus 7 out-of-distribution lines in `OOD` (plain, Hiberno-English, two Hebrish with the
bank's canonical romanisations, spelled-out numbers, a question, a long sentence), for every
checkpoint, plus `baseline` (base model zero-shot from `ref.wav`) and `real` (the processed test
recordings). Inference uses `generate_custom_voice(speaker="daniel", language="English")`.

### Upstream as checked (2026-10-21)

`QwenLM/Qwen3-TTS` `finetuning/` last changed in `022e286b98` (2026-03-17, "fix finetuning bug");
the image pins that commit. `qwen-tts` 0.1.1 is still the latest PyPI release (pins
transformers 4.57.3, accelerate 1.12.0); torch 2.8.0 as in the predecessor. CLI arguments of
`prepare_data.py` and `sft_12hz.py`, the checkpoint layout (`checkpoint-epoch-N`, speaker at
`spk_id` 3000) and the 24 kHz assert on `ref_audio` match what the predecessor assumed. The
`flash_attention_2` → `sdpa` patch is still needed; the image build now fails if the sed stops
matching. Upstream's own script is no longer what trains: see "Upstream script bugs, patched". Not yet verified by a real run: GPU memory at batch 8, epoch time,. `Accelerator(log_with="tensorboard")`
without a project dir was not harmless: it is fix 7 below.

## Release and naming

The model is published **publicly** under the `Hebrish` HF org, as a TTS model for code-switched
English–Hebrew ("Hebrish") with one voice, Daniel's. The model line is named **Meorav** (2026-10-22; Hebrew
"mixed", after *meorav Yerushalmi*; Hebrish names its model lines after Israeli food: Sabich, Jachnun, Shakshuka). Checkpoints before the
first mature release are **Meorav PoC-1, PoC-2, …**, never a bare "v1": PoC-1 is `first-test`,
published as `Hebrish/TTS-V1`. The first mature release goes to HF as **Meorav v1.0**, and the HF
repo takes the Meorav name then. Model cards link the dataset `Hebrish/Hebrish-TTS-Voice-Recordings`,
state the base model (Qwen3-TTS-12Hz-1.7B-Base, Apache 2.0) and the single speaker. PoC-1 is
CC BY-NC 4.0, matching the dataset, which is non-commercial because of the Leipzig news sentences.

## Optional experiment: old 16 kHz takes as warm-up

An older collection holds ~2.5 h of Daniel's phone takes (16 kHz, −41 dBFS
room tone). A two-stage run (old data first, then the 48 kHz set) might give the model more accent
evidence, at the risk of band-limited output. Only if the main run lacks accent. If tried, AudioSR
bandwidth extension on the *old* takes is the one place enhancement is allowed (`docs/recording-spec.md`).
