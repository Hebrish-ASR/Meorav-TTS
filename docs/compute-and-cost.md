# Compute and cost: recording time and training cost

Written 2026-10-21, before any training run. **Estimates, not measurements**: replace each row with
the real figure once a stage has run (`runs/<name>/train.log`, the Modal dashboard's usage page).
Prices are as checked on the date shown and will drift.

## Recording time

Measured from `recorded_at` on all 217 takes (two sessions, 2026-10-21): **0.33 min of audio per
minute at the mic** (48.8 min of sessions → 16.1 min of takes, mean clip 4.45 s). Trimming for
training keeps ~90% (15.5 → 13.9 min on the train split).

| Milestone | Training audio | Raw audio still needed | Mic time at 0.33 |
|---|---|---|---|
| Probe | 1 h | ~51 min | **~2.6 h** |
| Main run (low end) | 2.5 h | ~150 min | **~7.6 h** |
| Main run (high end) | 3 h | ~184 min | **~9.3 h** |

Levers: the `long` set (44–59 words) and free speech give more audio per prompt, so the ratio
rises as their share does; the `short` set lowers it. Free speech at ~30% of hours (the plan)
should bring the main run nearer 6–7 h at the mic. Re-measure after a few more sessions.

## Modal cost (personal workspace)

Prices from <https://modal.com/pricing> on 2026-10-21, per second, converted here (the page's
per-hour column was not used): A10 $0.000306/s = **$1.10/h**, L40S $0.000542/s = **$1.95/h**,
A100-80GB $0.000694/s = **$2.50/h**, H100 $0.001097/s = $3.95/h. CPU $0.047/core-h, memory
$0.008/GiB-h. Volumes $0.09/GiB-month with 1 TiB/month free.

Throughput is a guess until the probe runs: Qwen3-TTS-1.7B full SFT, ~4.5 s clips ≈ 60 frames at
12 Hz plus text, assumed **5–15 clips/s** on an A100-80GB at batch 8, plus ~5 min per run for image
start and model load and ~30–60 s per 4.5 GB checkpoint save.

| Stage | GPU | Work | Time (est.) | Cost (est.) |
|---|---|---|---|---|
| `upload` (models + codec codes) | A10 | 1 h data ≈ 800 clips; 3 h ≈ 2,400 | 10–20 min | $0.20–0.40 |
| `probe` at 1 h | A100-80GB | ~800 clips × 6 epochs | 15–30 min | $0.60–1.25 |
| `listen` after probe | L40S | 7 checkpoints + baseline, 27 lines each | 15–25 min | $0.50–0.80 |
| `sweep` low (2e-6 × 10) | A100-80GB | ~2,400 clips × 10 | 40–90 min | $1.70–3.75 |
| `sweep` mid (7e-6 × 6) | A100-80GB | × 6 | 25–55 min | $1.05–2.30 |
| `sweep` high (2e-5 × 3) | A100-80GB | × 3 | 15–30 min | $0.60–1.25 |
| `listen` after sweep | L40S | 19 checkpoints, 27 lines each | 40–70 min | $1.30–2.30 |
| CPU and memory on top | | ~32 GiB per container | | ~$1 |
| Volume | | base + tokenizer ~10 GB, checkpoints ~85 GB | | $0 (under 1 TiB free) |

**Probe cycle ≈ $1.50–2.50. Main cycle (upload + sweep + listen) ≈ $6–11. Both ≈ $8–14.**
The sweep runs its three trainings in parallel, so wall time is the low run's (~1–1.5 h), not
the sum. A re-run of one sweep point costs one row above.

What would break the estimate: an OOM at batch 8 (halve the batch: same epochs, ~same cost);
throughput under 5 clips/s (cost scales inversely); moving to H100 (~1.6× the price per hour,
usually ~2× the speed, so roughly the same per run). The optional two-stage run with the old
2.5 h of 16 kHz takes (`docs/training-plan.md`) would roughly double the sweep row.

Release (HF upload of one checkpoint, ~4.5 GB): free; Modal egress is $0.04/GiB if it is pushed
from the volume, ~$0.20.

## Cheaper options (checked 2026-10-21)

**VRAM floor.** The checkpoint has 1.93B parameters (talker 1.73B, of which the text embedding is
311M; code predictor 175M). With fp32 master weights and AdamW (the patched script,
`docs/training-plan.md`): **~35 GB** (estimate), so a 48 GB card. Upstream's unpatched script peaks
~20 GB only because it keeps everything in bf16, which loses most updates. Ways down, none built:
8-bit AdamW (~23 GB), plus a frozen text embedding (~20 GB, a 24 GB card), bf16 + torchao
stochastic-rounding AdamW8bit (~15 GB), LoRA r16 (~8 GB, changes what is tested).

| Option | Price | Per A100-equivalent hour* | Sweep (est.) | Notes |
|---|---|---|---|---|
| Modal A100-80GB (current) | $2.50/h | $2.50 | $3.40–7.30 | Pipeline written; per-second billing |
| Modal L40S 48GB | $1.95/h | ~$2.40 | about the same | No real saving |
| RunPod Community A100-80GB PCIe | $1.19/h | $1.19 | $1.60–3.50 | Same GPU, half the price |
| RunPod Community RTX A6000 / A40 48GB | $0.33 / $0.35/h | ~$0.75 | $1.00–2.20 | ~0.45× A100 speed; fits batch 8 |
| Vast.ai interruptible | often 50%+ under on-demand | lower still | | Pre-emptible: a run can die mid-epoch |
| Kaggle / Colab free (T4/P100 16 GB) | $0 | | | No bf16, 16 GB: full SFT does not fit |
| Local Radeon RX 7700-series (12–16 GB, ROCm; model not checked) | $0 | | | Does not fit; ROCm adds friction |

\*Price divided by rough speed relative to an A100, for this one job. RunPod prices from
<https://www.runpod.io/pricing> (Community Cloud, on-demand); Vast.ai publishes no fixed list. Community
Cloud runs on third-party hosts; that matters little here because the dataset is public anyway.

**Modal free credit.** Modal's Starter plan includes $30 of compute a month, so a small run early in
a month, before other GPU work, can fall inside the credit.

**Free training code.** The official script turned out to have training bugs, now patched here
(`docs/training-plan.md`). Low-VRAM community code exists: `instavar/qwen3-tts-lora-finetuning`
(PEFT LoRA, fixes the label shift; a T4 fits it), `modelscope/ms-swift` (native `qwen3_tts`),
`grimavatar/Qwen3-TTS-Train` (8-bit Adam, aimed at Colab T4, unproven). Unsloth does not support
Qwen3-TTS. Free T4s (Kaggle, Colab) lack bf16, so fp16 overflow is a further risk. LoRA stays the
fallback: it changes what the experiment tests.

**Bigger lever than the provider:** run only what the probe justifies. If the probe already sounds
right, one main run at the probe's LR instead of three cuts the sweep row by ~⅔ on any provider.

**Hugging Face PRO (checked 2026-10-21).** PRO adds ZeroGPU quota (25 min/day of
an H200 slice, per the Hub docs; beyond that $1 per 10 min from prepaid credits) and $2/month of
inference credits. HF Jobs / dedicated hardware cost the same as Modal (A100 80 GB $2.50/h,
L40S $1.80/h), so no saving there. ZeroGPU is built for Spaces inference: each `@spaces.GPU` call
is short and the GPU state does not survive between calls, so a full SFT would have to move
~30 GB of weights and optimizer state on and off the GPU in every chunk. Possible in principle
(25 min/day ≈ a probe a day), fragile in practice; not planned. **Where it fits: `listen`**
(inference only, ~$2–3 of each cycle on Modal) and the public demo Space at release, both free
within the daily quota.

## More providers and free credit (checked 2026-10-21)

Figures from the provider's own page unless marked (3p). Prices move, especially Vast.

| Provider | Free credit | Useful GPU, price | Catch |
|---|---|---|---|
| Lightning AI | One-time 30 credits (5 on sign-up + 25 for adding a card), 12 months; was 15/month until ~2026-09-21 (3p) | A100 40/80 GB, L40S; page lists 10 free A100-40 h / 5 L40S h | A100 sessions capped at 4 h; 40 GB is tight for ~35 GB; credit value not confirmed |
| AMD Developer Cloud | $100 to approved AI Developer Program members, ~50 MI300X h | MI300X 192 GB, ~$1.99/h (3p) | Approval up to 3 business days; expires 30 days after deposit; card auto-charged after; ROCm untested for this script |
| Vast.ai | None | A6000 $0.40, A100 SXM 80 GB $1.15 on demand; interruptible A6000 $0.16, A100 SXM $0.31 (live API) | Host reliability varies; interruptible can be preempted |
| Thunder Compute | None ($20 for US students) | A6000 $0.35, A100 80 GB $1.09, per minute, never preempted | |
| RunPod referral | $5–500 random (EU flat $5) after spending $10 | | New account via Google SSO only |
| Google Cloud $300 trial | | | Trial cannot add GPUs or request quota (provider docs) |
| Azure / AWS free | | | GPU quota 0 on free accounts |

## Proof of concept (2026-10-21)

`poc`: 208 clips × 4 epochs at batch 2 on an L40S ($1.95/h) at an assumed 5–15 clips/s ≈ 1–3 min training,
plus model load, four checkpoint saves and samples for baseline + 4 checkpoints (~12–18 min): **~20–30 min,
~$0.70–1.00** on Modal (upload/tokenize adds ~$0.20). The same on a Lightning free L40S hour or a
RunPod/Thunder A6000 would be $0–0.30, but needs the pipeline ported first; not worth it at this size.
