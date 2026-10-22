# Model selection

Decided 2026-10-21. An earlier plan (2026-10-15) reached the same primary but planned to train on
16 kHz phone audio. "Verified" = read in the
repo, code or HF API on 2026-10-21; otherwise second-hand.

## The problem

Daniel has an Irish accent. Two earlier clones came out American: an **Orpheus** fine-tune (~2025)
and a ~1 h **Resemble / zero-shot** clone. Zero-shot copies timbre and takes
prosody from the base model, which is where the US cadence comes from. Do not re-suggest either.

**No source shows any open TTS fine-tune keeping a non-US English accent** (Irish, Scottish,
Australian, Indian). The closest evidence is negative: [Qwen3-TTS #323](https://github.com/QwenLM/Qwen3-TTS/issues/323),
7 h of Bengali at lr 2e-6 for 10 epochs, came out Hindi-accented, and the loss stalled at epochs
2–3. So data design (natural register, free speech) matters as much as the model.

## Primary: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, full SFT

- Apache-2.0 (verified). Official `finetuning/sft_12hz.py`, single speaker only (verified).
- Raw text through the Qwen tokenizer, **no phonemiser** (verified, `dataset.py`). A phonemiser
  is the failure route for accent: espeak has no Irish-English voice.
- Full SFT updates every weight (`AdamW(qwen3tts.model.parameters())`, verified), so the talker
  can move off its US prior.
- 24 kHz: `dataset.py` asserts `sr == 24000` on the reference audio (verified). Capture is
  48 kHz and is downsampled at prep.
- Code facts that shape the sweep (verified in `sft_12hz.py` / `dataset.py`):
  - **No LR scheduler or warmup**: constant LR, grad-accumulation 4, clip 1.0. Add a short
    warmup if the higher-LR run is unstable.
  - The JSONL `language` field is read but **never used** (the prefill is always
    `codec_nothink_id`), so a language tag cannot be used to steer accent.
  - The speaker embedding is taken from the first batch: use **one fixed `ref_audio`** for every
    row (the README "strongly recommends" it), an obviously Irish, conversational 5–10 s take,
    and the same clip at inference.
  - The README contradicts itself (lr 2e-6 × 10 epochs vs. its script's 2e-5 × 3), hence a sweep.

## Fallback 1: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, SFT of the LLM stage, then flow

Apache-2.0, official recipe (`examples/libritts/cosyvoice3/run.sh`; the yaml says lr 1e-5,
`constantlr` for SFT; verified). The semantic-token LLM is where accent mostly lives and the flow
stage is timbre, so accent can be targeted on its own. A commenter on #323 reports a CosyVoice3
fine-tune doing better than Qwen3 on the same accent-sensitive problem (second-hand).

## Fallback 2: Chatterbox T3, LoRA (or full), community kit

MIT. No official training script; [gokhaneraslan/chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning)
(118 stars, last push 2026-06-01, verified). Its author recommends LoRA under 10 h; Turbo full
fine-tunes reportedly turn to noise. Keep `cfg_weight` low at inference. The podcast already
serves Chatterbox, so deployment is solved.

## Ruled out

| Model | Reason |
|---|---|
| IndexTTS2 (was fallback 1 on 2026-10-15) | bilibili licence §3.4(c) forbids using it to improve other models; no official training script |
| F5-TTS, Llasa, Spark-TTS, MaskGCT, Voxtral-TTS (CC-BY-NC); Fish S2 (research licence); Higgs Audio v3 (non-commercial); XTTS-v2 (CPML) | Licence |
| Kokoro, StyleTTS2, Zonos, NVIDIA Magpie (English) | espeak / IPA front end overrides the accent before the model sees it |
| Orpheus, Resemble/zero-shot | Already failed (above) |
| VibeVoice | TTS fine-tuning code absent (only `finetuning-asr/`) |
| Kyutai TTS 1.6B | No fine-tune recipe |

## Evaluation

No automatic accent metric is trustworthy, so it is a listening test: 20 held-out `test` prompts
plus out-of-distribution text, base vs. fine-tune at several epochs, real recordings as the anchor.
Pass = someone who knows Daniel's voice hears Irish, not American, cadence. Judge by ear, not loss.

Sources: [Qwen3-TTS finetuning](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning),
[CosyVoice](https://github.com/FunAudioLLM/CosyVoice), [Chatterbox](https://github.com/resemble-ai/chatterbox),
[IndexTTS](https://github.com/index-tts/index-tts), [Unsloth TTS guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning),
[CLARITY, arXiv 2511.11104](https://arxiv.org/html/2511.11104v2) (accent bias as data imbalance).
