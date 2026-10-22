"""Modal pipeline for the Meorav fine-tune: upload -> tokenize -> poc | probe | sweep -> listen -> publish.

Order, after `training/prepare.py` has written data/prepared/ locally (set MODAL_PROFILE if your
Modal CLI has more than one profile):

    modal run training/modal_app.py::upload    # put data/prepared on the volume, fetch models, add codec codes
    modal run training/modal_app.py::poc       # proof of concept on whatever is recorded (L40S, ~$1)
    modal run training/modal_app.py::probe     # one short run + its samples (~1 h of data)
    modal run training/modal_app.py::sweep     # low / mid / high (2.5-3 h of data)
    modal run training/modal_app.py::listen    # samples per checkpoint for every run on the volume, + baseline
    modal volume get --force meorav-tts samples data   # -> data/samples/

`probe` takes --lr --epochs --batch-size --name; `listen` takes --runs low,mid (default: all).
`training/run.sh` wraps the whole sequence from fetching the takes.

Environment (all optional): MEORAV_MODAL_APP (app and volume name, default `meorav-tts`),
MEORAV_SPEAKER (the checkpoint's custom-voice name, default `daniel`), HF_TOKEN (only `publish`
needs it).

Volume `meorav-tts`:
    prepared/ (train_raw.jsonl, train_with_codes.jsonl, test.jsonl, ref.wav, ref.json, train/, test/)
    models/base/ models/tokenizer/ runs/<name>/{config.json,train.log,checkpoint-epoch-N/}
    samples/{baseline,real,<run>/checkpoint-epoch-N}/<line>.wav, samples/prompts.json

Upstream checked 2026-10-21: QwenLM/Qwen3-TTS finetuning/ unchanged since 022e286b98
(2026-03-17, "fix finetuning bug"); qwen-tts 0.1.1 is still the newest release (pins transformers
4.57.3, accelerate 1.12.0).
"""
import json
import os
from pathlib import Path

import modal

APP = os.environ.get("MEORAV_MODAL_APP", "meorav-tts")
VOL = modal.Volume.from_name(APP, create_if_missing=True)
V = Path("/vol")
QWEN_REPO_REF = "022e286b98fbec7e1e916cb940cdf532cd9f488e"  # main on 2026-10-21
QWEN_DIR = "/opt/Qwen3-TTS/finetuning"
BASE_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
SPEAKER = os.environ.get("MEORAV_SPEAKER", "daniel")  # sft_12hz.py --speaker_name; the checkpoint's only custom voice (spk_id 3000)
# HF_TOKEN goes to push_to_hf only when set locally, so the other stages run without it.
HF_SECRET = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else []
LOCAL_PREPARED = Path(__file__).resolve().parent.parent / "data/prepared"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "sox", "libsndfile1", "ffmpeg")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0", "qwen-tts==0.1.1", "tensorboard",
                 "scipy", "soundfile", "numpy", "huggingface_hub[hf_transfer]")
    .run_commands(
        f"git clone https://github.com/QwenLM/Qwen3-TTS.git /opt/Qwen3-TTS && git -C /opt/Qwen3-TTS checkout {QWEN_REPO_REF}",
        # flash-attn has no prebuilt wheel for this stack; SDPA is the supported fallback. The grep
        # fails the build if upstream renames the argument and the sed silently stops matching.
        f"sed -i 's/flash_attention_2/sdpa/' {QWEN_DIR}/sft_12hz.py && grep -q 'attn_implementation=\"sdpa\"' {QWEN_DIR}/sft_12hz.py",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Training runs this copy, not upstream's: label-shift, sub-talker, text_projection and fp32-master
    # fixes (header of the file). It imports dataset.py from the same directory.
    .add_local_file(Path(__file__).parent / "sft_12hz_patched.py", f"{QWEN_DIR}/sft_12hz_patched.py", copy=True)
)
app = modal.App(APP, image=image)

# Sweep (docs/training-plan.md). sft_12hz.py hard-codes Accelerator(gradient_accumulation_steps=4),
# so one optimizer update = 4 x batch_size clips. Batch 8 everywhere (effective 32) keeps LR the only
# variable. Clips average 4.0 s after trim (208 takes, 2026-10-21), so 2.5-3 h ~ 2,250-2,700 clips:
#   updates/epoch = ceil(clips / 32) ~ 70-85
#   low  2e-6 x 10 -> ~700-850 updates  (README's documented LR/epochs; it used batch 32 = eff. 128, ~210 updates: too few here)
#   mid  7e-6 x 6  -> ~420-510 updates
#   high 2e-5 x 3  -> ~210-255 updates  (README one-click script's LR/epochs; constant LR, no warmup: if loss spikes, add warmup)
# Each epoch writes a full checkpoint (~4.5 GB): the sweep leaves 19 (~85 GB) on the volume; prune after listening.
BATCH = 8
RUNS = {
    "low": dict(lr=2e-6, batch_size=BATCH, epochs=10),
    "mid": dict(lr=7e-6, batch_size=BATCH, epochs=6),
    "high": dict(lr=2e-5, batch_size=BATCH, epochs=3),
}
# Probe at ~1 h (~900 clips): batch 4 (eff. 16) -> ~57 updates/epoch, mid LR x 6 = ~340 updates. Minutes on an A100.
PROBE = dict(lr=7e-6, batch_size=4, epochs=6)
# Proof of concept (2026-10-21: the minimum viable path, just to establish that it works):
# whatever is recorded (208 clips / 14 min on 2026-10-21). Small data overfits fast: a fixed-script
# run on 12.6 min had the voice at epoch 2 and runaway generations by epoch 6 (PR #278 thread), so
# few epochs, a checkpoint every epoch, judged by ear. Batch 2 (eff. 8) -> ~26 updates/epoch, 4 epochs
# ~100 updates at 1e-5. fp32 master 1.7B ~35 GB -> an L40S (48 GB). docs/data-guide.md.
POC = dict(lr=1e-5, batch_size=2, epochs=4, save_every=1, gpu="L40S")

# Out-of-distribution lines: never in the prompt bank. Numbers spelled out; Hebrew words in the one
# canonical romanisation per word that the bank uses (docs/dataset.md).
OOD = [
    ("ood-plain", "I left the laptop updating overnight, and by the morning it had decided to forget every password I own."),
    ("ood-irish", "Ah, sure look, it'll be grand. I'll ring you back after the match, around half seven or so."),
    ("ood-hebrish-1", "I popped into the makolet on the way home, and honestly, the queue for the arnona office was worse than anything in Yerushalayim this week."),
    ("ood-hebrish-2", "Yalla, tachles: Bituach Leumi wants the form by Thursday, and I still need a copy of my teudat zehut from Misrad Hapnim."),
    ("ood-numbers", "The flight leaves at twenty past six on the fourteenth of March, and the fare came to four hundred and twelve shekels."),
    ("ood-question", "Wait, you're telling me the whole thing ran for three weeks and nobody noticed? Seriously?"),
    ("ood-long", "When I first moved to Jerusalem I couldn't understand why everyone kept telling me to wait until after the chagim, but by the second autumn I was saying it myself, and meaning it."),
]
N_TEST = 20


def _abs(p):
    return str(V / "prepared" / p)


@app.function(gpu="A10", timeout=60 * 60, volumes={V: VOL})
def tokenize():
    """Fetch base model + tokenizer, then add codec codes (upstream prepare_data.py) to the train JSONL."""
    import subprocess

    from huggingface_hub import snapshot_download

    VOL.reload()
    # sft_12hz.py copytree()s --init_model_path into every checkpoint, so the base must be a local dir.
    snapshot_download(BASE_ID, local_dir=V / "models/base")
    snapshot_download(TOKENIZER_ID, local_dir=V / "models/tokenizer")
    prep = V / "prepared"
    rows = [json.loads(l) for l in open(prep / "train_raw.jsonl") if l.strip()]
    with open(prep / "train_raw_abs.jsonl", "w") as f:  # prepare.py writes paths relative to prepared/
        for r in rows:
            f.write(json.dumps({**r, "audio": _abs(r["audio"]), "ref_audio": _abs(r["ref_audio"])}, ensure_ascii=False) + "\n")
    subprocess.run(["python", "prepare_data.py", "--device", "cuda:0",
                    "--tokenizer_model_path", str(V / "models/tokenizer"),
                    "--input_jsonl", str(prep / "train_raw_abs.jsonl"),
                    "--output_jsonl", str(prep / "train_with_codes.jsonl")],
                   cwd=QWEN_DIR, check=True)
    VOL.commit()
    return len(rows)


@app.function(gpu="A100-80GB", timeout=3 * 60 * 60, volumes={V: VOL})
def train(name: str, lr: float, batch_size: int, epochs: int, save_every: int = 1):
    import subprocess

    VOL.reload()
    out = V / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _ in open(V / "prepared/train_with_codes.jsonl"))
    ref = json.loads((V / "prepared/ref.json").read_text())
    (out / "config.json").write_text(json.dumps(dict(
        lr=lr, batch_size=batch_size, grad_accum=4, epochs=epochs, clips=n,
        updates_per_epoch=-(-n // (4 * batch_size)), base=BASE_ID, qwen_repo_ref=QWEN_REPO_REF,
        script="sft_12hz_patched.py", fp32_master=True, save_every=save_every,
        speaker=SPEAKER, ref_id=ref["id"]), indent=2))
    with open(out / "train.log", "w") as log:
        subprocess.run(["python", "sft_12hz_patched.py",
                        "--init_model_path", str(V / "models/base"),
                        "--output_model_path", str(out),
                        "--train_jsonl", str(V / "prepared/train_with_codes.jsonl"),
                        "--batch_size", str(batch_size), "--lr", str(lr),
                        "--num_epochs", str(epochs), "--speaker_name", SPEAKER,
                        "--save_every", str(save_every)],
                       cwd=QWEN_DIR, check=True, stdout=log, stderr=subprocess.STDOUT)
    VOL.commit()
    return name


def _test_lines():
    """Up to N_TEST held-out test-split sentences, round-robin across prompt sets."""
    p = V / "prepared/test.jsonl"
    rows = [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []
    by_set = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        by_set.setdefault(r.get("set"), []).append(r)
    picked = []
    while len(picked) < N_TEST and any(by_set.values()):
        for s in sorted(by_set, key=str):
            if by_set[s] and len(picked) < N_TEST:
                picked.append(by_set[s].pop(0))
    return picked


@app.function(gpu="L40S", timeout=2 * 60 * 60, volumes={V: VOL})
def synthesize(name: str):
    import shutil

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    VOL.reload()
    tests = _test_lines()
    items = [(r["id"], r["text"]) for r in tests] + OOD
    root = V / "samples"
    root.mkdir(exist_ok=True)
    (root / "prompts.json").write_text(json.dumps(dict(items), indent=2, ensure_ascii=False))

    def load(path):
        return Qwen3TTSModel.from_pretrained(str(path), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

    if name == "baseline":
        # The "before": zero-shot clone from the base model with the same reference, plus the real
        # (podcast-processed, 24 kHz) test recordings as the anchor.
        ref = json.loads((V / "prepared/ref.json").read_text())
        tts, dst = load(V / "models/base"), root / "baseline"
        dst.mkdir(exist_ok=True)
        for i, text in items:
            wavs, sr = tts.generate_voice_clone(text=text, language="English",
                                                ref_audio=str(V / "prepared/ref.wav"), ref_text=ref["text"])
            sf.write(dst / f"{i}.wav", wavs[0], sr)
        (root / "real").mkdir(exist_ok=True)
        for r in tests:
            shutil.copy(V / "prepared" / r["audio"], root / "real" / f"{r['id']}.wav")
    else:
        ckpts = sorted((V / "runs" / name).glob("checkpoint-epoch-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
        for ckpt in ckpts:
            dst = root / name / ckpt.name
            if dst.exists() and all((dst / f"{i}.wav").exists() for i, _ in items):
                continue  # already sampled; listen is safe to re-run after pruning or adding runs
            dst.mkdir(parents=True, exist_ok=True)
            tts = load(ckpt)
            for i, text in items:
                wavs, sr = tts.generate_custom_voice(text=text, speaker=SPEAKER, language="English")
                sf.write(dst / f"{i}.wav", wavs[0], sr)
            del tts
            torch.cuda.empty_cache()
    VOL.commit()
    return name


@app.function(timeout=60 * 60, volumes={V: VOL}, secrets=HF_SECRET)
def push_to_hf(run: str, epoch: int, repo: str, private: bool = True):
    """Push runs/<run>/checkpoint-epoch-<epoch> to a HF model repo, volume to HF with no local hop."""
    from huggingface_hub import HfApi

    VOL.reload()
    ckpt = V / "runs" / run / f"checkpoint-epoch-{epoch}"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not on the volume")
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=str(ckpt), commit_message=f"{run} checkpoint-epoch-{epoch}")
    cfg = V / "runs" / run / "config.json"
    if cfg.exists():
        api.upload_file(path_or_fileobj=str(cfg), path_in_repo="training_config.json", repo_id=repo)
    return f"https://huggingface.co/{repo}"


@app.local_entrypoint()
def publish(run: str, epoch: int, repo: str, public: bool = False):
    """One checkpoint -> HF (private unless --public). Needs HF_TOKEN in the local environment."""
    print("pushed", push_to_hf.remote(run, epoch, repo, not public))


@app.local_entrypoint()
def upload():
    """Put local data/prepared/ on the volume (overwriting), then tokenize on a GPU."""
    need = ["train_raw.jsonl", "ref.wav", "ref.json", "train"]
    missing = [n for n in need if not (LOCAL_PREPARED / n).exists()]
    if missing:
        raise SystemExit(f"{LOCAL_PREPARED} lacks {missing}: run `uv run training/prepare.py` first")
    try:  # drop the previous upload so takes removed locally do not linger on the volume
        VOL.remove_file("prepared", recursive=True)
    except Exception:
        pass
    with VOL.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_PREPARED), "/prepared")
    print(f"uploaded {LOCAL_PREPARED} -> {APP}:/prepared")
    print(f"tokenized {tokenize.remote()} train clips")


@app.local_entrypoint()
def poc(lr: float = POC["lr"], epochs: int = POC["epochs"], save_every: int = POC["save_every"], name: str = "poc"):
    """Smallest end-to-end check: train on what is uploaded, then baseline + checkpoint samples."""
    print("trained", train.with_options(gpu=POC["gpu"]).remote(name, lr, POC["batch_size"], epochs, save_every))
    for done in synthesize.map(["baseline", name]):
        print("sampled", done)
    print(f"download: modal volume get --force {APP} samples data")


@app.local_entrypoint()
def probe(lr: float = PROBE["lr"], epochs: int = PROBE["epochs"], batch_size: int = PROBE["batch_size"], name: str = "probe"):
    print("trained", train.remote(name, lr, batch_size, epochs))
    for done in synthesize.map(["baseline", name]):
        print("sampled", done)
    print(f"download: modal volume get --force {APP} samples data")


@app.local_entrypoint()
def sweep():
    for done in train.starmap([(n, c["lr"], c["batch_size"], c["epochs"]) for n, c in RUNS.items()]):
        print("trained", done)


@app.local_entrypoint()
def listen(runs: str = ""):
    names = [r for r in runs.split(",") if r]
    if not names:
        try:
            names = [Path(e.path).name for e in VOL.listdir("runs")]
        except Exception:  # no runs/ yet: baseline only
            names = []
    for done in synthesize.map(["baseline", *names]):
        print("sampled", done)
    print(f"download: modal volume get --force {APP} samples data")
