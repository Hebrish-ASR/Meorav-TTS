#!/usr/bin/env bash
# The training pipeline end to end, one stage per argument (run from anywhere):
#
#   training/run.sh fetch     # HF dataset -> data/takes/ (train) and data/takes/test/ (held out); new files only
#   training/run.sh clean     # scripts/prep_clean.py, podcast chain -> data/processed/podcast/, data/processed/test/podcast/
#   training/run.sh prepare   # training/prepare.py -> data/prepared/ (24 kHz, trimmed, Qwen JSONL, fixed ref)
#   training/run.sh local     # fetch + clean + prepare (no Modal, no cost)
#   training/run.sh upload    # Modal: data/prepared -> volume, download models, codec-tokenize (A10)
#   training/run.sh poc       # Modal: proof of concept on whatever is recorded + samples (L40S, ~$1)
#   training/run.sh probe     # Modal: one short run + samples (~1 h of data)
#   training/run.sh sweep     # Modal: low / mid / high (2.5-3 h of data)
#   training/run.sh listen    # Modal: samples per checkpoint, every run on the volume + baseline
#   training/run.sh get       # volume samples/ -> data/samples/
#
# Modal stages bill whichever Modal profile is active (set MODAL_PROFILE to choose). The default HF
# dataset is public; set MEORAV_DATASET to train on your own recordings in the same layout
# (docs/dataset.md). HF_TOKEN is sent if set. Nothing here pushes to HF.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${MEORAV_DATASET:-Hebrish/Hebrish-TTS-Voice-Recordings}
APP=training/modal_app.py
VOLUME=${MEORAV_MODAL_APP:-meorav-tts}

fetch_split() {  # $1 = HF split, $2 = local dir
  local split=$1 dir=$2 auth=()
  [ -n "${HF_TOKEN:-}" ] && auth=(-H "authorization: Bearer $HF_TOKEN")
  mkdir -p "$dir/audio"
  curl -sfL "${auth[@]}" -o "$dir/metadata.jsonl" "https://huggingface.co/datasets/$REPO/resolve/main/data/$split/metadata.jsonl"
  local n=0
  for f in $(python3 -c "import json,sys;[print(json.loads(l)['file_name']) for l in open(sys.argv[1]) if l.strip()]" "$dir/metadata.jsonl"); do
    [ -s "$dir/$f" ] && continue
    curl -sfL "${auth[@]}" -o "$dir/$f" "https://huggingface.co/datasets/$REPO/resolve/main/data/$split/$f"; n=$((n + 1))
  done
  echo "$split: $(wc -l < "$dir/metadata.jsonl") rows, $n new files -> $dir"
}

stage() {
  case $1 in
    fetch) fetch_split train data/takes; fetch_split test data/takes/test ;;
    clean) uv run scripts/prep_clean.py
           uv run scripts/prep_clean.py --takes data/takes/test --out data/processed/test ;;
    prepare) uv run training/prepare.py ;;
    local) stage fetch; stage clean; stage prepare ;;
    upload|poc|probe|sweep|listen) modal run "$APP::$1" ;;
    get) modal volume get --force "$VOLUME" samples data ;;
    *) sed -n '2,13p' "$0"; exit 1 ;;
  esac
}

[ $# -gt 0 ] || { sed -n '2,13p' "$0"; exit 1; }
for s in "$@"; do stage "$s"; done
