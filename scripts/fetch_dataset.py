"""Fetch one split of the HF dataset into a local audiofolder (metadata.jsonl + audio/), new files only.

    python3 scripts/fetch_dataset.py train data/takes            # what training/run.sh fetch does
    python3 scripts/fetch_dataset.py test data/takes/test
    python3 scripts/fetch_dataset.py train data/takes-from-stt --config from-stt

The dataset (MEORAV_DATASET, default Hebrish/Hebrish-TTS-Voice-Recordings) is read in either layout:
  - Parquet (the Hebrish datasets since 2026-10-30; present once meta/schema.json exists): index
    meta/<split>.jsonl (meta/<config>/<split>.jsonl for a second config) gives each take's shard and the byte
    range of its audio file inside it (`audio_offset`, `audio_length`; audio is stored verbatim). Only the
    shards holding a missing file are downloaded, one at a time to a temp file, and deleted after the files
    are cut out of them.
  - Audiofolder (data/<split>/metadata.jsonl + audio files; your own dataset, or an older revision): one
    request per missing file.
Either way the output is the same audiofolder, so scripts/prep_clean.py and training/prepare.py read it
unchanged. Stdlib only. HF_TOKEN is sent if set (needed for a private or gated dataset).
"""
import argparse
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
INDEX_KEYS = ("shard", "audio_path", "audio_offset", "audio_length")


def get(url, dest=None):
    """Bytes of `url`, or stream it to `dest`. The token goes to huggingface.co only, not to the CDN it redirects to."""
    class Strip(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            new = super().redirect_request(req, fp, code, msg, headers, newurl)
            if new is not None and urllib.parse.urlparse(newurl).hostname != "huggingface.co":
                new.headers.pop("Authorization", None)
                new.unredirected_hdrs.pop("Authorization", None)
            return new
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    with urllib.request.build_opener(Strip).open(urllib.request.Request(url, headers=headers), timeout=300) as r:
        if dest is None:
            return r.read()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)


def exists(url):
    try:
        get(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    return True


def jsonl(data):
    return [json.loads(l) for l in data.decode().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split")
    ap.add_argument("out", type=Path)
    ap.add_argument("--config", default=None, help="second config dir, e.g. from-stt (default: the default config)")
    ap.add_argument("--repo", default=os.environ.get("MEORAV_DATASET", "Hebrish/Hebrish-TTS-Voice-Recordings"))
    ap.add_argument("--revision", default="main")
    a = ap.parse_args()
    # Pin one commit so the index and the shards it names agree.
    sha = json.loads(get(f"{HF}/api/datasets/{a.repo}/revision/{urllib.parse.quote(a.revision, safe='')}"))["sha"]
    res = f"{HF}/datasets/{a.repo}/resolve/{sha}"
    (a.out / "audio").mkdir(parents=True, exist_ok=True)
    meta = "meta/schema.json" if not a.config else f"meta/{a.config}/schema.json"
    n = 0
    if exists(f"{res}/{meta}"):
        idx = jsonl(get(f"{res}/meta/{a.split}.jsonl" if not a.config else f"{res}/meta/{a.config}/{a.split}.jsonl"))
        missing = {}
        for r in idx:
            if not (a.out / r["audio_path"]).is_file() or (a.out / r["audio_path"]).stat().st_size != r["audio_length"]:
                missing.setdefault(r["shard"], []).append(r)
        for shard, rows in sorted(missing.items()):
            with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
                print(f"  {shard}: {len(rows)} takes", flush=True)
                get(f"{res}/{shard}", tmp.name)
                with open(tmp.name, "rb") as f:
                    for r in rows:
                        f.seek(r["audio_offset"])
                        dest = a.out / r["audio_path"]
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(f.read(r["audio_length"]))
                        n += 1
        rows = [{"file_name": r["audio_path"], **{k: v for k, v in r.items() if k not in INDEX_KEYS}} for r in idx]
        layout = "parquet"
    else:
        base = f"{res}/{a.config or 'data'}/{a.split}"
        rows = jsonl(get(f"{base}/metadata.jsonl"))
        for r in rows:
            dest = a.out / r["file_name"]
            if dest.is_file() and dest.stat().st_size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            get(f"{base}/{r['file_name']}", dest)
            n += 1
        layout = "audiofolder"
    (a.out / "metadata.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{a.split}{' (' + a.config + ')' if a.config else ''}: {len(rows)} rows, {n} new files -> {a.out} "
          f"({layout}, {a.repo}@{sha[:12]})")


if __name__ == "__main__":
    main()
