"""Download a size-capped subset of AudioCaps clips from YouTube.

AudioCaps (Kim et al. 2019, NAACL) is a manually-captioned subset of AudioSet
designed for audio captioning / text-to-audio tasks. Each entry is a 10 s
YouTube-sourced audio clip with a free-text human caption.

This script mirrors `down.py` (which downloads VGGSound) but produces
audio-only outputs because AudioCaps does not need the visual stream:
the dataset is meant to expand the audio-text training pool used by
MMAudio when joint-training with audio-text pairs.

Output layout, after a successful run:

    data/audiocaps/audio/{youtube_id}_{start:06d}.wav   # 16 kHz mono PCM
    data/audiocaps/audiocaps-{train,val,test}.tsv        # local manifest
        columns: id<TAB>caption<TAB>youtube_id<TAB>start_time

The TSVs are emitted in the same shape that
`src/training/extract_audio_training_latents.py` expects via its
`--captions_tsv` argument, so they slot directly into the existing audio
extraction pipeline.

Requires: yt-dlp, ffmpeg on PATH.
"""

import csv
import io
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from multiprocessing import Pool

# AudioCaps splits live in the official GitHub repo. If the URLs ever
# change, edit these and rerun.
AUDIOCAPS_URLS = {
    "train": "https://raw.githubusercontent.com/cdminix/audiocaps/master/dataset/train.csv",
    "val":   "https://raw.githubusercontent.com/cdminix/audiocaps/master/dataset/val.csv",
    "test":  "https://raw.githubusercontent.com/cdminix/audiocaps/master/dataset/test.csv",
}

OUTPUT_DIR = "./data/audiocaps/audio"
TSV_DIR = "./data/audiocaps"          # one tsv per split goes here
MAX_BYTES = 8 * 1024 ** 3              # ~8 GB cap; raise if you want more
SPLIT_QUOTAS = {"val": 200, "test": 400, "train": 4000}
NUM_WORKERS = 2                        # keep low to avoid bot-detection
CLIP_LENGTH_SEC = 10                   # AudioCaps clips are 10 s
AUDIO_SR = 16000                       # matches MMAudio 16 kHz pipeline
YT_RETRIES = 3

# YouTube cookies — same scheme as down.py. None disables.
COOKIES_FILE: str | None = "cookies.txt"

JITTER_MIN = 1.0
JITTER_MAX = 4.0


def fetch_audiocaps_csvs() -> dict[str, list[dict]]:
    """Download the three official CSVs and parse them into row dicts.

    Returned schema per row: audiocap_id, youtube_id, start_time, caption.
    """
    out: dict[str, list[dict]] = {}
    for split, url in AUDIOCAPS_URLS.items():
        print(f"[fetch] {split} ← {url}", flush=True)
        with urllib.request.urlopen(url) as resp:
            text = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        for row in reader:
            row["start_time"] = int(float(row["start_time"]))
            rows.append(row)
        print(f"[fetch] {split}: {len(rows)} entries", flush=True)
        out[split] = rows
    return out


def existing_clip_ids() -> set[str]:
    """{youtube_id}_{start:06d} of every .wav already on disk."""
    if not os.path.isdir(OUTPUT_DIR):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")}


def total_bytes() -> int:
    if not os.path.isdir(OUTPUT_DIR):
        return 0
    n = 0
    for f in os.listdir(OUTPUT_DIR):
        try:
            n += os.path.getsize(os.path.join(OUTPUT_DIR, f))
        except OSError:
            pass
    return n


def build_download_plan(splits: dict[str, list[dict]]) -> list[tuple[str, dict]]:
    """Interleave per-split picks at a ratio matching the remaining quota
    of each split (existing on-disk clips are already counted).
    """
    on_disk = existing_clip_ids()
    print(f"[plan] {len(on_disk)} clips already on disk", flush=True)

    # Per-split: count existing matches, remaining quota, candidate pool
    pools: dict[str, list[dict]] = {}
    remaining: dict[str, int] = {}
    for split, rows in splits.items():
        already = sum(
            1 for r in rows
            if f'{r["youtube_id"]}_{int(r["start_time"]):06d}' in on_disk
        )
        remaining[split] = max(0, SPLIT_QUOTAS[split] - already)
        pools[split] = [
            r for r in rows
            if f'{r["youtube_id"]}_{int(r["start_time"]):06d}' not in on_disk
        ]
        random.shuffle(pools[split])
        print(f"[plan] {split}: {already} on disk / {SPLIT_QUOTAS[split]} target → "
              f"need {remaining[split]} more, {len(pools[split])} candidates",
              flush=True)

    # Round-robin interleave at the ratio given by `remaining`.
    plan: list[tuple[str, dict]] = []
    cursors = {s: 0 for s in pools}
    while True:
        # Largest deficit picks next.
        ranked = sorted(
            ((s, remaining[s] - sum(1 for q, _ in plan if q == s)) for s in pools),
            key=lambda kv: -kv[1],
        )
        if all(diff <= 0 for _, diff in ranked):
            break
        progress = False
        for split, _ in ranked:
            if cursors[split] >= len(pools[split]):
                continue
            current = sum(1 for q, _ in plan if q == split)
            if current >= remaining[split]:
                continue
            plan.append((split, pools[split][cursors[split]]))
            cursors[split] += 1
            progress = True
            break
        if not progress:
            break
    print(f"[plan] {len(plan)} clips queued for download", flush=True)
    return plan


def yt_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"


def download_one(task: tuple[str, dict]) -> tuple[str, str, bool, str]:
    split, row = task
    yid = row["youtube_id"]
    start = int(row["start_time"])
    out_id = f"{yid}_{start:06d}"
    out_path = os.path.join(OUTPUT_DIR, f"{out_id}.wav")
    if os.path.exists(out_path):
        return (split, out_id, True, "already-on-disk")

    time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))

    cookies = ["--cookies", COOKIES_FILE] if COOKIES_FILE and os.path.exists(COOKIES_FILE) else []

    # yt-dlp -x extracts audio only; -f bestaudio prefers audio streams.
    # We download to a temp file, then trim+resample with ffmpeg.
    tmp = out_path + ".raw.m4a"
    cmd = [
        "yt-dlp",
        "-q",
        "-x",                         # extract audio only
        "-f", "bestaudio",
        "--audio-format", "m4a",
        "--no-playlist",
        "--retries", str(YT_RETRIES),
        "-o", tmp,
        yt_url(yid),
        *cookies,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return (split, out_id, False, f"yt-dlp: {type(exc).__name__}")

    # ffmpeg: trim to [start, start+10s], mono, 16 kHz, PCM s16 wav.
    ff = [
        "ffmpeg",
        "-y", "-loglevel", "error",
        "-ss", str(start),
        "-t", str(CLIP_LENGTH_SEC),
        "-i", tmp,
        "-ac", "1",
        "-ar", str(AUDIO_SR),
        "-c:a", "pcm_s16le",
        out_path,
    ]
    try:
        subprocess.run(ff, check=True, capture_output=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return (split, out_id, False, f"ffmpeg: {type(exc).__name__}")
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    return (split, out_id, True, "downloaded")


def write_tsvs(splits: dict[str, list[dict]]) -> None:
    """Write one TSV per split listing only the clips currently on disk.

    Format matches `extract_audio_training_latents.py`'s `--captions_tsv`:
        id<TAB>caption<TAB>youtube_id<TAB>start_time
    """
    os.makedirs(TSV_DIR, exist_ok=True)
    on_disk = existing_clip_ids()
    for split, rows in splits.items():
        out = os.path.join(TSV_DIR, f"audiocaps-{split}.tsv")
        n = 0
        with open(out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "caption", "youtube_id", "start_time"])
            for r in rows:
                cid = f'{r["youtube_id"]}_{int(r["start_time"]):06d}'
                if cid not in on_disk:
                    continue
                w.writerow([cid, r["caption"], r["youtube_id"], r["start_time"]])
                n += 1
        print(f"[tsv]   wrote {n} rows → {out}", flush=True)


def main() -> int:
    if not shutil.which("yt-dlp"):
        print("error: yt-dlp not on PATH", file=sys.stderr); return 2
    if not shutil.which("ffmpeg"):
        print("error: ffmpeg not on PATH", file=sys.stderr); return 2

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TSV_DIR, exist_ok=True)

    splits = fetch_audiocaps_csvs()
    plan = build_download_plan(splits)

    used = total_bytes()
    print(f"[run] disk used: {used/1e9:.2f} GB / cap {MAX_BYTES/1e9:.2f} GB", flush=True)

    with Pool(NUM_WORKERS) as pool:
        for i, (split, cid, ok, msg) in enumerate(pool.imap_unordered(download_one, plan)):
            tag = "ok " if ok else "FAIL"
            print(f"[{i+1:5d}/{len(plan)}] {tag} [{split:5s}] {cid}: {msg}", flush=True)
            if total_bytes() >= MAX_BYTES:
                print(f"[run] hit MAX_BYTES cap ({MAX_BYTES/1e9:.2f} GB), stopping", flush=True)
                pool.terminate()
                break

    write_tsvs(splits)
    print(f"[run] done. final disk used: {total_bytes()/1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
