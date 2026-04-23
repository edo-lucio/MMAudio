"""Download a size-capped subset of VGGSound clips from YouTube.

Pulls the official VGGSound CSV from Oxford VGG, filters to ids present in
the local sets/vgg-{train,val,test}.tsv manifests, and uses yt-dlp + ffmpeg
to fetch each source trimmed to the 10 s annotated window. Output filenames
match the id format expected by the training extractor: {youtube_id}_{start:06d}.mp4
at OUTPUT_DIR, which lines up with VGGSound.root in config/data/base.yaml.

All settings are module-level constants below. The run stops once
accumulated on-disk size in OUTPUT_DIR reaches MAX_BYTES.

Clips are downloaded in proportionally interleaved order across splits, so
that the val/test/train ratio stays consistent no matter when the size cap
fires. E.g. with quotas val=50, test=100, train=850, every 100 clips you
download will be ~5 val, ~10 test, ~85 train.

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

VGGSOUND_CSV_URL = "https://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv"
SETS_DIR = "sets"                    # local split manifests (vgg-*.tsv)
OUTPUT_DIR = "./data/video"          # matches config/data/base.yaml VGGSound.root
MAX_BYTES = 5 * 1024 ** 3            # ~5 GB cap; change here for more/less
# Per-split quotas. The *ratio* of these values determines the proportion of
# each split in the download — order here no longer matters for fairness
# because interleaving is done proportionally (see parse_plan).
SPLIT_QUOTAS = {"val": 50, "test": 100, "train": 850}
NUM_WORKERS = 2                      # keep low to avoid triggering bot detection
CLIP_LENGTH_SEC = 10                 # VGGSound clips are all 10 s
VIDEO_HEIGHT = 384                   # matches training extractor resize
AUDIO_SR = 16000                     # matches 16 kHz training pipeline
YT_RETRIES = 3

# --- YouTube authentication ------------------------------------------------
# YouTube requires cookies to avoid bot-detection errors. Export them once
# and point COOKIES_FILE at the result. Two ways to get the file:
#
#   Option A — yt-dlp built-in export (easiest, run once in your terminal):
#     yt-dlp --cookies-from-browser chrome --cookies cookies.txt \
#            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
#
#   Option B — browser extension:
#     Install "Get cookies.txt LOCALLY" (Chrome) or "cookies.txt" (Firefox),
#     visit youtube.com while logged in, export → save as cookies.txt.
#
# Set COOKIES_FILE to the path of that file. Leave as None to omit (will
# likely hit bot-detection for any non-trivial batch).
COOKIES_FILE: str | None = "cookies.txt"   # ← set your path here

# Seconds to sleep before each yt-dlp call (uniform random in [MIN, MAX]).
# Spread across NUM_WORKERS this keeps the request rate low enough to avoid
# triggering YouTube rate limits.
JITTER_MIN = 1.0
JITTER_MAX = 4.0


def load_split_ids() -> dict[str, set[str]]:
    """Return {split: set(ids)} from local vgg-*.tsv manifests."""
    per_split: dict[str, set[str]] = {}
    for split in SPLIT_QUOTAS:
        path = os.path.join(SETS_DIR, f"vgg-{split}.tsv")
        if not os.path.exists(path):
            print(f"warning: {path} missing, skipping")
            per_split[split] = set()
            continue
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            per_split[split] = {row["id"] for row in reader}
    return per_split


def fetch_csv() -> str:
    with urllib.request.urlopen(VGGSOUND_CSV_URL) as r:
        return r.read().decode("utf-8")


def parse_plan(
    csv_text: str, per_split: dict[str, set[str]]
) -> list[tuple[str, str, int]]:
    """Return [(vid_id, youtube_id, start_sec), ...] interleaved proportionally
    across splits so every split's quota is sampled at the same fractional rate.

    For each clip we compute a sort key = (position_in_split / split_quota),
    which places the i-th percentile clip from every split at the same position
    in the global list. The size cap can therefore fire at any point and still
    yield a val/test/train ratio that matches SPLIT_QUOTAS.
    """
    # bucket CSV rows by split
    buckets: dict[str, list[tuple[str, str, int]]] = {s: [] for s in SPLIT_QUOTAS}
    reader = csv.reader(io.StringIO(csv_text))
    for row in reader:
        if len(row) < 4:
            continue
        ytid = row[0].strip()
        try:
            start_s = int(row[1])
        except ValueError:
            continue
        vid_id = f"{ytid}_{start_s:06d}"
        for split, ids in per_split.items():
            if vid_id in ids:
                buckets[split].append((vid_id, ytid, start_s))
                break

    # Build a proportionally interleaved plan.
    #
    # Sort key = fractional position within each split's quota:
    #   clip at index i of a split with quota Q  →  key = i / Q
    #
    # This means the first clip of every split sorts to ~0.0, the halfway
    # clip of every split sorts to ~0.5, etc. A global sort by this key
    # interleaves splits at the correct ratio throughout the list.
    tagged: list[tuple[float, str, tuple[str, str, int]]] = []
    for split, quota in SPLIT_QUOTAS.items():
        chosen = buckets[split][:quota]
        n = len(chosen)
        print(f"  {split}: {n}/{len(buckets[split])} available (quota {quota})")
        for i, item in enumerate(chosen):
            key = i / quota          # fractional position — not i/n — so that
                                     # a short split doesn't crowd the front
            tagged.append((key, split, item))

    tagged.sort(key=lambda x: x[0])
    plan = [item for _, _split, item in tagged]
    return plan


def already_have(vid_id: str) -> bool:
    return os.path.exists(os.path.join(OUTPUT_DIR, f"{vid_id}.mp4"))


def download_one(task: tuple[str, str, int]) -> tuple[str, int, str | None]:
    vid_id, ytid, start_s = task
    out_mp4 = os.path.join(OUTPUT_DIR, f"{vid_id}.mp4")
    if os.path.exists(out_mp4):
        return vid_id, 0, None

    url = f"https://www.youtube.com/watch?v={ytid}"
    src_tmpl = os.path.join(OUTPUT_DIR, f".{vid_id}.src.%(ext)s")

    # build the base yt-dlp command, injecting cookies if configured
    yt_base = ["yt-dlp", "-q", "--no-playlist", "--no-warnings",
               "-f", "mp4/bestvideo*+bestaudio/best",
               "--merge-output-format", "mp4"]
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        yt_base += ["--cookies", COOKIES_FILE]

    try:
        for attempt in range(YT_RETRIES + 1):
            # jitter before every attempt to spread load across workers
            time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
            rc = subprocess.call(yt_base + ["-o", src_tmpl, url])
            if rc == 0:
                break
            # exponential back-off on retries
            if attempt < YT_RETRIES:
                time.sleep(2 ** attempt * 5)
        else:
            return vid_id, 0, "yt-dlp failed"

        src = next(
            (os.path.join(OUTPUT_DIR, f)
             for f in os.listdir(OUTPUT_DIR)
             if f.startswith(f".{vid_id}.src.")),
            None,
        )
        if src is None:
            return vid_id, 0, "no source file"

        end_s = start_s + CLIP_LENGTH_SEC
        subprocess.check_call([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start_s), "-to", str(end_s), "-i", src,
            "-vf", f"scale=-2:{VIDEO_HEIGHT}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-ac", "1", "-ar", str(AUDIO_SR),
            out_mp4,
        ])
        os.remove(src)
    except Exception as e:
        for p in (out_mp4,):
            if os.path.exists(p):
                os.remove(p)
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith(f".{vid_id}.src."):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, f))
                except OSError:
                    pass
        return vid_id, 0, str(e)

    size = os.path.getsize(out_mp4) if os.path.exists(out_mp4) else 0
    return vid_id, size, None


def current_size() -> int:
    total = 0
    for f in os.listdir(OUTPUT_DIR):
        p = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(p) and not f.startswith("."):
            total += os.path.getsize(p)
    return total


if __name__ == "__main__":
    if shutil.which("yt-dlp") is None:
        sys.exit("yt-dlp not on PATH; install with `pip install yt-dlp`")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not on PATH")
    if COOKIES_FILE and not os.path.exists(COOKIES_FILE):
        print(
            f"warning: COOKIES_FILE={COOKIES_FILE!r} not found. "
            "  yt-dlp --cookies-from-browser chrome --cookies cookies.txt "
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    per_split = load_split_ids()
    total_allowed = sum(len(s) for s in per_split.values())
    print(f"{total_allowed} ids across splits: "
          f"{ {k: len(v) for k, v in per_split.items()} }")

    print(f"fetching {VGGSOUND_CSV_URL}")
    plan = parse_plan(fetch_csv(), per_split)
    plan = [t for t in plan if not already_have(t[0])]
    print(f"{len(plan)} clips to download "
          f"(target ratio  val:{SPLIT_QUOTAS['val']}  "
          f"test:{SPLIT_QUOTAS['test']}  train:{SPLIT_QUOTAS['train']})")

    start_size = current_size()
    print(f"current size: {start_size / 1e9:.2f} GB  cap: {MAX_BYTES / 1e9:.2f} GB")
    if start_size >= MAX_BYTES:
        print("cap already reached; nothing to do")
        sys.exit(0)
    if not plan:
        sys.exit(0)

    downloaded = 0
    running = start_size
    counts: dict[str, int] = {s: 0 for s in SPLIT_QUOTAS}

    with Pool(NUM_WORKERS) as pool:
        it = pool.imap_unordered(download_one, plan, chunksize=1)
        for vid_id, size, err in it:
            if err:
                print(f"[skip] {vid_id}: {err}")
                continue
            # infer split from vid_id for progress tracking
            for split, ids in per_split.items():
                if vid_id in ids:
                    counts[split] += 1
                    break
            running += size
            downloaded += 1
            if downloaded % 10 == 0:
                split_summary = "  ".join(
                    f"{s}={counts[s]}" for s in SPLIT_QUOTAS
                )
                print(f"  {downloaded} clips [{split_summary}]  "
                      f"+{(running - start_size) / 1e9:.2f} GB")
            if running >= MAX_BYTES:
                print(f"reached cap at {running / 1e9:.2f} GB; stopping")
                pool.terminate()
                break

    split_summary = "  ".join(f"{s}={counts[s]}" for s in SPLIT_QUOTAS)
    print(f"done. [{split_summary}]  final size: {current_size() / 1e9:.2f} GB")