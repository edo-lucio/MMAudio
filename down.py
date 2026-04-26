"""Download a size-capped subset of VGGSound clips from YouTube.

Pulls the official VGGSound CSV from Oxford VGG, filters to ids present in
the local sets/vgg-{train,val,test}.tsv manifests, and uses yt-dlp + ffmpeg
to fetch each source trimmed to the 10 s annotated window. Output filenames
match the id format expected by the training extractor: {youtube_id}_{start:06d}.mp4
at OUTPUT_DIR, which lines up with VGGSound.root in config/data/base.yaml.

All settings are module-level constants below. The run stops once
accumulated on-disk size in OUTPUT_DIR reaches MAX_BYTES.

Resume-aware: clips already on disk are counted per split before building the
download plan, so the remaining quota for each split is adjusted accordingly.
The proportional interleaving is computed on *remaining* quotas, not total
quotas, meaning the val/test/train ratio of newly-downloaded clips will
compensate for any imbalance already present on disk.

E.g. quotas val=50, test=100, train=850. If you already have val=48, test=30,
train=200 on disk, the remaining quotas are val=2, test=70, train=650 and the
new downloads are interleaved at that corrected ratio.

Candidate clips are shuffled randomly before selection so that restarts explore
different parts of the dataset instead of retrying the same failing videos.

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
OUTPUT_DIR = "./data/video"         # matches config/data/base.yaml VGGSound.root
MAX_BYTES = 5 * 1024 ** 3            # ~5 GB cap; change here for more/less
# Per-split quotas. The *ratio* of these values determines the target proportion.
# On resume the downloader computes per-split deficits (quota − already on disk)
# and interleaves only the missing clips, so the final collection always trends
# toward this ratio regardless of how many times you restart.
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


def already_have(vid_id: str) -> bool:
    return os.path.exists(os.path.join(OUTPUT_DIR, f"{vid_id}.mp4"))


def count_existing_per_split(per_split: dict[str, set[str]]) -> dict[str, int]:
    """Count clips already on disk for each split."""
    return {
        split: sum(1 for vid_id in ids if already_have(vid_id))
        for split, ids in per_split.items()
    }


def parse_plan(
    csv_text: str,
    per_split: dict[str, set[str]],
    existing_counts: dict[str, int],
) -> list[tuple[str, str, int]]:
    """Return [(vid_id, youtube_id, start_sec), ...] interleaved proportionally
    across splits using *remaining* per-split quotas (quota − already on disk).

    Candidates are shuffled randomly before slicing so that each run explores
    a different subset of the available clips. This prevents repeated restarts
    from retrying the same failing videos indefinitely.

    Sort key = i / remaining_quota, which places the i-th clip of each split
    at the correct fractional position in the global list. Because we use the
    remaining quota as the denominator, splits that still need many clips are
    spread evenly through the list while splits that are nearly full appear
    briefly at the front and then stop contributing — exactly what you want
    when resuming a partially-completed download.
    """
    # Bucket CSV rows by split, skipping already-downloaded clips.
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
                if not already_have(vid_id):   # skip clips already on disk
                    buckets[split].append((vid_id, ytid, start_s))
                break

    tagged: list[tuple[float, str, tuple[str, str, int]]] = []
    for split, quota in SPLIT_QUOTAS.items():
        have = existing_counts.get(split, 0)
        remaining = max(0, quota - have)
        candidates = buckets[split]          # already filtered above
        random.shuffle(candidates)           # randomise so restarts vary
        chosen = candidates[:remaining]
        print(
            f"  {split}: {have} on disk + {len(chosen)} to fetch  "
            f"(quota {quota}, {len(candidates)} available in CSV)"
        )
        if remaining == 0:
            continue
        for i, item in enumerate(chosen):
            # Denominator is remaining_quota so that a split with few clips
            # left doesn't artificially crowd the early part of the list.
            key = i / remaining
            tagged.append((key, split, item))

    tagged.sort(key=lambda x: x[0])
    return [item for _, _split, item in tagged]


def download_one(task: tuple[str, str, int]) -> tuple[str, int, str | None]:
    vid_id, ytid, start_s = task
    out_mp4 = os.path.join(OUTPUT_DIR, f"{vid_id}.mp4")
    if os.path.exists(out_mp4):
        return vid_id, 0, None

    url = f"https://www.youtube.com/watch?v={ytid}"
    src_tmpl = os.path.join(OUTPUT_DIR, f".{vid_id}.src.%(ext)s")

    yt_base = ["yt-dlp", "-q", "--no-playlist", "--no-warnings",
               "-f", "mp4/bestvideo*+bestaudio/best",
               "--merge-output-format", "mp4"]
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        yt_base += ["--cookies", COOKIES_FILE]

    try:
        for attempt in range(YT_RETRIES + 1):
            time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
            rc = subprocess.call(yt_base + ["-o", src_tmpl, url])
            if rc == 0:
                break
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
        if os.path.exists(out_mp4):
            os.remove(out_mp4)
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
            "Run:  yt-dlp --cookies-from-browser chrome --cookies cookies.txt "
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    per_split = load_split_ids()
    print(f"manifest ids: { {k: len(v) for k, v in per_split.items()} }")

    # Count what's already on disk before touching the CSV, so parse_plan can
    # compute per-split deficits and build a correctly balanced remaining plan.
    existing_counts = count_existing_per_split(per_split)
    print(f"already on disk: {existing_counts}  "
          f"(total {sum(existing_counts.values())})")

    print(f"fetching {VGGSOUND_CSV_URL}")
    plan = parse_plan(fetch_csv(), per_split, existing_counts)

    remaining_needed = {s: max(0, SPLIT_QUOTAS[s] - existing_counts[s]) for s in SPLIT_QUOTAS}
    print(
        f"{len(plan)} clips to download  "
        f"(remaining quotas: "
        + "  ".join(f"{s}={remaining_needed[s]}" for s in SPLIT_QUOTAS)
        + ")"
    )

    start_size = current_size()
    print(f"current size: {start_size / 1e9:.2f} GB  cap: {MAX_BYTES / 1e9:.2f} GB")
    if start_size >= MAX_BYTES:
        print("cap already reached; nothing to do")
        sys.exit(0)
    if not plan:
        print("all quotas already satisfied; nothing to do")
        sys.exit(0)

    downloaded = 0
    running = start_size
    # Session counts (clips fetched this run, not counting what was already on disk)
    session_counts: dict[str, int] = {s: 0 for s in SPLIT_QUOTAS}

    with Pool(NUM_WORKERS) as pool:
        it = pool.imap_unordered(download_one, plan, chunksize=1)
        for vid_id, size, err in it:
            if err:
                print(f"[skip] {vid_id}: {err}")
                continue
            for split, ids in per_split.items():
                if vid_id in ids:
                    session_counts[split] += 1
                    break
            running += size
            downloaded += 1
            if downloaded % 10 == 0:
                # Show combined totals (existing + this session) for each split
                combined = {s: existing_counts[s] + session_counts[s] for s in SPLIT_QUOTAS}
                split_summary = "  ".join(
                    f"{s}={combined[s]}/{SPLIT_QUOTAS[s]}" for s in SPLIT_QUOTAS
                )
                print(
                    f"  {downloaded} new clips [{split_summary}]  "
                    f"+{(running - start_size) / 1e9:.2f} GB this run"
                )
            if running >= MAX_BYTES:
                print(f"reached cap at {running / 1e9:.2f} GB; stopping")
                pool.terminate()
                break

    combined = {s: existing_counts[s] + session_counts[s] for s in SPLIT_QUOTAS}
    split_summary = "  ".join(f"{s}={combined[s]}/{SPLIT_QUOTAS[s]}" for s in SPLIT_QUOTAS)
    print(f"done. [{split_summary}]  final size: {current_size() / 1e9:.2f} GB")