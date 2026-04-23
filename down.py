"""Download a size-capped subset of VGGSound clips from YouTube.

Pulls the official VGGSound CSV from Oxford VGG, filters to ids present in
the local sets/vgg-{train,val,test}.tsv manifests, and uses yt-dlp + ffmpeg
to fetch each source trimmed to the 10 s annotated window. Output filenames
match the id format expected by the training extractor: {youtube_id}_{start:06d}.mp4
at OUTPUT_DIR, which lines up with VGGSound.root in config/data/base.yaml.

All settings are module-level constants below. The run stops once
accumulated on-disk size in OUTPUT_DIR reaches MAX_BYTES.

Requires: yt-dlp, ffmpeg on PATH.
"""

import csv
import io
import os
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
SPLITS = ("train", "val", "test")    # which local splits to pull ids from
NUM_WORKERS = 4                      # parallel downloads
CLIP_LENGTH_SEC = 10                 # VGGSound clips are all 10 s
VIDEO_HEIGHT = 384                   # matches training extractor resize
AUDIO_SR = 16000                     # matches 16 kHz training pipeline
YT_RETRIES = 2


def load_allowed_ids() -> set[str]:
    """Return {youtube_id}_{start:06d} ids from local vgg-*.tsv splits."""
    allowed: set[str] = set()
    for split in SPLITS:
        path = os.path.join(SETS_DIR, f"vgg-{split}.tsv")
        if not os.path.exists(path):
            print(f"warning: {path} missing, skipping")
            continue
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                allowed.add(row["id"])
    return allowed


def fetch_csv() -> str:
    with urllib.request.urlopen(VGGSOUND_CSV_URL) as r:
        return r.read().decode("utf-8")


def parse_plan(csv_text: str, allowed: set[str]) -> list[tuple[str, str, int]]:
    """Return [(vid_id, youtube_id, start_sec), ...] for rows in `allowed`."""
    plan: list[tuple[str, str, int]] = []
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
        if vid_id in allowed:
            plan.append((vid_id, ytid, start_s))
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

    try:
        for _ in range(YT_RETRIES + 1):
            rc = subprocess.call([
                "yt-dlp", "-q", "--no-playlist", "--no-warnings",
                "-f", "mp4/bestvideo*+bestaudio/best",
                "--merge-output-format", "mp4",
                "-o", src_tmpl, url,
            ])
            if rc == 0:
                break
            time.sleep(1)
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
        # also clean any stray source temp
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    allowed = load_allowed_ids()
    print(f"{len(allowed)} ids in local vgg-*.tsv manifests")

    print(f"fetching {VGGSOUND_CSV_URL}")
    plan = parse_plan(fetch_csv(), allowed)
    # skip ones we already have to avoid re-downloading
    plan = [t for t in plan if not already_have(t[0])]
    print(f"{len(plan)} rows to download")

    start_size = current_size()
    print(f"current size: {start_size / 1e9:.2f} GB  cap: {MAX_BYTES / 1e9:.2f} GB")
    if start_size >= MAX_BYTES:
        print("cap already reached; nothing to do")
        sys.exit(0)
    if not plan:
        sys.exit(0)

    downloaded = 0
    running = start_size
    with Pool(NUM_WORKERS) as pool:
        it = pool.imap_unordered(download_one, plan, chunksize=1)
        for vid_id, size, err in it:
            if err:
                print(f"[skip] {vid_id}: {err}")
                continue
            running += size
            downloaded += 1
            if downloaded % 5 == 0:
                print(f"  {downloaded} clips, +{(running - start_size) / 1e9:.2f} GB")
            if running >= MAX_BYTES:
                print(f"reached cap at {running / 1e9:.2f} GB; stopping")
                pool.terminate()
                break

    print(f"done. final size: {current_size() / 1e9:.2f} GB")
