"""
Build the v2 (non-rotated) LIBERO-Mem-LeRobot dataset by un-rotating the
v1 mp4 videos. The v1 conversion stored each frame after a 180° rotation
applied to the raw mujoco render; we now want frames in their natural
upright orientation, but the source HDF5 with `obs/` images is no longer
on disk (we only kept LIBERO-Mem-Raw).

Strategy:
    1) Copy parquet (data/) and meta files from v1 to v2 unchanged.
    2) For each mp4 under v1/videos/, transcode it with `vflip,hflip`
       (= another 180° rotation), which cancels the v1 rotation and
       leaves the saved frames in raw mujoco orientation.

Eval clients should pass --no-rotate when using v2.

Usage:
    python tools/rerotate_lerobot_videos.py \\
        --src /rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot \\
        --dst /rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2 \\
        --workers 4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


def list_videos(src: Path) -> List[Tuple[Path, Path]]:
    """Return [(src_mp4, rel_path), ...] for every mp4 under src/videos/."""
    out: List[Tuple[Path, Path]] = []
    videos_root = src / "videos"
    if not videos_root.is_dir():
        return out
    for mp4 in sorted(videos_root.rglob("*.mp4")):
        rel = mp4.relative_to(src)
        out.append((mp4, rel))
    return out


def transcode_one(src_mp4: Path, dst_mp4: Path, threads: int,
                  vfilter: str) -> Tuple[Path, bool, str]:
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if dst_mp4.exists():
        return src_mp4, True, "skipped (exists)"
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(src_mp4),
        "-vf", vfilter,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "veryfast",
        "-threads", str(threads),
        str(dst_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return src_mp4, True, "ok"
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")[-300:]
        return src_mp4, False, f"ffmpeg failed: {err}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source LeRobot dataset")
    ap.add_argument("--dst", required=True, help="output dataset dir")
    ap.add_argument("--filter", default="vflip,hflip",
                    help="ffmpeg -vf filter expression. Examples: "
                         "'vflip,hflip' (180° rotation, default), "
                         "'hflip' (left/right flip), "
                         "'vflip' (top/bottom flip), "
                         "'null' (no transform — just copy as h264).")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads-per-job", type=int, default=2,
                    help="ffmpeg threads per job; workers * threads = total parallelism")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not (src / "meta" / "info.json").is_file():
        print(f"ERROR: src does not look like a LeRobot dataset: {src}", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)

    # 1) Copy parquet + meta unchanged
    for sub in ("data", "meta"):
        s = src / sub
        d = dst / sub
        if not s.is_dir():
            continue
        if d.exists():
            print(f"  [copy] {sub}/ exists at dst, skipping")
            continue
        print(f"  [copy] {s} -> {d}")
        shutil.copytree(s, d)

    # 2) Find videos and transcode in parallel
    pairs = list_videos(src)
    if not pairs:
        print(f"WARNING: no mp4 found under {src}/videos", file=sys.stderr)
        return 0
    print(f"\n=== transcoding {len(pairs)} videos with filter='{args.filter}' "
          f"(workers={args.workers}, threads={args.threads_per_job}) ===")

    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for src_mp4, rel in pairs:
            dst_mp4 = dst / rel
            jobs.append(ex.submit(transcode_one, src_mp4, dst_mp4,
                                  args.threads_per_job, args.filter))

        n_ok = n_fail = 0
        for fut in tqdm(as_completed(jobs), total=len(jobs)):
            src_mp4, ok, msg = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                print(f"  ! {src_mp4.name}: {msg}", file=sys.stderr)

    print(f"\n✓ done. {n_ok} ok, {n_fail} failed.")
    print(f"  dataset at {dst}")
    print(f"  use --no-rotate in eval clients when training/evaluating on this set.")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
