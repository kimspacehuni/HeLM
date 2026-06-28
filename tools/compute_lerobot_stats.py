"""
Compute meta/stats.json for a HeLM-flavored LeRobot dataset.

Walks every parquet (action / observation.state / scalar fields) and every mp4
(per-channel image stats), accumulates running sums, and writes mean/std/min/max
in the schema HeLM training expects.

Usage:
    python tools/compute_lerobot_stats.py --root ~/data/LIBERO-Mem/LIBERO-Mem-LeRobot

    # quick check on smoke-test output
    python tools/compute_lerobot_stats.py --root /tmp/libero_smoke

    # subsample image frames for speed (every Nth frame)
    python tools/compute_lerobot_stats.py --root ~/data/LIBERO-Mem/LIBERO-Mem-LeRobot \
        --image-stride 4

Output:
    <root>/meta/stats.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# Vector-valued columns to compute stats for.
VECTOR_KEYS = ("action", "observation.state")
# Scalar-int / scalar-float columns. Stored as length-1 stats.
SCALAR_KEYS = ("timestamp", "frame_index", "episode_index", "task_index", "index")


# ---------- Running stats accumulators ----------

class VecStats:
    """Streaming mean/std/min/max for an N-dim vector."""

    def __init__(self) -> None:
        self.n: int = 0
        self.sum: np.ndarray | None = None
        self.sumsq: np.ndarray | None = None
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None

    def update(self, x: np.ndarray) -> None:
        # x: (T, D) or (T,)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        x = x.astype(np.float64, copy=False)
        if self.sum is None:
            d = x.shape[1]
            self.sum = np.zeros(d, dtype=np.float64)
            self.sumsq = np.zeros(d, dtype=np.float64)
            self.min = np.full(d, np.inf, dtype=np.float64)
            self.max = np.full(d, -np.inf, dtype=np.float64)
        self.n += x.shape[0]
        self.sum += x.sum(axis=0)
        self.sumsq += (x * x).sum(axis=0)
        self.min = np.minimum(self.min, x.min(axis=0))
        self.max = np.maximum(self.max, x.max(axis=0))

    def finalize(self) -> Dict[str, list]:
        if self.n == 0 or self.sum is None:
            return {"mean": [], "std": [], "min": [], "max": []}
        mean = self.sum / self.n
        var = self.sumsq / self.n - mean * mean
        var = np.clip(var, 0.0, None)  # numerical safety
        std = np.sqrt(var)
        return {
            "mean": mean.astype(np.float32).tolist(),
            "std":  std.astype(np.float32).tolist(),
            "min":  self.min.astype(np.float32).tolist(),
            "max":  self.max.astype(np.float32).tolist(),
        }


class ImgStats:
    """Streaming per-channel mean/std/min/max for image streams (uint8)."""

    def __init__(self, num_channels: int = 3) -> None:
        self.c = num_channels
        self.n: int = 0  # pixel count (H*W*T summed)
        self.sum = np.zeros(num_channels, dtype=np.float64)
        self.sumsq = np.zeros(num_channels, dtype=np.float64)
        self.min = np.full(num_channels, np.inf, dtype=np.float64)
        self.max = np.full(num_channels, -np.inf, dtype=np.float64)

    def update(self, frame: np.ndarray) -> None:
        # frame: (H, W, C) uint8 → flatten to (H*W, C)
        f = frame.reshape(-1, self.c).astype(np.float32)
        self.n += f.shape[0]
        self.sum += f.sum(axis=0)
        self.sumsq += (f.astype(np.float64) ** 2).sum(axis=0)
        self.min = np.minimum(self.min, f.min(axis=0))
        self.max = np.maximum(self.max, f.max(axis=0))

    def finalize(self) -> Dict[str, list]:
        # LeRobot stores image stats as (C, 1, 1) for broadcast against (C, H, W)
        if self.n == 0:
            empty = [[[0.0]] for _ in range(self.c)]
            return {"mean": empty, "std": empty, "min": empty, "max": empty}
        mean = self.sum / self.n
        var = self.sumsq / self.n - mean * mean
        var = np.clip(var, 0.0, None)
        std = np.sqrt(var)

        def shaped(v: np.ndarray) -> list:
            return [[[float(v[c])]] for c in range(self.c)]

        return {
            "mean": shaped(mean),
            "std":  shaped(std),
            "min":  shaped(self.min),
            "max":  shaped(self.max),
        }


# ---------- Per-feature workers ----------

def collect_parquet_stats(
    parquet_files: List[Path],
) -> Dict[str, Dict[str, list]]:
    vector_stats = {k: VecStats() for k in VECTOR_KEYS}
    scalar_stats = {k: VecStats() for k in SCALAR_KEYS}

    for fp in tqdm(parquet_files, desc="parquet"):
        df = pd.read_parquet(fp, engine="pyarrow")
        for k in VECTOR_KEYS:
            if k not in df.columns:
                continue
            arr = np.asarray(df[k].tolist())  # list of lists → (T, D)
            vector_stats[k].update(arr)
        for k in SCALAR_KEYS:
            if k not in df.columns:
                continue
            arr = np.asarray(df[k].tolist()).reshape(-1)
            scalar_stats[k].update(arr)

    out: Dict[str, Dict[str, list]] = {}
    for k, s in vector_stats.items():
        if s.n > 0:
            out[k] = s.finalize()
    for k, s in scalar_stats.items():
        if s.n > 0:
            out[k] = s.finalize()
    return out


def collect_image_stats(
    video_files_by_key: Dict[str, List[Path]],
    image_stride: int = 1,
) -> Dict[str, Dict[str, list]]:
    out: Dict[str, Dict[str, list]] = {}
    for key, paths in video_files_by_key.items():
        stats = ImgStats(num_channels=3)
        # iterate frames across all videos for this camera
        for vp in tqdm(paths, desc=f"video[{key}]"):
            try:
                reader = imageio.get_reader(str(vp))
            except Exception as e:
                print(f"  ! cannot open {vp}: {e}", file=sys.stderr)
                continue
            try:
                for i, frame in enumerate(reader):
                    if image_stride > 1 and (i % image_stride) != 0:
                        continue
                    stats.update(np.asarray(frame))
            finally:
                reader.close()
        out[key] = stats.finalize()
    return out


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path to LeRobot dataset root (containing meta/, data/, videos/)")
    ap.add_argument("--image-stride", type=int, default=1,
                    help="Sample every Nth frame from each video for image stats (default 1=all)")
    ap.add_argument("--skip-images", action="store_true",
                    help="Skip image stats (only state/action/scalar)")
    ap.add_argument("--out", default=None,
                    help="Output path (default: <root>/meta/stats.json)")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not (root / "meta" / "info.json").is_file():
        print(f"meta/info.json not found in {root}", file=sys.stderr)
        return 1

    info = json.loads((root / "meta" / "info.json").read_text())

    # ---- discover files ----
    parquet_files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_files:
        print(f"no parquet files under {root}/data/", file=sys.stderr); return 1
    print(f"found {len(parquet_files)} parquet file(s)")

    image_keys = sorted(
        k for k, v in info.get("features", {}).items()
        if isinstance(v, dict) and v.get("dtype") == "video"
    )
    video_files_by_key: Dict[str, List[Path]] = {}
    for k in image_keys:
        files = sorted(root.glob(f"videos/chunk-*/{k}/episode_*.mp4"))
        video_files_by_key[k] = files
        print(f"found {len(files):>4} video(s) for {k}")

    # ---- collect stats ----
    print("\n== parquet stats ==")
    stats = collect_parquet_stats(parquet_files)

    if not args.skip_images and image_keys:
        print("\n== image stats ==")
        stats.update(collect_image_stats(video_files_by_key, args.image_stride))

    # ---- write ----
    out_path = Path(args.out).expanduser() if args.out else (root / "meta" / "stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ wrote {out_path}")

    # ---- short report ----
    print("\nfeatures with stats:")
    for k, v in stats.items():
        if isinstance(v.get("mean"), list) and v["mean"] and isinstance(v["mean"][0], list):
            shape = f"({len(v['mean'])},1,1)"  # image-style
        else:
            shape = f"({len(v.get('mean', []))},)"
        print(f"  {k:<35} shape={shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
