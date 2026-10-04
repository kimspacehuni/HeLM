"""
Split a unified LeRobot dataset (one tree, multiple tasks) into per-task LeRobot
subdirectories. No video re-encoding — mp4 files are copied or moved as-is, and
parquet is rewritten only to reset task_index/episode_index/index columns.

Input layout (produced by tools/libero_to_lerobot.py):
    <src>/
      meta/{info.json, episodes.jsonl, tasks.jsonl}
      data/chunk-XXX/episode_NNNNNN.parquet     # episodes mixed across tasks
      videos/chunk-XXX/observation.images.<key>/episode_NNNNNN.mp4

Output layout (one subdir per task):
    <out>/
      task_00_<short_name>/
        meta/{info.json, episodes.jsonl, tasks.jsonl}
        data/chunk-XXX/episode_NNNNNN.parquet     # re-indexed 0..N-1
        videos/chunk-XXX/observation.images.<key>/episode_NNNNNN.mp4
      task_01_<short_name>/
        ...

Within each per-task dataset:
  - episode_index runs 0..N-1 (was scattered indices in the unified set)
  - task_index is always 0 (only one task per output dir)
  - global index is reset to start at 0
  - chunk numbering is recomputed from chunks_size (default 50)

Usage:
    # copy (safe, doubles disk usage temporarily)
    python tools/split_lerobot_by_task.py \
        --src ~/data/LIBERO-Mem/LIBERO-Mem-LeRobot \
        --out ~/data/LIBERO-Mem/LIBERO-Mem-LeRobot-PerTask

    # move (saves disk, destroys unified)
    python tools/split_lerobot_by_task.py \
        --src ~/data/LIBERO-Mem/LIBERO-Mem-LeRobot \
        --out ~/data/LIBERO-Mem/LIBERO-Mem-LeRobot-PerTask \
        --move
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------- helpers ----------

def sanitize(name: str, max_len: int = 60) -> str:
    """Make a string safe for use as a directory name."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip().replace(" ", "_"))
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] or "task"


def task_dir_name(task_idx: int, task_name: str) -> str:
    return f"task_{task_idx:02d}_{sanitize(task_name)}"


def episode_chunk(episode_idx: int, chunks_size: int) -> int:
    return episode_idx // chunks_size


def load_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------- core ----------

def split_dataset(src: Path, out: Path, move: bool, dry_run: bool) -> int:
    if not (src / "meta" / "info.json").is_file():
        print(f"meta/info.json not found in {src}", file=sys.stderr); return 1

    info = json.loads((src / "meta" / "info.json").read_text())
    src_tasks = {t["task_index"]: t["task"]
                 for t in load_jsonl(src / "meta" / "tasks.jsonl")}
    src_episodes = load_jsonl(src / "meta" / "episodes.jsonl")
    chunks_size = int(info.get("chunks_size", 50))
    print(f"src: {src}")
    print(f"  total_episodes={info.get('total_episodes')}, "
          f"total_frames={info.get('total_frames')}, "
          f"total_tasks={info.get('total_tasks')}")
    print(f"  chunks_size={chunks_size}")

    image_keys = sorted(
        k for k, v in info.get("features", {}).items()
        if isinstance(v, dict) and v.get("dtype") == "video"
    )
    print(f"  image_keys={image_keys}")

    # ---- bucket episodes by task_index (from parquet, ground truth) ----
    by_task: Dict[int, List[dict]] = defaultdict(list)
    for ep in tqdm(src_episodes, desc="indexing"):
        orig_ep = int(ep["episode_index"])
        orig_chunk = episode_chunk(orig_ep, chunks_size)
        pq_path = src / "data" / f"chunk-{orig_chunk:03d}" / f"episode_{orig_ep:06d}.parquet"
        if not pq_path.is_file():
            print(f"  ! missing parquet for episode {orig_ep}: {pq_path}", file=sys.stderr)
            continue
        # Read just the first row's task_index — fast.
        df = pd.read_parquet(pq_path, columns=["task_index"])
        task_idx = int(df["task_index"].iloc[0])
        by_task[task_idx].append({
            "orig_ep": orig_ep,
            "orig_chunk": orig_chunk,
            "length": int(ep["length"]),
            "tasks": list(ep["tasks"]),
        })

    print(f"\nbucketed: {len(by_task)} task(s)")
    for ti in sorted(by_task):
        print(f"  task_{ti:02d} ({src_tasks.get(ti, '?')!r}): {len(by_task[ti])} episodes")

    if dry_run:
        print("\n[dry-run] no files moved/copied.")
        return 0

    out.mkdir(parents=True, exist_ok=True)

    # ---- create one per-task dataset ----
    for orig_task_idx in sorted(by_task):
        task_name = src_tasks.get(orig_task_idx, f"task_{orig_task_idx}")
        dst_root = out / task_dir_name(orig_task_idx, task_name)
        dst_root.mkdir(parents=True, exist_ok=True)
        (dst_root / "meta").mkdir(parents=True, exist_ok=True)
        print(f"\n=== {dst_root.name} ===")

        # ----- new episode/index numbering -----
        new_episodes_meta: List[dict] = []
        running_index = 0  # global frame index within this per-task dataset

        # Sort by original episode index so order is deterministic.
        episodes = sorted(by_task[orig_task_idx], key=lambda x: x["orig_ep"])

        for new_ep, item in enumerate(tqdm(episodes, desc=f"  task {orig_task_idx}")):
            orig_ep = item["orig_ep"]
            orig_chunk = item["orig_chunk"]
            length = item["length"]
            new_chunk = episode_chunk(new_ep, chunks_size)

            # ----- parquet: rewrite episode_index/task_index/index columns -----
            src_pq = src / "data" / f"chunk-{orig_chunk:03d}" / f"episode_{orig_ep:06d}.parquet"
            dst_pq = dst_root / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_ep:06d}.parquet"
            dst_pq.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_parquet(src_pq)
            df["episode_index"] = np.full(length, new_ep, dtype=np.int64)
            df["task_index"] = np.zeros(length, dtype=np.int64)
            df["index"] = np.arange(length, dtype=np.int64) + running_index
            df.to_parquet(dst_pq, engine="pyarrow")
            running_index += length

            # ----- videos: copy or move per camera -----
            for ik in image_keys:
                src_mp4 = (src / "videos" / f"chunk-{orig_chunk:03d}" / ik
                           / f"episode_{orig_ep:06d}.mp4")
                dst_mp4 = (dst_root / "videos" / f"chunk-{new_chunk:03d}" / ik
                           / f"episode_{new_ep:06d}.mp4")
                dst_mp4.parent.mkdir(parents=True, exist_ok=True)
                if not src_mp4.is_file():
                    print(f"  ! missing video {src_mp4}", file=sys.stderr)
                    continue
                if move:
                    shutil.move(str(src_mp4), str(dst_mp4))
                else:
                    shutil.copyfile(src_mp4, dst_mp4)

            new_episodes_meta.append({
                "episode_index": int(new_ep),
                "tasks": item["tasks"],
                "length": length,
            })

        # ----- meta files for this per-task dataset -----
        n_eps = len(episodes)
        n_frames = running_index
        n_chunks = (n_eps + chunks_size - 1) // chunks_size

        per_task_info = deepcopy(info)
        per_task_info["total_episodes"] = n_eps
        per_task_info["total_frames"] = n_frames
        per_task_info["total_tasks"] = 1
        per_task_info["total_videos"] = n_eps * len(image_keys)
        per_task_info["total_chunks"] = n_chunks
        per_task_info["splits"] = {"train": f"0:{n_eps}"}

        with open(dst_root / "meta" / "info.json", "w") as f:
            json.dump(per_task_info, f, indent=2)
        write_jsonl(dst_root / "meta" / "episodes.jsonl", new_episodes_meta)
        write_jsonl(dst_root / "meta" / "tasks.jsonl",
                    [{"task_index": 0, "task": task_name}])
        print(f"  → {n_eps} episodes, {n_frames} frames, {n_chunks} chunks")

    print(f"\n✓ wrote {len(by_task)} per-task dataset(s) under {out}")
    if move:
        print("  (videos moved from src → dst)")
        # Optionally clean up empty src dirs, but leave that to the user.
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Unified LeRobot dataset root")
    ap.add_argument("--out", required=True, help="Output root for per-task subdirs")
    ap.add_argument("--move", action="store_true",
                    help="Move mp4 files instead of copying (saves disk; destroys src videos)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print bucket counts only; no I/O")
    args = ap.parse_args()
    return split_dataset(
        src=Path(args.src).expanduser(),
        out=Path(args.out).expanduser(),
        move=args.move,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
