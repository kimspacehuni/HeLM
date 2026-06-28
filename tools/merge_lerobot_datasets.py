"""
Merge two or more LeRobot v2.1 datasets into a single one, suitable as input to
train_pi0 with --train_dataset.root pointing at the merged dir.

What it does:
  1. Build unified meta/tasks.jsonl as the union of source task strings; each
     unique task gets a fresh task_index starting at 0 (preserves order of
     first appearance).
  2. For each source's episodes (in input order), assign new global episode
     indices starting at 0. Read each parquet, rewrite episode_index and
     task_index columns to the new IDs, save to dst data/chunk-XXX/.
  3. Symlink (or copy if --copy) videos to dst videos/chunk-XXX/<key>/.
  4. Rewrite meta/episodes.jsonl, meta/episodes_stats.jsonl with new IDs;
     aggregate meta/stats.json over all merged frames.
  5. Update meta/info.json (total_episodes, total_frames, total_chunks).

Image stats (observation.images.*) are propagated from the first source — those
should be identical across LIBERO-Mem subsets since same env. State/action stats
are recomputed properly from per-episode stats.

Usage:
  python tools/merge_lerobot_datasets.py \\
      --srcs /path/to/A /path/to/B [/path/to/C ...] \\
      --dst /path/to/merged \\
      [--chunks-size 500] \\
      [--copy]              # default is symlink for video mp4s
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    out = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def _dump_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_tasks_union(src_roots: List[Path]) -> Tuple[List[dict], List[Dict[int, int]]]:
    """Return (unified_tasks_jsonl_rows, per_src_old_to_new_taskidx).

    Task ordering: first appearance across sources, preserved.
    """
    unified: "OrderedDict[str, int]" = OrderedDict()
    per_src_map: List[Dict[int, int]] = []
    for src in src_roots:
        src_tasks = _load_jsonl(src / "meta/tasks.jsonl")
        old_to_new: Dict[int, int] = {}
        for t in src_tasks:
            text = str(t["task"]).strip()
            old_idx = int(t["task_index"])
            if text not in unified:
                unified[text] = len(unified)
            old_to_new[old_idx] = unified[text]
        per_src_map.append(old_to_new)
    rows = [{"task_index": idx, "task": text} for text, idx in unified.items()]
    return rows, per_src_map


def merge(
    src_roots: List[Path],
    dst_root: Path,
    chunks_size: int | None,
    copy_videos: bool,
) -> None:
    if dst_root.exists():
        raise FileExistsError(
            f"{dst_root} already exists. Remove or pick a different --dst."
        )

    # ---- 1. unified tasks
    unified_tasks_rows, per_src_taskidx_map = build_tasks_union(src_roots)
    print(f"[merge] unified tasks ({len(unified_tasks_rows)}):")
    for t in unified_tasks_rows:
        print(f"  task_index={t['task_index']} task={t['task']!r}")

    # ---- 2. global episode index assignment
    src_infos = [json.loads((s / "meta/info.json").read_text()) for s in src_roots]
    src_episode_counts = [int(i.get("total_episodes", 0)) for i in src_infos]
    total_eps = sum(src_episode_counts)
    if chunks_size is None:
        chunks_size = max(int(i.get("chunks_size", 500)) for i in src_infos)
        chunks_size = max(chunks_size, total_eps)  # so a single chunk-000 fits
    print(f"[merge] total_episodes={total_eps}  chunks_size={chunks_size}")

    # video keys: inspect first src's videos dir
    first_videos_root = src_roots[0] / "videos"
    if first_videos_root.is_dir():
        first_chunk = next(first_videos_root.iterdir())
        video_keys = sorted(d.name for d in first_chunk.iterdir() if d.is_dir())
    else:
        video_keys = []
    print(f"[merge] video keys: {video_keys}")

    # ---- 3. iterate, rewrite parquets, link videos
    dst_data_chunk0 = dst_root / "data" / "chunk-000"
    dst_data_chunk0.mkdir(parents=True, exist_ok=True)
    for key in video_keys:
        (dst_root / "videos" / "chunk-000" / key).mkdir(parents=True, exist_ok=True)

    merged_episodes_rows: List[dict] = []
    merged_eps_stats_rows: List[dict] = []
    next_global_ep = 0
    all_action = []
    all_state = []

    for src_idx, src in enumerate(src_roots):
        info = src_infos[src_idx]
        n_src = src_episode_counts[src_idx]
        src_chunks_size = int(info.get("chunks_size", 500))
        taskidx_map = per_src_taskidx_map[src_idx]

        # build local ep_idx -> task language map from src episodes.jsonl
        src_episodes = _load_jsonl(src / "meta/episodes.jsonl")
        src_eps_stats = _load_jsonl(src / "meta/episodes_stats.jsonl")
        ep_idx_to_episode = {int(e["episode_index"]): e for e in src_episodes}
        ep_idx_to_eps_stats = {int(e["episode_index"]): e for e in src_eps_stats}

        print(f"\n[merge] src[{src_idx}]: {src.name}  ({n_src} episodes)")
        for old_ep in range(n_src):
            new_ep = next_global_ep
            old_chunk = old_ep // src_chunks_size
            old_pq = src / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_ep:06d}.parquet"
            if not old_pq.is_file():
                print(f"  WARN: missing {old_pq}, skipping", file=sys.stderr)
                continue

            df = pd.read_parquet(old_pq)
            # remap task_index
            df["task_index"] = df["task_index"].astype(int).map(taskidx_map).astype("int64")
            # rewrite episode_index to new global id
            df["episode_index"] = np.full(len(df), new_ep, dtype=np.int64)
            # global running 'index' column will be recomputed in a second pass below
            new_pq = dst_data_chunk0 / f"episode_{new_ep:06d}.parquet"
            df.to_parquet(new_pq, index=False)

            # accumulate for stats
            actions = np.stack(df["action"].values).astype(np.float32)
            states = np.stack(df["observation.state"].values).astype(np.float32)
            all_action.append(actions)
            all_state.append(states)

            # episodes.jsonl entry
            ep_meta = ep_idx_to_episode.get(old_ep, {})
            merged_episodes_rows.append({
                "episode_index": new_ep,
                "tasks": ep_meta.get("tasks", []),
                "length": int(len(df)),
            })

            # episodes_stats.jsonl: keep src's per-episode stats but update
            # episode_index (and action stats may be slightly different after the
            # earlier transform; trust what's there).
            src_stats = ep_idx_to_eps_stats.get(old_ep, {}).get("stats", {})
            merged_eps_stats_rows.append({
                "episode_index": new_ep,
                "stats": src_stats,
            })

            # symlink videos
            for key in video_keys:
                old_mp4 = src / "videos" / f"chunk-{old_chunk:03d}" / key / f"episode_{old_ep:06d}.mp4"
                if not old_mp4.is_file():
                    continue
                new_mp4 = dst_root / "videos" / "chunk-000" / key / f"episode_{new_ep:06d}.mp4"
                if copy_videos:
                    shutil.copyfile(old_mp4, new_mp4)
                else:
                    try:
                        new_mp4.symlink_to(old_mp4.resolve())
                    except FileExistsError:
                        new_mp4.unlink()
                        new_mp4.symlink_to(old_mp4.resolve())
            next_global_ep += 1

    # ---- 3b. recompute global 'index' column across all merged parquets
    print(f"\n[merge] recomputing global 'index' column ...")
    running = 0
    for new_ep in range(next_global_ep):
        new_pq = dst_data_chunk0 / f"episode_{new_ep:06d}.parquet"
        df = pd.read_parquet(new_pq)
        df["index"] = np.arange(running, running + len(df), dtype=np.int64)
        df.to_parquet(new_pq, index=False)
        running += len(df)
    total_frames = running

    # ---- 4. meta files
    print(f"[merge] writing meta files ...")
    _dump_jsonl(dst_root / "meta/tasks.jsonl", unified_tasks_rows)
    _dump_jsonl(dst_root / "meta/episodes.jsonl", merged_episodes_rows)
    _dump_jsonl(dst_root / "meta/episodes_stats.jsonl", merged_eps_stats_rows)

    # info.json: clone from first src, then update counts
    info = dict(src_infos[0])
    info["total_episodes"] = next_global_ep
    info["total_frames"] = total_frames
    info["total_chunks"] = 1
    info["chunks_size"] = chunks_size
    if "total_tasks" in info:
        info["total_tasks"] = len(unified_tasks_rows)
    if "total_videos" in info:
        info["total_videos"] = next_global_ep * len(video_keys)
    (dst_root / "meta/info.json").write_text(json.dumps(info, indent=2))

    # stats.json: aggregate over all merged frames. Image stats are propagated
    # from first src (same env, same camera, identical pixel stats).
    print(f"[merge] computing aggregate stats over {total_frames} frames ...")
    all_action_np = np.concatenate(all_action, axis=0)
    all_state_np = np.concatenate(all_state, axis=0)

    def _agg(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(axis=0).astype(np.float64).tolist(),
            "std": arr.std(axis=0).astype(np.float64).tolist(),
            "min": arr.min(axis=0).astype(np.float64).tolist(),
            "max": arr.max(axis=0).astype(np.float64).tolist(),
            "count": int(arr.shape[0]),
        }

    src_stats = json.loads((src_roots[0] / "meta/stats.json").read_text())
    new_stats = dict(src_stats)
    new_stats["action"] = _agg(all_action_np)
    new_stats["observation.state"] = _agg(all_state_np)
    (dst_root / "meta/stats.json").write_text(json.dumps(new_stats, indent=2))

    print(f"\n[merge] DONE")
    print(f"  total episodes : {next_global_ep}")
    print(f"  total frames   : {total_frames}")
    print(f"  tasks          : {len(unified_tasks_rows)}")
    print(f"  output dir     : {dst_root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--srcs", nargs="+", required=True, help="Source LeRobot v2 dataset roots")
    ap.add_argument("--dst", required=True, help="Destination dir (must not exist)")
    ap.add_argument("--chunks-size", type=int, default=None,
                    help="chunks_size for merged dataset. Default: max(src chunks_size, total_eps).")
    ap.add_argument("--copy", action="store_true",
                    help="Copy video mp4s instead of symlinking (slower, more disk).")
    args = ap.parse_args()

    src_roots = [Path(s).resolve() for s in args.srcs]
    for s in src_roots:
        if not (s / "meta/info.json").is_file():
            print(f"ERROR: not a LeRobot dataset (missing meta/info.json): {s}", file=sys.stderr)
            return 2
    dst_root = Path(args.dst).resolve()
    merge(src_roots, dst_root, args.chunks_size, args.copy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
