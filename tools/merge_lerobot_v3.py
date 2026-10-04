"""Merge two or more LeRobot v3.0 datasets into a single one.

Layout difference vs v2.1 (which tools/merge_lerobot_datasets.py handles):
- v3.0 stores all-episodes-per-chunk in one parquet (data/chunk-XXX/file-XXX.parquet)
- videos are one MP4 per video-key per chunk-file (concatenated streams)
- episode metadata lives at meta/episodes/chunk-XXX/file-XXX.parquet with
  per-episode pointers (chunk_index, file_index, from_timestamp, to_timestamp)
  into the data/video files.

Merge strategy: keep each source's video files as a SEPARATE file_index under
the merged dst (chunk-000/file-{src_idx}.parquet) — no MP4 concatenation needed.
Concat data parquets into ONE dst file, reindex episode_index and task_index
globally, rewrite per-episode meta to point at the right file_index.

Usage:
  python tools/merge_lerobot_v3.py \\
      --srcs <root1> <root2> [...] \\
      --dst <out_root>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_info(p: Path) -> dict:
    return json.loads((p / "meta/info.json").read_text())


def load_tasks(p: Path) -> pd.DataFrame:
    """v3 stores tasks at meta/tasks.parquet, indexed by task_name string."""
    df = pd.read_parquet(p / "meta/tasks.parquet")
    df = df.reset_index().rename(columns={df.index.name or "index": "task"})
    return df[["task_index", "task"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--srcs", nargs="+", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--copy", action="store_true", help="copy mp4s instead of symlink")
    args = ap.parse_args()

    src_roots = [Path(s).expanduser().resolve() for s in args.srcs]
    dst_root = Path(args.dst).expanduser().resolve()
    if dst_root.exists():
        print(f"ERROR: {dst_root} already exists", file=sys.stderr)
        return 1

    src_infos = [load_info(s) for s in src_roots]
    for i, info in enumerate(src_infos):
        if info.get("codebase_version") != "v3.0":
            print(f"ERROR: src[{i}] {src_roots[i].name} is not v3.0", file=sys.stderr)
            return 1

    # ---- 1) Unified task vocabulary
    unified: "OrderedDict[str, int]" = OrderedDict()
    src_taskidx_maps: List[Dict[int, int]] = []
    for s in src_roots:
        df = load_tasks(s)
        m: Dict[int, int] = {}
        for _, row in df.iterrows():
            text = str(row["task"]).strip()
            if text not in unified:
                unified[text] = len(unified)
            m[int(row["task_index"])] = unified[text]
        src_taskidx_maps.append(m)
    print(f"[merge] unified tasks: {len(unified)}")
    for t, i in unified.items():
        print(f"  {i:>2}  {t}")

    # ---- 2) Concat data parquets
    dst_data_dir = dst_root / "data" / "chunk-000"
    dst_data_dir.mkdir(parents=True, exist_ok=True)
    dst_episodes_dir = dst_root / "meta" / "episodes" / "chunk-000"
    dst_episodes_dir.mkdir(parents=True, exist_ok=True)
    dst_videos_root = dst_root / "videos"

    next_global_ep = 0
    running_idx = 0
    data_frames: List[pd.DataFrame] = []
    ep_meta_frames: List[pd.DataFrame] = []
    src_to_global_ep_map: List[Dict[int, int]] = []

    # video keys from first src
    info0 = src_infos[0]
    video_keys: List[str] = []
    for k, feat in info0.get("features", {}).items():
        if feat.get("dtype") == "video":
            video_keys.append(k)
    print(f"[merge] video keys: {video_keys}")

    for src_idx, src in enumerate(src_roots):
        info = src_infos[src_idx]
        n_eps = int(info["total_episodes"])
        print(f"\n[merge] src[{src_idx}] {src.name}: {n_eps} episodes")
        # data: read all chunks in source (usually just chunk-000/file-000)
        src_data_files = sorted((src / "data").rglob("file-*.parquet"))
        for f in src_data_files:
            df = pd.read_parquet(f)
            # remap episode_index globally
            old_eps = df["episode_index"].astype(int).values
            new_eps = old_eps + next_global_ep
            df["episode_index"] = new_eps.astype(np.int64)
            # remap task_index
            df["task_index"] = df["task_index"].astype(int).map(src_taskidx_maps[src_idx]).astype(np.int64)
            # reset global running index
            df["index"] = np.arange(running_idx, running_idx + len(df), dtype=np.int64)
            running_idx += len(df)
            data_frames.append(df)

        # episode metadata
        src_ep_files = sorted((src / "meta/episodes").rglob("file-*.parquet"))
        for f in src_ep_files:
            df = pd.read_parquet(f)
            old_eps = df["episode_index"].astype(int).values
            new_eps = old_eps + next_global_ep
            df["episode_index"] = new_eps.astype(np.int64)
            # rewrite data file pointer: every ep in this src maps to merged dst's
            # single data file at chunk-000/file-000
            df["data/chunk_index"] = np.zeros(len(df), dtype=np.int64)
            df["data/file_index"] = np.zeros(len(df), dtype=np.int64)
            # rewrite dataset_from/to_index to GLOBAL row positions in merged data
            # (compute per-row from the data we just appended)
            # Build a lookup: for each new ep, its global row range in merged data.
            ep_meta_frames.append(df)

        src_to_global_ep_map.append({i: i + next_global_ep for i in range(n_eps)})
        next_global_ep += n_eps

    # Write merged data parquet (single file)
    merged_data = pd.concat(data_frames, ignore_index=True)
    merged_data_pq = dst_data_dir / "file-000.parquet"
    merged_data.to_parquet(merged_data_pq, index=False)
    print(f"[merge] wrote {merged_data_pq} ({len(merged_data)} rows)")

    # Recompute dataset_from/to_index per episode using merged data
    ep_lens = merged_data.groupby("episode_index").size().to_dict()
    # cumulative row index per episode (sorted by episode_index ascending)
    sorted_eps = sorted(ep_lens.keys())
    cum = {}
    running = 0
    for ep in sorted_eps:
        cum[ep] = (running, running + ep_lens[ep])
        running += ep_lens[ep]

    # Update video pointers: each src's videos live at chunk-000/file-{src_idx}.mp4
    # in dst. We also reset the from_timestamp/to_timestamp as-is since the source
    # mp4 boundaries didn't change (we're just relocating, not concatenating).
    for src_idx, ep_df in enumerate(ep_meta_frames):
        for k in video_keys:
            ep_df[f"videos/{k}/chunk_index"] = np.zeros(len(ep_df), dtype=np.int64)
            ep_df[f"videos/{k}/file_index"] = np.full(len(ep_df), src_idx, dtype=np.int64)

    merged_ep_meta = pd.concat(ep_meta_frames, ignore_index=True)
    merged_ep_meta = merged_ep_meta.sort_values("episode_index").reset_index(drop=True)
    merged_ep_meta["dataset_from_index"] = merged_ep_meta["episode_index"].map(
        lambda e: cum[int(e)][0]).astype(np.int64)
    merged_ep_meta["dataset_to_index"] = merged_ep_meta["episode_index"].map(
        lambda e: cum[int(e)][1]).astype(np.int64)
    merged_ep_meta["length"] = merged_ep_meta["episode_index"].map(
        lambda e: ep_lens[int(e)]).astype(np.int64)
    merged_ep_meta["meta/episodes/chunk_index"] = np.zeros(len(merged_ep_meta), dtype=np.int64)
    merged_ep_meta["meta/episodes/file_index"] = np.zeros(len(merged_ep_meta), dtype=np.int64)
    merged_ep_pq = dst_episodes_dir / "file-000.parquet"
    merged_ep_meta.to_parquet(merged_ep_pq, index=False)
    print(f"[merge] wrote {merged_ep_pq}")

    # ---- 3) Copy / symlink videos
    for src_idx, src in enumerate(src_roots):
        for k in video_keys:
            src_mp4 = src / "videos" / k / "chunk-000" / "file-000.mp4"
            if not src_mp4.is_file():
                # Try alternate v3 video layout
                cands = list((src / "videos" / k).rglob("file-*.mp4"))
                if not cands:
                    print(f"  WARN: no mp4 for {k} in {src.name}")
                    continue
                src_mp4 = cands[0]
            dst_mp4_dir = dst_videos_root / k / "chunk-000"
            dst_mp4_dir.mkdir(parents=True, exist_ok=True)
            dst_mp4 = dst_mp4_dir / f"file-{src_idx:03d}.mp4"
            if dst_mp4.exists():
                dst_mp4.unlink()
            if args.copy:
                shutil.copyfile(src_mp4, dst_mp4)
            else:
                dst_mp4.symlink_to(src_mp4.resolve())
            print(f"  videos[{k}] src[{src_idx}] -> {dst_mp4.name}")

    # ---- 4) tasks.parquet
    tasks_rows = [{"task": t, "task_index": i} for t, i in unified.items()]
    tasks_df = pd.DataFrame(tasks_rows).set_index("task")
    tasks_df.to_parquet(dst_root / "meta/tasks.parquet")
    print(f"[merge] wrote tasks.parquet ({len(tasks_df)} tasks)")

    # ---- 5) info.json
    new_info = dict(src_infos[0])
    new_info["total_episodes"] = next_global_ep
    new_info["total_frames"] = int(running_idx)
    new_info["total_tasks"] = len(unified)
    new_info["total_chunks"] = 1
    new_info["chunks_size"] = max(next_global_ep, int(new_info.get("chunks_size", 1000)))
    if "splits" in new_info:
        new_info["splits"] = {"train": f"0:{next_global_ep}"}
    (dst_root / "meta/info.json").write_text(json.dumps(new_info, indent=2))
    print(f"[merge] wrote info.json")

    # ---- 6) stats.json — aggregate action/state from merged data
    act = np.stack(merged_data["action"].values).astype(np.float64)
    st = np.stack(merged_data["observation.state"].values).astype(np.float64)
    src_stats = json.loads((src_roots[0] / "meta/stats.json").read_text()) \
        if (src_roots[0] / "meta/stats.json").is_file() else {}
    new_stats = dict(src_stats)

    def _agg(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "count": int(arr.shape[0]),
        }

    new_stats["action"] = _agg(act)
    new_stats["observation.state"] = _agg(st)
    (dst_root / "meta/stats.json").write_text(json.dumps(new_stats, indent=2))
    print(f"[merge] wrote stats.json")

    print(f"\n[merge] DONE")
    print(f"  episodes : {next_global_ep}")
    print(f"  frames   : {running_idx}")
    print(f"  tasks    : {len(unified)}")
    print(f"  output   : {dst_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
