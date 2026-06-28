#!/usr/bin/env python3
"""Rule-based event annotator for swap/rotate-style sub-episodes.

For each sub-episode in a (v2.1) _sub LeRobot dataset:
  1. (Optional) Trigger extract_frames first to populate the frame_data tree at
     <out_root>/frames_<fps>hz/chunk-XXX/episode_NNNNNN/{table,wrist}/frame_*.jpg
     plus episode_NNNNNN.json template.
  2. Detect the release event from action[:, -1] gripper command:
        release_20hz = (last frame gripper > 0) + 1
        event_20hz   = release_20hz + settle_offset_20hz
        event_5hz    = event_20hz // (20 / fps)
  3. Write event_frame_idx + event_frame_idxs into each episode_NNNNNN.json.
  4. Refresh the top-level index.json with updated events block.

Boundary policy: assumes the source _sub dataset was split with midpoint
boundary (so the release frame is well inside the sub-episode, not at the end).

Usage:
  python tools/auto_event_annotate.py \
      --lerobot-root ~/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-100ep/task_07_sub \
      --out-root    ~/data/helm_data/frame_data/swap_2_bowls \
      --fps 5 \
      --settle-offset 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------- gripper / release detection ----------

def detect_release_20hz(gripper: np.ndarray, min_grasp_len: int = 5) -> Optional[int]:
    """Return the first 20Hz frame index after the last 'closed' segment.

    For a midpoint-boundary sub-episode the gripper signal is:
        open ... close (grasp) ... open (release) ... open (carry-to-next-grasp area)
    so we want the END of the first closed run + 1.
    """
    closed = gripper > 0.0
    T = len(closed)
    i = 0
    while i < T:
        if closed[i]:
            j = i
            while j < T and closed[j]:
                j += 1
            if j - i >= min_grasp_len:
                # release happens at j (first open frame after the run)
                return j
            i = j
        else:
            i += 1
    return None


# ---------- frame_data path helpers (must match helm_datasets.extract_frames) ----------

def episode_json_path(out_root: Path, fps_frames: int, chunk: str, ep_id: str) -> Path:
    return out_root / f"frames_{fps_frames}hz" / chunk / ep_id / f"{ep_id}.json"


def frames_dir(out_root: Path, fps_frames: int, chunk: str, ep_id: str, view: str) -> Path:
    return out_root / f"frames_{fps_frames}hz" / chunk / ep_id / view


# ---------- driver ----------

def run_extract_frames(lerobot_root: Path,
                        out_root: Path,
                        fps: int,
                        overwrite: bool,
                        python_bin: str) -> None:
    """Invoke helm_datasets.extract_frames via subprocess so we reuse its
    chunking / metadata logic. Camera keys are LIBERO-Mem fixed."""
    cmd = [
        python_bin, "-m", "helm_datasets.extract_frames",
        "--lerobot_root", str(lerobot_root),
        "--out_root", str(out_root),
        "--fps_frames", str(fps),
        "--exterior_key", "observation.images.base_0_rgb",
        "--wrist_key",    "observation.images.left_wrist_0_rgb",
        "--overwrite", "1" if overwrite else "0",
    ]
    print(f"[extract] {' '.join(cmd)}")
    repo_root = Path(__file__).resolve().parents[1]
    env_path = str(repo_root)
    subprocess.run(cmd, cwd=str(repo_root), env={"PYTHONPATH": env_path, **__import__("os").environ},
                   check=True)


def chunk_for_episode(lerobot_root: Path, ep_idx: int) -> str:
    """Find which video chunk holds this episode. We look at the videos/ tree."""
    for chunk_dir in sorted((lerobot_root / "videos").iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        # any video key works for existence check
        for key_dir in chunk_dir.iterdir():
            if not key_dir.is_dir():
                continue
            mp4 = key_dir / f"episode_{ep_idx:06d}.mp4"
            if mp4.exists():
                return chunk_dir.name
            break
    raise FileNotFoundError(f"no chunk containing episode {ep_idx} under {lerobot_root}/videos")


def list_episodes(lerobot_root: Path) -> List[int]:
    eps: List[int] = []
    for chunk_dir in (lerobot_root / "videos").iterdir():
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        # pick first video_key sub-dir
        sub_dirs = [d for d in chunk_dir.iterdir() if d.is_dir()]
        if not sub_dirs:
            continue
        for mp4 in sub_dirs[0].glob("episode_*.mp4"):
            try:
                eps.append(int(mp4.stem.replace("episode_", "")))
            except Exception:
                pass
    return sorted(set(eps))


def parquet_path(lerobot_root: Path, ep_idx: int, chunks_size: int) -> Path:
    chunk = ep_idx // chunks_size
    return lerobot_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot-root", required=True,
                   help="LeRobot v2.1 _sub dataset (e.g. task_07_sub)")
    p.add_argument("--out-root", required=True,
                   help="frame_data/<TASKSPEC> output dir")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--settle-offset", type=int, default=5,
                   help="(deprecated; see --event-at) 20Hz frames after release")
    p.add_argument("--event-at", choices=("midpoint", "release", "lifted-stable"),
                   default="midpoint",
                   help="midpoint = last 5Hz frame of sub-episode (=midpoint between "
                        "release and next grasp, given midpoint boundary_split). "
                        "release = release + settle_offset (legacy). "
                        "lifted-stable = window from 'state[gripper_l] fully open + "
                        "gripper-open command' (~release+0.5s) until 'xy velocity "
                        "exceeds xy_vel_threshold' (=start of motion toward next "
                        "subtask). Excludes the midpoint-overlap region.")
    p.add_argument("--grip-open-th", type=float, default=0.035,
                   help="state[gripper_l] threshold for 'fully open'. Default 0.035.")
    p.add_argument("--xy-vel-th", type=float, default=0.015,
                   help="per-frame xy displacement threshold (20Hz). Above this = "
                        "robot is moving toward next subtask, so end of event window. "
                        "Default 0.015 (=0.3 m/s).")
    p.add_argument("--episodes", nargs="+", type=int, default=None,
                   help="Limit to these episode indices")
    p.add_argument("--skip-extract", action="store_true",
                   help="Do not run extract_frames; assume frame_data is already populated")
    p.add_argument("--overwrite-frames", action="store_true",
                   help="Pass overwrite=1 to extract_frames (re-encode jpgs)")
    p.add_argument("--python-bin", default="/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python")
    args = p.parse_args()

    lerobot_root = Path(args.lerobot_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    # 1. extract frames + create base JSONs
    if not args.skip_extract:
        run_extract_frames(lerobot_root, out_root, args.fps,
                            overwrite=args.overwrite_frames,
                            python_bin=args.python_bin)
    else:
        print("[skip] extract_frames skipped per --skip-extract")

    # 2. enumerate sub-episodes
    eps = list_episodes(lerobot_root)
    if args.episodes is not None:
        eps = [e for e in eps if e in set(args.episodes)]
    print(f"[event] processing {len(eps)} sub-episodes")

    # chunks_size from info.json
    info = json.loads((lerobot_root / "meta" / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))

    twenty_to_five = 20.0 / args.fps  # frames in 20Hz per 5Hz frame
    now_iso = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z") or \
              dt.datetime.now().isoformat()

    events_index: Dict[str, Dict[str, Dict]] = {}
    no_release_ep: List[int] = []
    written = 0

    for ep_idx in eps:
        try:
            parq = parquet_path(lerobot_root, ep_idx, chunks_size)
            if not parq.exists():
                # find via direct glob (handles non-power-of-2 chunks_size)
                hit = list((lerobot_root / "data").glob(f"**/episode_{ep_idx:06d}.parquet"))
                if not hit:
                    print(f"  WARN ep {ep_idx}: parquet not found")
                    continue
                parq = hit[0]
            df = pd.read_parquet(parq)
            actions = np.stack(df["action"].values).astype(np.float32)
            states = np.stack(df["observation.state"].values).astype(np.float32)
            T_20hz = len(actions)
            release = detect_release_20hz(actions[:, -1])
            if release is None:
                no_release_ep.append(ep_idx)
                continue

            # locate target episode JSON
            chunk = chunk_for_episode(lerobot_root, ep_idx)
            ep_id = f"episode_{ep_idx:06d}"
            json_p = episode_json_path(out_root, args.fps, chunk, ep_id)
            if not json_p.exists():
                print(f"  WARN ep {ep_idx}: episode json missing at {json_p} "
                      f"(was extract_frames run?)")
                continue

            data = json.loads(json_p.read_text())
            n_5hz = int(data.get("n_frames", 0))

            if args.event_at == "midpoint":
                # last 5Hz frame of the sub-episode = midpoint of (release, next_grasp)
                # because the source _sub dataset was split with midpoint boundary.
                event_5hz = max(0, n_5hz - 1)
                event_5hz_end = n_5hz  # inclusive of last frame
            elif args.event_at == "release":
                event_20hz = release + args.settle_offset
                event_20hz = min(event_20hz, T_20hz - 1)
                event_5hz = int(event_20hz // twenty_to_five)
                event_5hz = min(event_5hz, max(0, n_5hz - 1))
                event_5hz_end = n_5hz  # legacy: all frames from event to end are pos
            else:  # lifted-stable
                # start: first 20Hz frame ≥ release where state[gripper_l] ≥ threshold
                # AND action[-1] < 0 (gripper-open command). ≈ release + 10 frames.
                start_20hz = None
                for t in range(release, T_20hz):
                    if states[t, 6] >= args.grip_open_th and actions[t, -1] < 0:
                        start_20hz = t; break
                if start_20hz is None:
                    no_release_ep.append(ep_idx)
                    continue
                # end: first 20Hz frame > start where |xy(t) - xy(t-1)| ≥ xy_vel_th
                # (=robot has begun moving toward next target). Pos pool covers frames
                # in [start, end) — the "lifted but xy-stationary" stretch.
                end_20hz = T_20hz - 1
                for t in range(start_20hz + 1, T_20hz):
                    if np.linalg.norm(states[t, :2] - states[t-1, :2]) >= args.xy_vel_th:
                        end_20hz = t; break
                event_5hz = int(start_20hz // twenty_to_five)
                event_5hz = min(event_5hz, max(0, n_5hz - 1))
                # end of pos window (exclusive) in 5Hz; cap by n_5hz
                event_5hz_end = min(n_5hz, int(end_20hz // twenty_to_five) + 1)
                event_5hz_end = max(event_5hz_end, event_5hz + 1)

            data["event_frame_idx"] = event_5hz
            data["event_frame_idxs"] = list(range(event_5hz, event_5hz_end))

            json_p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")

            events_index.setdefault(chunk, {})[ep_id] = {
                "event_frame_idx": event_5hz,
                "n_frames": n_5hz,
                "updated_at": now_iso,
            }
            written += 1
        except Exception as e:
            print(f"  ERROR ep {ep_idx}: {e}")

    # 3. write/merge index.json
    idx_p = out_root / "index.json"
    base = {}
    if idx_p.exists():
        try:
            base = json.loads(idx_p.read_text())
        except Exception:
            base = {}
    base.setdefault("events", {})
    for chunk, mp in events_index.items():
        base["events"].setdefault(chunk, {}).update(mp)
    idx_p.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    print(f"\n[event] wrote event_frame_idx for {written} sub-episodes")
    if no_release_ep:
        print(f"[event] {len(no_release_ep)} episodes had no detectable grasp/release: "
              f"{no_release_ep[:10]}{'...' if len(no_release_ep) > 10 else ''}")
    print(f"[event] index.json -> {idx_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
