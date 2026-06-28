"""
Convert LIBERO-Mem (full HDF5 with rendered images) to HeLM-flavored LeRobot v2 format.

Source schema (per LIBERO_10_dataset_builder.py pattern, applies to LIBERO-Mem):

    data/
      demo_N/
        actions:       (T, 7) float                 # 6-DoF EE delta + gripper
        states:        (T, 58) float                # full mujoco state, ignored
        init_state:    (58,)  float                 # ignored (eval-only field)
        obs/
          ee_states:       (T, 7)  float            # eef_pos(3) + eef_quat(4)
          gripper_states:  (T, 2)  float            # left/right gripper qpos
          joint_states:    (T, 7)  float            # ignored
          agentview_rgb:    (T, H, W, 3) uint8      # exterior camera
          eye_in_hand_rgb:  (T, H, W, 3) uint8      # wrist camera

Output schema:

    <out>/
      meta/info.json, episodes.jsonl, tasks.jsonl
      data/chunk-XXX/episode_NNNNNN.parquet
        columns: action, observation.state, timestamp,
                 frame_index, episode_index, index, task_index
      videos/chunk-XXX/observation.images.image/episode_NNNNNN.mp4
      videos/chunk-XXX/observation.images.wrist_image/episode_NNNNNN.mp4

Usage:
    # convert everything
    python tools/libero_to_lerobot.py \
        --src ~/data/LIBERO-Mem \
        --out ~/data/LIBERO-Mem/LIBERO-Mem-LeRobot

    # quick smoke (1 task, 5 demos)
    python tools/libero_to_lerobot.py \
        --src ~/data/LIBERO-Mem \
        --out /tmp/libero_smoke \
        --limit-tasks 1 --limit-demos 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----- constants -----

FPS = 20                # LIBERO control freq (env_info.control_freq=20)
CHUNK_SIZE = 50         # episodes per chunk dir, matches HeLM piper convention

# (HDF5 obs key, LeRobot output feature key)
CAMERAS: List[Tuple[str, str]] = [
    ("agentview_rgb",   "observation.images.image"),
    ("eye_in_hand_rgb", "observation.images.wrist_image"),
]


# ----- helpers -----

def parse_language(filename: str) -> str:
    """Extract task language from LIBERO HDF5 filename.

    >>> parse_language('KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times_demo.hdf5')
    'lift the bowl and place it back on the plate 3 times'
    """
    base = os.path.basename(filename)
    base = re.sub(r"_demo\.hdf5$", "", base)
    m = re.match(r"^[A-Z_]+SCENE\d+_\d+_(.+)$", base)
    return (m.group(1) if m else base).replace("_", " ")


def list_demos(h5: h5py.File) -> List[str]:
    """Return demo keys sorted by trailing numeric index."""
    keys = list(h5["data"].keys())
    keys.sort(key=lambda x: int(x.split("_")[-1]))
    return keys


def write_video(out_root: Path, episode_idx: int, video_key: str,
                frames: np.ndarray, rotate_180: bool = True) -> None:
    """Encode (T, H, W, 3) uint8 frames as mp4 (libx264, yuv420p)."""
    chunk = episode_idx // CHUNK_SIZE
    out = (out_root / "videos" / f"chunk-{chunk:03d}" / video_key
           / f"episode_{episode_idx:06d}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    if rotate_180:
        frames = frames[:, ::-1, ::-1, :]

    writer = imageio.get_writer(
        str(out), fps=FPS, codec="libx264",
        pixelformat="yuv420p", quality=8, macro_block_size=1,
    )
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def write_parquet(out_root: Path, episode_idx: int, task_idx: int,
                  actions: np.ndarray, state: np.ndarray, index_offset: int) -> int:
    n = len(actions)
    chunk = episode_idx // CHUNK_SIZE
    out = (out_root / "data" / f"chunk-{chunk:03d}"
           / f"episode_{episode_idx:06d}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "action":            [r.tolist() for r in actions.astype(np.float32)],
        "observation.state": [r.tolist() for r in state.astype(np.float32)],
        "timestamp":     (np.arange(n, dtype=np.float32) / FPS),
        "frame_index":    np.arange(n, dtype=np.int64),
        "episode_index":  np.full(n, episode_idx, dtype=np.int64),
        "task_index":     np.full(n, task_idx, dtype=np.int64),
        "index":          np.arange(n, dtype=np.int64) + index_offset,
    })
    df.to_parquet(out, engine="pyarrow")
    return n


def write_meta(out_root: Path, total_episodes: int, total_frames: int,
               tasks_dict: dict[int, str], image_shape: Tuple[int, int, int],
               state_dim: int, action_dim: int) -> None:
    meta = out_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    h, w, c = image_shape
    image_feature = {
        "dtype": "video",
        "shape": [h, w, c],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": float(FPS),
            "video.height": int(h),
            "video.width": int(w),
            "video.channels": int(c),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }

    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks_dict),
        "total_videos": total_episodes * len(CAMERAS),
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"][:action_dim],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": ["x", "y", "z", "qx", "qy", "qz", "qw",
                          "gripper_l", "gripper_r"][:state_dim],
            },
            **{lerobot_key: image_feature for _, lerobot_key in CAMERAS},
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        },
    }
    with open(meta / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(meta / "tasks.jsonl", "w") as f:
        for ti in sorted(tasks_dict.keys()):
            f.write(json.dumps({"task_index": int(ti),
                                "task": tasks_dict[ti]}) + "\n")


def write_episodes(out_root: Path, records: List[dict]) -> None:
    with open(out_root / "meta" / "episodes.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ----- main -----

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Dir containing LIBERO-Mem *.hdf5 files")
    ap.add_argument("--out", required=True,
                    help="Output LeRobot dataset directory")
    ap.add_argument("--limit-tasks", type=int, default=None,
                    help="Only convert first N task files (for smoke testing)")
    ap.add_argument("--limit-demos", type=int, default=None,
                    help="Only convert first N demos per task")
    ap.add_argument("--no-rotate", action="store_true",
                    help="Skip 180° image rotation (default: rotate to match LIBERO viewer convention)")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    if not src.is_dir():
        print(f"src not found: {src}", file=sys.stderr); return 1

    files = sorted(p for p in src.glob("*.hdf5") if p.name.endswith("_demo.hdf5"))
    if args.limit_tasks is not None:
        files = files[: args.limit_tasks]
    if not files:
        print(f"no *_demo.hdf5 files in {src}", file=sys.stderr); return 1

    print(f"found {len(files)} task file(s) in {src}")
    out.mkdir(parents=True, exist_ok=True)

    # ----- detect schema from first demo -----
    with h5py.File(files[0], "r") as f:
        demos = list_demos(f)
        d0 = f["data"][demos[0]]
        if "obs" not in d0:
            print(f"ERROR: '{files[0]}' has no obs/ subgroup. This looks "
                  f"like LIBERO-Mem-Raw (no images). Use the full LIBERO-Mem.",
                  file=sys.stderr)
            return 2
        cam0 = CAMERAS[0][0]
        if cam0 not in d0["obs"]:
            print(f"ERROR: '{cam0}' not found in obs/ of {files[0]}",
                  file=sys.stderr)
            print(f"  available obs keys: {list(d0['obs'].keys())}", file=sys.stderr)
            return 2
        sample_img = d0["obs"][cam0][0]
        image_shape = tuple(int(x) for x in sample_img.shape)  # (H, W, C)
        ee_dim = int(d0["obs"]["ee_states"].shape[1])
        gr_dim = int(d0["obs"]["gripper_states"].shape[1])
        state_dim = ee_dim + gr_dim
        action_dim = int(d0["actions"].shape[1])
    print(f"detected: image_shape={image_shape}, state_dim={state_dim}, "
          f"action_dim={action_dim}, fps={FPS}")

    # ----- convert -----
    tasks_dict: dict[int, str] = {}
    episode_records: List[dict] = []
    global_ep = 0
    global_idx = 0

    for task_idx, fp in enumerate(files):
        language = parse_language(fp.name)
        tasks_dict[task_idx] = language
        print(f"\n[{task_idx:>2}] {fp.name}\n      → {language!r}")

        with h5py.File(fp, "r") as f:
            demos = list_demos(f)
            if args.limit_demos is not None:
                demos = demos[: args.limit_demos]

            for demo_key in tqdm(demos, desc=f"task {task_idx}", leave=False):
                d = f["data"][demo_key]
                actions = d["actions"][()]
                state = np.concatenate(
                    [d["obs"]["ee_states"][()], d["obs"]["gripper_states"][()]],
                    axis=-1,
                )

                for hdf_key, lerobot_key in CAMERAS:
                    write_video(out, global_ep, lerobot_key,
                                d["obs"][hdf_key][()],
                                rotate_180=not args.no_rotate)

                n = write_parquet(out, global_ep, task_idx,
                                  actions, state, global_idx)

                episode_records.append({
                    "episode_index": int(global_ep),
                    "tasks": [language],
                    "length": int(n),
                })
                global_ep += 1
                global_idx += n

    # ----- meta files -----
    write_meta(out, total_episodes=global_ep, total_frames=global_idx,
               tasks_dict=tasks_dict, image_shape=image_shape,
               state_dim=state_dim, action_dim=action_dim)
    write_episodes(out, episode_records)

    print(f"\n✓ converted {global_ep} episodes ({global_idx} frames) "
          f"from {len(files)} task(s)")
    print(f"  → {out}")
    print(f"\nNext: compute dataset stats (mean/std for state/action/images).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
