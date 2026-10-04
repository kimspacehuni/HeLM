"""
Rename image feature keys in a LeRobot dataset (no re-encoding).

pi0_base ckpt was trained with feature names:
    observation.images.base_0_rgb         (exterior camera)
    observation.images.left_wrist_0_rgb   (wrist camera)
    observation.images.right_wrist_0_rgb  (third camera, missing → zero-pad)

Our LIBERO-Mem-LeRobot was written with the OpenVLA-style names:
    observation.images.image         (= agentview, exterior)
    observation.images.wrist_image   (= eye-in-hand, wrist)

For fine-tuning to inherit pi0_base's vision input projection weights, the
dataset's feature keys must match what the policy expects. This script renames
both the mp4 directory names and the keys in meta/info.json.

Default mapping:
    image        → base_0_rgb
    wrist_image  → left_wrist_0_rgb

Usage:
    # one dataset (unified)
    python tools/rename_lerobot_image_keys.py --root ~/data/LIBERO-Mem/LIBERO-Mem-LeRobot

    # one per-task dataset
    python tools/rename_lerobot_image_keys.py \
        --root ~/data/LIBERO-Mem/LIBERO-Mem-LeRobot-PerTask/task_01_pick_up_the_bowl

    # all per-task datasets at once
    for d in ~/data/LIBERO-Mem/LIBERO-Mem-LeRobot-PerTask/*/; do
        python tools/rename_lerobot_image_keys.py --root "$d"
    done

    # custom mapping (override defaults)
    python tools/rename_lerobot_image_keys.py --root ... \
        --map image=base_0_rgb wrist_image=my_custom_wrist
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


DEFAULT_MAP = {
    "image": "base_0_rgb",
    "wrist_image": "left_wrist_0_rgb",
}

PREFIX = "observation.images."


def parse_map(items: list[str] | None) -> Dict[str, str]:
    if not items:
        return DEFAULT_MAP
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--map item must be 'old=new', got {it!r}")
        old, new = it.split("=", 1)
        out[old.strip()] = new.strip()
    return out


def rename_dataset(root: Path, mapping: Dict[str, str], dry_run: bool) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        print(f"meta/info.json not found at {info_path}", file=sys.stderr); return 1

    info = json.loads(info_path.read_text())
    features = info.get("features", {})

    # Build full-key (with PREFIX) mapping, retaining only keys that actually exist.
    full_map: Dict[str, str] = {}
    for short_old, short_new in mapping.items():
        old_full = f"{PREFIX}{short_old}"
        new_full = f"{PREFIX}{short_new}"
        if old_full in features:
            full_map[old_full] = new_full
        else:
            print(f"  (skip) {old_full} not in info.features")

    if not full_map:
        print(f"nothing to rename at {root}")
        return 0

    print(f"\nrename plan for {root}:")
    for o, n in full_map.items():
        print(f"  {o}  →  {n}")
    if dry_run:
        print("\n[dry-run] no changes written")
        return 0

    # ---- 1) rename video subdirectories under videos/chunk-XXX/ ----
    videos_root = root / "videos"
    if videos_root.is_dir():
        for chunk_dir in sorted(videos_root.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for old_full, new_full in full_map.items():
                src = chunk_dir / old_full
                dst = chunk_dir / new_full
                if src.is_dir():
                    if dst.exists():
                        print(f"  ! target exists, skipping: {dst}", file=sys.stderr)
                        continue
                    src.rename(dst)
                    print(f"  mv {src.relative_to(root)} → {dst.relative_to(root)}")

    # ---- 2) rewrite meta/info.json: features dict + video_path template ----
    new_features = {}
    for k, v in features.items():
        new_features[full_map.get(k, k)] = v
    info["features"] = new_features
    # video_path template uses {video_key} so doesn't need rewriting
    info_path.write_text(json.dumps(info, indent=2))
    print(f"  ✓ updated meta/info.json")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LeRobot dataset root")
    ap.add_argument("--map", nargs="*",
                    help="Custom mapping items 'old=new' (default: image=base_0_rgb wrist_image=left_wrist_0_rgb)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan only; no file moves or info.json writes")
    args = ap.parse_args()
    return rename_dataset(
        root=Path(args.root).expanduser(),
        mapping=parse_map(args.map),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
