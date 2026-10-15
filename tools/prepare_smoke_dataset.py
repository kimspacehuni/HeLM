"""Create a separate, normalized LeRobot subset for local smoke checks.

The source is read-only. This fixture repairs subset metadata and numeric indices;
it retains source image statistics and is intended for loader checks, not training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def prepare_smoke_dataset(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Destination must be separate from the source dataset")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    info = json.loads((source / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    episodes.sort(key=lambda row: row["episode_index"])
    original_stats = json.loads((source / "meta/stats.json").read_text())
    tasks, normalized_episodes, tables = [], [], []
    offset = 0
    chunks_size = info["chunks_size"]
    destination.mkdir(parents=True)
    for index, episode in enumerate(episodes):
        original_index = episode["episode_index"]
        original_chunk = original_index // chunks_size
        chunk = index // chunks_size
        task = episode["tasks"]
        if isinstance(task, list):
            if len(task) != 1:
                raise ValueError("Smoke fixture requires one task text per episode")
            task = task[0]
        if task not in tasks:
            tasks.append(task)
        original_path = info["data_path"].format(episode_chunk=original_chunk, episode_index=original_index)
        table = pd.read_parquet(source / original_path)
        if len(table) != episode["length"]:
            raise ValueError(f"Episode {original_index} length differs from metadata")
        table["frame_index"] = np.arange(len(table), dtype=np.int64)
        table["index"] = np.arange(offset, offset + len(table), dtype=np.int64)
        table["episode_index"] = np.full(len(table), index, dtype=np.int64)
        table["task_index"] = np.full(len(table), tasks.index(task), dtype=np.int64)
        target = destination / info["data_path"].format(episode_chunk=chunk, episode_index=index)
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target, index=False)
        for key, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            src = source / info["video_path"].format(episode_chunk=original_chunk, episode_index=original_index, video_key=key)
            dst = destination / info["video_path"].format(episode_chunk=chunk, episode_index=index, video_key=key)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        normalized_episodes.append({"episode_index": index, "tasks": [task], "length": len(table)})
        tables.append(table)
        offset += len(table)

    combined = pd.concat(tables, ignore_index=True)
    stats = {key: value for key, value in original_stats.items()
             if key in info["features"] and info["features"][key]["dtype"] == "video"}
    for key in combined.columns:
        values = np.stack(combined[key].to_numpy())
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {key}")
        feature_shape = list(values.shape[1:]) or [1]
        info["features"].setdefault(key, {"dtype": str(values.dtype), "shape": feature_shape, "names": None})
        keepdims = values.ndim == 1
        stats[key] = {name: function(values, axis=0, keepdims=keepdims).tolist()
                      for name, function in [("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)]}
    info.update(total_episodes=len(episodes), total_frames=offset, total_tasks=len(tasks),
                total_videos=len(episodes) * sum(feature["dtype"] == "video" for feature in info["features"].values()),
                total_chunks=(len(episodes) + chunks_size - 1) // chunks_size,
                splits={"train": f"0:{len(episodes)}"})
    meta = destination / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    (meta / "stats.json").write_text(json.dumps(stats, indent=2))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in normalized_episodes))
    (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": index, "task": task}) + "\n" for index, task in enumerate(tasks)))
    report = {"source": str(source), "destination": str(destination), "episodes": len(episodes),
              "frames": offset, "tasks": len(tasks), "image_statistics": "retained from source; smoke fixture only"}
    (destination / "smoke_fixture.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_smoke_dataset(args.source, args.destination), indent=2))
