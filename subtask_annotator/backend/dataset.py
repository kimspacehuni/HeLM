"""lerobot v2.1 dataset reader (no lerobot dependency, pandas-only)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EpisodeMeta:
    episode_index: int
    length: int
    tasks: list[str]


@dataclass
class DatasetInfo:
    codebase_version: str
    fps: int
    chunks_size: int
    data_path: str
    video_path: str
    video_keys: list[str]
    total_episodes: int


class LerobotV21Dataset:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.meta_dir = self.root / "meta"

        info = json.loads((self.meta_dir / "info.json").read_text())
        video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
        self.info = DatasetInfo(
            codebase_version=info["codebase_version"],
            fps=int(info["fps"]),
            chunks_size=int(info["chunks_size"]),
            data_path=info["data_path"],
            video_path=info["video_path"],
            video_keys=video_keys,
            total_episodes=int(info["total_episodes"]),
        )

        episodes: dict[int, EpisodeMeta] = {}
        with (self.meta_dir / "episodes.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                ep = EpisodeMeta(
                    episode_index=int(row["episode_index"]),
                    length=int(row["length"]),
                    tasks=list(row.get("tasks", [])),
                )
                episodes[ep.episode_index] = ep
        self.episodes = episodes

    def episode_chunk(self, episode_index: int) -> int:
        return episode_index // self.info.chunks_size

    def video_path(self, episode_index: int, video_key: str) -> Path:
        rel = self.info.video_path.format(
            episode_chunk=self.episode_chunk(episode_index),
            video_key=video_key,
            episode_index=episode_index,
        )
        return self.root / rel

    def data_path(self, episode_index: int) -> Path:
        rel = self.info.data_path.format(
            episode_chunk=self.episode_chunk(episode_index),
            episode_index=episode_index,
        )
        return self.root / rel

    def list_episodes(self) -> list[EpisodeMeta]:
        return [self.episodes[i] for i in sorted(self.episodes)]

    def get_episode(self, episode_index: int) -> EpisodeMeta:
        return self.episodes[episode_index]
