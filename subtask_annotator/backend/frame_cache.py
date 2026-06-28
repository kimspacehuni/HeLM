"""On-demand frame extraction via ffmpeg with LRU cache."""
from __future__ import annotations

import subprocess
import threading
from collections import OrderedDict
from pathlib import Path


class FrameCache:
    def __init__(self, fps: int, jpeg_quality: int = 4, max_entries: int = 512) -> None:
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.max_entries = max_entries
        self._store: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._lock = threading.Lock()

    def _get_cached(self, key: tuple[str, int]) -> bytes | None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            return None

    def _put(self, key: tuple[str, int], data: bytes) -> None:
        with self._lock:
            self._store[key] = data
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def get(self, video_path: Path, frame_index: int) -> bytes:
        key = (str(video_path), frame_index)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        ts = frame_index / float(self.fps)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{ts:.6f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", str(self.jpeg_quality),
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0 or not result.stdout:
            raise RuntimeError(
                f"ffmpeg failed for {video_path} frame={frame_index}: {result.stderr.decode(errors='replace')}"
            )
        data = result.stdout
        self._put(key, data)
        return data
