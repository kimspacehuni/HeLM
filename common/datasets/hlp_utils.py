"""Image and label validation shared by the HLP datasets."""
from pathlib import Path

from PIL import Image
import torch


def load_hlp_images(item, num_images):
    if num_images not in (1, 2):
        raise ValueError("num_images must be 1 or 2")
    uid = item.get("uid", "unknown")
    paths = item.get("images")
    if not isinstance(paths, dict):
        raise ValueError(f"[{uid}] images must be a dictionary")
    images = []
    for camera in ("table", "wrist")[:num_images]:
        path = paths.get(camera)
        if not path:
            raise ValueError(f"[{uid}] missing {camera} image path")
        try:
            with Image.open(Path(path)) as image:
                images.append(image.convert("RGB"))
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"[{uid}] cannot load {camera} image: {path}") from error
    return images


def mask_assistant_tokens(input_ids, prompt_ids, uid="unknown"):
    """Supervise only the response after the processor's exact prompt prefix."""
    prompt_length = prompt_ids.numel()
    if prompt_length >= input_ids.numel():
        raise ValueError(f"[{uid}] assistant response has no tokens")
    if not torch.equal(input_ids[:prompt_length], prompt_ids):
        raise ValueError(f"[{uid}] prompt tokens are not a prefix of the full conversation")
    labels = input_ids.clone()
    labels[:prompt_length] = -100
    return labels


def require_hlp_batch(features):
    if not features or any(feature is None for feature in features):
        raise ValueError("HLP batch is empty or contains a missing sample; validate input images first")
