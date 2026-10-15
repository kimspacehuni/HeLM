# helm_dataset.py (SmolVLM 버전)
from __future__ import annotations

import json
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from transformers import AutoProcessor
from common.datasets.hlp_utils import load_hlp_images as _load_images, mask_assistant_tokens, require_hlp_batch
from torch.utils.data import Dataset


def read_jsonl(path_or_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path_or_dir)
    rows: List[Dict[str, Any]] = []
    if p.is_file():
        files = [p]
    else:
        files = sorted(p.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found under: {p}")

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise RuntimeError(f"JSON parse error: {fp} line {ln}: {e}")
    return rows


@dataclass
class SmolVLMDatasetConfig:
    jsonl_path: str
    model_name_or_path: str
    num_images: int = 1
    trust_remote_code: bool = True
    use_fast: bool = True
    padding_side: str = "left"


class HelmJsonlDatasetSmolVLM(Dataset):
    def __init__(self, cfg: SmolVLMDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.rows = read_jsonl(cfg.jsonl_path)
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
            use_fast=cfg.use_fast,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = cfg.padding_side

        self.pools: Dict[str, List[int]] = {}
        for i, r in enumerate(self.rows):
            lab = str(r.get("label", "UNKNOWN"))
            self.pools.setdefault(lab, []).append(i)

    def __len__(self) -> int:
        return len(self.rows)

    def get_pools(self) -> Dict[str, List[int]]:
        return {k: list(v) for k, v in self.pools.items()}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.rows[idx]
        user_prompt = str(item.get("user_prompt", ""))
        target_text = str(item.get("gt_text", ""))
        if not target_text.strip():
            raise ValueError(f"[{item.get('uid', 'unknown')}] gt_text is empty")

        imgs = _load_images(item, self.cfg.num_images)

        # 1. 전체 메시지 구성
        user_content = [{"type": "image"}] * self.cfg.num_images
        user_content.append({"type": "text", "text": user_prompt})

        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
        ]

        # 2. 질문만 포함된 메시지 구성 (정확한 마스킹 위치 계산용)
        prompt_messages = [
            {"role": "user", "content": user_content},
        ]

        # 3. 각각 템플릿 적용
        full_string = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_string = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # 4. 토큰화
        model_inputs = self.processor(text=full_string, images=imgs, return_tensors="pt", padding=False)
        prompt_inputs = self.processor(text=prompt_string, images=imgs, return_tensors="pt", padding=False)

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)

        labels = mask_assistant_tokens(
            input_ids, prompt_inputs["input_ids"].squeeze(0), item.get("uid", "unknown"),
        )

        pixel_values = model_inputs["pixel_values"].squeeze(0)
        pixel_attention_mask = model_inputs.get("pixel_attention_mask")
        if pixel_attention_mask is not None:
            pixel_attention_mask = pixel_attention_mask.squeeze(0)

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
        if pixel_attention_mask is not None:
            res["pixel_attention_mask"] = pixel_attention_mask
        return res


class DataCollatorForSmolVLM:
    def __init__(self, processor: AutoProcessor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def _left_pad_1d(self, tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(t.size(0) for t in tensors)
        out = tensors[0].new_full((len(tensors), max_len), pad_value)
        for i, t in enumerate(tensors):
            out[i, -t.size(0):] = t
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        require_hlp_batch(features)

        pad_id = self.tokenizer.pad_token_id
        input_ids = self._left_pad_1d([f["input_ids"] for f in features], pad_id)
        attention_mask = self._left_pad_1d([f["attention_mask"] for f in features], 0)
        labels = self._left_pad_1d([f["labels"] for f in features], -100)

        pixel_values = torch.stack([f["pixel_values"] for f in features], dim=0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }

        if "pixel_attention_mask" in features[0]:
            pixel_attention_mask = torch.stack([f["pixel_attention_mask"] for f in features], dim=0)
            batch["pixel_attention_mask"] = pixel_attention_mask

        return batch