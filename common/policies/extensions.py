from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ExtendedConfig:
    core: str = "vanilla"
    target_keywords: list[str] = field(default_factory=lambda: ["all-linear"])
    pretrained_expert: bool = False

    adapter_file_path: Optional[list[str | Path]] = None
    aux_loss_cfg: Optional[dict] = None
    is_train: bool = True

    expert_source: Optional[str] = "lora"
