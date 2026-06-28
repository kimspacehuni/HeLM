"""
Minimal FastAPI server that wraps HeLM's PI0Policy for remote inference.

Run inside the HeLM env (with pi0 deps available) on a GPU worker node:

    PYTHONPATH=$(pwd) \
    python tools/helm_pi0_server.py \
        --ckpt ~/ckpt/pi0_base \
        --device cuda:0 \
        --host 0.0.0.0 \
        --port 8000

The libero-side client posts JSON observations and receives one action per call.
The server keeps a per-process action queue; clients call POST /reset before each
new episode.
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# Ensure HeLM root is on PYTHONPATH so common.* imports resolve when the script
# is run with `python tools/helm_pi0_server.py` from the project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.policies.pi0.modeling_pi0 import PI0Policy  # noqa: E402

logger = logging.getLogger("helm_pi0_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------- Request/response models ----------

class PredictRequest(BaseModel):
    task: str
    state: List[float]
    images: Dict[str, str]  # key -> base64-encoded PNG/JPEG bytes


class PredictResponse(BaseModel):
    action: List[float]
    queue_size_after: int
    latency_ms: float


class ResetResponse(BaseModel):
    ok: bool


class HealthResponse(BaseModel):
    status: str
    device: str
    image_keys: List[str]
    state_dim: int
    action_dim: int
    queue_size: int


# ---------- Globals (single model per process) ----------

_POLICY: Optional[PI0Policy] = None
_DEVICE: Optional[torch.device] = None
_IMAGE_KEYS: List[str] = []
_STATE_DIM: int = 0
_ACTION_DIM: int = 0
_OBS_PREFIX = "observation.images."


# ---------- Helpers ----------

def _decode_image(b64: str) -> torch.Tensor:
    """base64 PNG/JPEG bytes → float tensor [1, 3, H, W] in [0, 1]."""
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H W C
    arr = np.transpose(arr, (2, 0, 1))               # C H W
    return torch.from_numpy(arr).unsqueeze(0)        # 1 C H W


def _build_batch(req: PredictRequest) -> Dict[str, torch.Tensor]:
    assert _DEVICE is not None
    batch: Dict[str, torch.Tensor] = {}

    # Images. Accept short keys like "base_0_rgb" or fully-qualified
    # "observation.images.base_0_rgb".
    for k, b64 in req.images.items():
        full_key = k if k.startswith(_OBS_PREFIX) else f"{_OBS_PREFIX}{k}"
        if full_key not in _IMAGE_KEYS:
            logger.warning("Ignoring unrecognized image key %s (expected one of %s)", full_key, _IMAGE_KEYS)
            continue
        batch[full_key] = _decode_image(b64).to(_DEVICE, non_blocking=True)

    # State. Normalize is applied BEFORE prepare_state (which pads), so the
    # client-side raw state dim must already match the model's max_state_dim.
    # We pad with zeros here so callers can send the natural robot state size.
    state = torch.tensor(req.state, dtype=torch.float32)  # [D]
    if _STATE_DIM > 0 and state.shape[0] < _STATE_DIM:
        state = torch.nn.functional.pad(
            state, (0, _STATE_DIM - state.shape[0]), value=0.0
        )
    elif _STATE_DIM > 0 and state.shape[0] > _STATE_DIM:
        logger.warning(
            "state dim %d exceeds model max %d; truncating",
            state.shape[0], _STATE_DIM,
        )
        state = state[:_STATE_DIM]
    batch["observation.state"] = state.unsqueeze(0).to(_DEVICE, non_blocking=True)
    batch["task"] = [req.task]
    return batch


# ---------- App ----------

app = FastAPI(title="HeLM pi0 inference server")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _POLICY is None or _DEVICE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        device=str(_DEVICE),
        image_keys=_IMAGE_KEYS,
        state_dim=_STATE_DIM,
        action_dim=_ACTION_DIM,
        queue_size=len(_POLICY._action_queue),
    )


@app.post("/reset", response_model=ResetResponse)
def reset() -> ResetResponse:
    if _POLICY is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    _POLICY.reset()
    return ResetResponse(ok=True)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _POLICY is None or _DEVICE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        batch = _build_batch(req)
        with torch.no_grad():
            action = _POLICY.select_action(batch).squeeze().detach().cpu().tolist()
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    # Truncate padded action down to the configured action dim (e.g. 32 → 7 for LIBERO).
    action = action[:_ACTION_DIM] if _ACTION_DIM else action
    return PredictResponse(
        action=action,
        queue_size_after=len(_POLICY._action_queue),
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


# ---------- Boot ----------

def _fill_identity_for_inf_stats(model: torch.nn.Module) -> int:
    """
    pi0_base ships without normalization stats (its policy_preprocessor.json
    has empty `features: {}`), so the Normalize/Unnormalize stats stay at the
    +inf sentinel after `from_pretrained`. For smoke-testing the eval loop we
    replace any inf entries with identity stats:
        mean=0, std=1   → MEAN_STD becomes a no-op
        min=-1, max=1   → MIN_MAX  becomes a no-op
    Real evaluation/training must overwrite these with stats from the actual
    training dataset. Tensors already populated (i.e. not inf) are untouched.

    Note: in the current Normalize implementation these are nn.Parameters
    (with requires_grad=False) wrapped in a ParameterDict, NOT torch buffers.
    We walk both named_parameters() and named_buffers() to be robust to
    either implementation.
    """
    def _walk():
        for n, p in model.named_parameters():
            yield n, p
        for n, b in model.named_buffers():
            yield n, b

    n_fixed = 0
    for name, t in _walk():
        if not torch.isinf(t).any():
            continue
        if name.endswith(".mean"):
            t.data.zero_()
        elif name.endswith(".std"):
            t.data.fill_(1.0)
        elif name.endswith(".min"):
            t.data.fill_(-1.0)
        elif name.endswith(".max"):
            t.data.fill_(1.0)
        else:
            continue
        n_fixed += 1
        logger.warning("[stats] filled identity values for inf entry: %s", name)
    return n_fixed


def _load_policy(ckpt: str, device: str) -> None:
    global _POLICY, _DEVICE, _IMAGE_KEYS, _STATE_DIM, _ACTION_DIM

    ckpt = os.path.expanduser(ckpt)
    logger.info("Loading PI0Policy from %s on %s ...", ckpt, device)
    t0 = time.time()

    _DEVICE = torch.device(device)
    policy = PI0Policy.from_pretrained(ckpt)

    # Patch missing normalization stats so inference doesn't trip the
    # `mean is infinity` assert. See _fill_identity_for_inf_stats docstring.
    n = _fill_identity_for_inf_stats(policy)
    if n:
        logger.warning("[stats] %d buffers were inf and have been set to identity", n)

    policy.to(_DEVICE)
    policy.eval()
    policy.reset()
    _POLICY = policy

    cfg = policy.config
    _IMAGE_KEYS = sorted(cfg.image_features.keys())
    state_feat = cfg.input_features.get("observation.state")
    _STATE_DIM = int(state_feat.shape[0]) if state_feat is not None else 0
    action_feat = cfg.output_features.get("action")
    _ACTION_DIM = int(action_feat.shape[0]) if action_feat is not None else 0

    logger.info("Loaded in %.1fs. image_keys=%s state_dim=%d action_dim=%d",
                time.time() - t0, _IMAGE_KEYS, _STATE_DIM, _ACTION_DIM)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.expanduser("~/ckpt/pi0_base"),
                   help="Path to pi0 pretrained checkpoint dir")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    _load_policy(args.ckpt, args.device)
    logger.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
