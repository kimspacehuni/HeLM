"""
FastAPI server wrapping lerobot's PI05Policy for remote inference.

Runs in a separate conda env (HeLM_pi05) because lerobot pi05 requires
transformers >= 5.x, which is incompatible with the main HeLM env.

Usage:
    /rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM_pi05/bin/python \
        tools/helm_pi05_server.py \
        --ckpt ~/ckpt/pi05_libero \
        --device cuda:0 --host 0.0.0.0 --port 8000

API matches helm_pi0_server.py so the existing libero clients (mem_8/mem_9)
work without changes: POST /predict, /reset, GET /health.
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from lerobot.policies.pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

logger = logging.getLogger("helm_pi05_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------- Request/response models (match helm_pi0_server.py) ----------

class PredictRequest(BaseModel):
    task: str
    state: List[float]
    images: Dict[str, str]


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
    pi_ckpt: Optional[str] = None
    abs_xyz_inverse: Optional[bool] = None
    abs_xyz_gain: Optional[float] = None


# ---------- Globals ----------

_POLICY: Optional[PI05Policy] = None
_PREPROCESSOR: Optional[PolicyProcessorPipeline] = None
_POSTPROCESSOR: Optional[PolicyProcessorPipeline] = None
_DEVICE: Optional[torch.device] = None
_IMAGE_KEYS: List[str] = []
_STATE_DIM: int = 0
_ACTION_DIM: int = 0
_OBS_PREFIX = "observation.images."
_CKPT_PATH: Optional[str] = None  # set in _load_policy, reported via /health

# abs_xyz inverse: if the model was fine-tuned on absolute EEF target xyz
# (as in HeLM's bowl_bottle_sub_abs_xyz dataset), the raw action coming out
# of pi05 is `[x_target, y_target, z_target, 0, 0, 0, gripper]` in world
# coordinates. The LIBERO env expects delta, so we convert here:
#     delta_xyz = (target_xyz - current_state_xyz) * gain
#     clamp to [-1, 1]
# This mirrors HeLM's pi0 `select_action` inverse with `action_abs_xyz=True`.
_ABS_XYZ_INVERSE: bool = False
_ABS_XYZ_GAIN: float = 80.0

# pi05_libero was trained with LIBERO standard camera key names (image, image2)
# but the libero-mem clients send the LeRobot v2 convention (base_0_rgb,
# left_wrist_0_rgb). Map at the server boundary so client doesn't change.
# empty_camera_0 is auto-filled with -1 padding by pi05._preprocess_images.
_CLIENT_TO_MODEL_KEY = {
    "base_0_rgb": "image",
    "left_wrist_0_rgb": "image2",
}


# ---------- Helpers ----------

def _decode_image(b64: str) -> torch.Tensor:
    """base64 PNG/JPEG → float tensor [1, C, H, W] in [0, 1]."""
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def _build_observation(req: PredictRequest) -> Dict[str, torch.Tensor]:
    """Build flat observation dict in lerobot format (no batch dim — the
    AddBatchDimensionProcessorStep in the preprocessor adds it)."""
    assert _DEVICE is not None
    obs: Dict[str, object] = {}

    # Images: server expects (C, H, W) without batch dim; preprocessor adds it.
    # Remap client-side key names to whatever names the loaded model expects.
    for k, b64 in req.images.items():
        bare = k[len(_OBS_PREFIX):] if k.startswith(_OBS_PREFIX) else k
        mapped = _CLIENT_TO_MODEL_KEY.get(bare, bare)
        full_key = f"{_OBS_PREFIX}{mapped}"
        if _IMAGE_KEYS and full_key not in _IMAGE_KEYS:
            logger.warning(
                "Ignoring image key %s (mapped to %s; model expects one of %s)",
                k, full_key, _IMAGE_KEYS,
            )
            continue
        obs[full_key] = _decode_image(b64).squeeze(0)  # (C, H, W)

    # State (raw, will be padded to max_state_dim by config and normalized by
    # NormalizerProcessorStep upstream of state-to-language discretization).
    state = torch.tensor(req.state, dtype=torch.float32)
    obs["observation.state"] = state

    # Task as a single string (preprocessor wraps into list when adding batch dim).
    obs["task"] = req.task
    return obs


# ---------- App ----------

app = FastAPI(title="HeLM pi05 inference server")


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
        pi_ckpt=_CKPT_PATH,
        abs_xyz_inverse=_ABS_XYZ_INVERSE,
        abs_xyz_gain=_ABS_XYZ_GAIN,
    )


@app.post("/reset", response_model=ResetResponse)
def reset() -> ResetResponse:
    if _POLICY is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    _POLICY.reset()
    return ResetResponse(ok=True)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _POLICY is None or _PREPROCESSOR is None or _POSTPROCESSOR is None or _DEVICE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        obs = _build_observation(req)
        obs = _PREPROCESSOR(obs)
        with torch.inference_mode():
            action = _POLICY.select_action(obs)
        action = _POSTPROCESSOR(action)
        action_vec = action.squeeze().detach().cpu().tolist()
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    if _ACTION_DIM:
        action_vec = action_vec[:_ACTION_DIM]

    # abs target xyz -> delta xyz conversion (see _ABS_XYZ_INVERSE docstring)
    if _ABS_XYZ_INVERSE and len(action_vec) >= 3 and len(req.state) >= 3:
        for i in range(3):
            d = (action_vec[i] - req.state[i]) * _ABS_XYZ_GAIN
            action_vec[i] = max(-1.0, min(1.0, d))
    return PredictResponse(
        action=action_vec,
        queue_size_after=len(_POLICY._action_queue),
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


# ---------- Boot ----------

def _load_policy(ckpt: str, device: str) -> None:
    global _POLICY, _PREPROCESSOR, _POSTPROCESSOR, _DEVICE
    global _IMAGE_KEYS, _STATE_DIM, _ACTION_DIM, _CKPT_PATH

    ckpt = os.path.expanduser(ckpt)
    _CKPT_PATH = ckpt
    logger.info("Loading PI05Policy from %s on %s ...", ckpt, device)
    t0 = time.time()

    _DEVICE = torch.device(device)
    policy = PI05Policy.from_pretrained(ckpt)
    policy.to(_DEVICE)
    policy.eval()
    policy.reset()
    _POLICY = policy

    _PREPROCESSOR = PolicyProcessorPipeline.from_pretrained(
        ckpt, config_filename="policy_preprocessor.json"
    )
    # The postprocessor takes a raw action tensor (PolicyAction) and produces a
    # raw action tensor — its internal steps work on transitions, so we pass
    # explicit tensor<->transition converters. Without these from_pretrained
    # falls back to batch_to_transition which expects a dict.
    _POSTPROCESSOR = PolicyProcessorPipeline.from_pretrained(
        ckpt,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    cfg = policy.config
    img_feats = getattr(cfg, "image_features", None) or {
        k: v for k, v in cfg.input_features.items() if k.startswith(_OBS_PREFIX)
    }
    _IMAGE_KEYS = sorted(img_feats.keys())

    state_feat = cfg.input_features.get("observation.state")
    _STATE_DIM = int(state_feat.shape[0]) if state_feat is not None else 0
    action_feat = cfg.output_features.get("action")
    _ACTION_DIM = int(action_feat.shape[0]) if action_feat is not None else 0

    logger.info(
        "Loaded in %.1fs. image_keys=%s state_dim=%d action_dim=%d",
        time.time() - t0, _IMAGE_KEYS, _STATE_DIM, _ACTION_DIM,
    )


def main() -> None:
    global _ABS_XYZ_INVERSE, _ABS_XYZ_GAIN
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.expanduser("~/ckpt/pi05_libero"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--abs-xyz-inverse", action="store_true",
                   help="Treat model output [0:3] as absolute EEF target xyz "
                        "and convert to delta via (target - state) * gain, clip ±1.")
    p.add_argument("--abs-xyz-gain", type=float, default=80.0)
    args = p.parse_args()

    _ABS_XYZ_INVERSE = bool(args.abs_xyz_inverse)
    _ABS_XYZ_GAIN = float(args.abs_xyz_gain)
    if _ABS_XYZ_INVERSE:
        logger.info("[abs_xyz_inverse] ENABLED (gain=%.1f)", _ABS_XYZ_GAIN)

    _load_policy(args.ckpt, args.device)
    logger.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
