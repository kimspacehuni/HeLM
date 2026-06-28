"""
FastAPI server that hosts BOTH pi0 (LLP) and Qwen2.5-VL (HLP) for the
HeLM × LIBERO smoke test. The libero-side client (running in helm_libero env)
talks to a single endpoint and gets DETECT, UPDATE, and pi0 PREDICT all on
this process.

Run inside the HeLM env on a GPU worker node:

    PYTHONPATH=$(pwd) \\
    python tools/helm_full_server.py \\
        --pi0-ckpt ~/ckpt/pi0_base \\
        --qwen-base ~/ckpt/Qwen2.5-VL-7B-Instruct \\
        --device cuda:0 \\
        --port 8000

For the smoke test, --qwen-adapter is left unset so we serve the base
Qwen2.5-VL with no LoRA / QLoRA finetuning. Pass it once a real HLP adapter
exists.

Endpoints:
    GET  /health           — both models loaded, capabilities summary
    POST /reset            — clear pi0 action queue (call at episode start)
    POST /predict          — pi0 next action (existing behavior)
    POST /detect           — Qwen DETECT (Event_Detected, Event)
    POST /update           — Qwen UPDATE (next memory + Action_Command)
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
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.policies.pi0.modeling_pi0 import PI0Policy  # noqa: E402
from evaluate.eval_HLP_LLP.eval_real_time_qwen import HLPQwenV2  # noqa: E402
from evaluate.eval_HLP_LLP.utils_batches import (  # noqa: E402
    create_hlp_detect_batch,
    create_hlp_update_batch,
)
from helm_datasets.core.templates import make_detect_prompt, make_update_prompt  # noqa: E402

logger = logging.getLogger("helm_full_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------- Globals ----------

_POLICY: Optional[PI0Policy] = None
_HLP: Optional[HLPQwenV2] = None
_DEVICE: Optional[torch.device] = None
_IMAGE_KEYS: List[str] = []
_STATE_DIM: int = 0
_ACTION_DIM: int = 0
_OBS_PREFIX = "observation.images."

# Optional forward mode: if _PI_FORWARD_URL is set, /predict, /reset, and the
# pi-related half of /health are proxied to that URL (e.g. a pi05 server
# running in a separate conda env). _POLICY stays None in that case.
_PI_FORWARD_URL: Optional[str] = None

# Provenance — set at startup, reported via /health for benchmark logging.
_PI_CKPT_PATH: Optional[str] = None
_QWEN_BASE_PATH: Optional[str] = None
_QWEN_ADAPTER_PATH: Optional[str] = None


# ---------- Request/response models ----------

class PredictRequest(BaseModel):
    task: str
    state: List[float]
    images: Dict[str, str]


class PredictResponse(BaseModel):
    action: List[float]
    queue_size_after: int
    latency_ms: float


class DetectRequest(BaseModel):
    task: str
    memory: Dict[str, str]
    event_list: str
    image: str


class DetectResponse(BaseModel):
    detected: bool
    event: str
    raw: str
    latency_ms: float


class UpdateRequest(BaseModel):
    task: str
    # Values may be None for the @init call (matches Qwen training distribution
    # where Previous_Memory was None | None | None at episode start).
    prev_memory: Dict[str, Optional[str]]
    allowed_actions: str
    event: str
    image: str


class UpdateResponse(BaseModel):
    memory: Dict[str, str]
    raw: str
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
    hlp_loaded: bool
    # Provenance info for offline analysis of saved benchmark runs.
    pi_ckpt: Optional[str] = None        # local pi0 ckpt path (None if forwarded)
    pi_forward_url: Optional[str] = None # external pi05 server URL (None if local pi0)
    qwen_base: Optional[str] = None
    qwen_adapter: Optional[str] = None


# ---------- Helpers ----------

def _decode_image_tensor(b64: str) -> torch.Tensor:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def _decode_image_pil(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _build_pi0_batch(req: PredictRequest) -> Dict[str, torch.Tensor]:
    assert _DEVICE is not None
    batch: Dict[str, torch.Tensor] = {}
    for k, b64 in req.images.items():
        full_key = k if k.startswith(_OBS_PREFIX) else f"{_OBS_PREFIX}{k}"
        if full_key not in _IMAGE_KEYS:
            logger.warning("Ignoring unrecognized image key %s (expected %s)", full_key, _IMAGE_KEYS)
            continue
        batch[full_key] = _decode_image_tensor(b64).to(_DEVICE, non_blocking=True)

    state = torch.tensor(req.state, dtype=torch.float32)
    if _STATE_DIM > 0 and state.shape[0] < _STATE_DIM:
        state = torch.nn.functional.pad(state, (0, _STATE_DIM - state.shape[0]), value=0.0)
    elif _STATE_DIM > 0 and state.shape[0] > _STATE_DIM:
        state = state[:_STATE_DIM]
    batch["observation.state"] = state.unsqueeze(0).to(_DEVICE, non_blocking=True)
    batch["task"] = [req.task]
    return batch


def _device_move(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


# ---------- App ----------

app = FastAPI(title="HeLM full (pi0 + Qwen2.5-VL) inference server")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _PI_FORWARD_URL:
        try:
            r = requests.get(f"{_PI_FORWARD_URL}/health", timeout=5)
            r.raise_for_status()
            pi = r.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"pi-forward server unreachable: {e}")
        return HealthResponse(
            status="ok",
            device=str(pi.get("device", "?")) + " (forwarded)",
            image_keys=list(pi.get("image_keys", [])),
            state_dim=int(pi.get("state_dim", 0)),
            action_dim=int(pi.get("action_dim", 0)),
            queue_size=int(pi.get("queue_size", 0)),
            hlp_loaded=_HLP is not None,
            pi_forward_url=_PI_FORWARD_URL,
            pi_ckpt=pi.get("pi_ckpt") or pi.get("ckpt"),
            qwen_base=_QWEN_BASE_PATH,
            qwen_adapter=_QWEN_ADAPTER_PATH,
        )
    if _POLICY is None or _DEVICE is None:
        raise HTTPException(status_code=503, detail="pi0 not loaded")
    return HealthResponse(
        status="ok",
        device=str(_DEVICE),
        image_keys=_IMAGE_KEYS,
        state_dim=_STATE_DIM,
        action_dim=_ACTION_DIM,
        queue_size=len(_POLICY._action_queue),
        hlp_loaded=_HLP is not None,
        pi_ckpt=_PI_CKPT_PATH,
        qwen_base=_QWEN_BASE_PATH,
        qwen_adapter=_QWEN_ADAPTER_PATH,
    )


@app.post("/reset", response_model=ResetResponse)
def reset() -> ResetResponse:
    if _PI_FORWARD_URL:
        r = requests.post(f"{_PI_FORWARD_URL}/reset", timeout=10)
        r.raise_for_status()
        return ResetResponse(**r.json())
    if _POLICY is None:
        raise HTTPException(status_code=503, detail="pi0 not loaded")
    _POLICY.reset()
    return ResetResponse(ok=True)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _PI_FORWARD_URL:
        try:
            r = requests.post(f"{_PI_FORWARD_URL}/predict", json=req.dict(), timeout=300.0)
            r.raise_for_status()
            return PredictResponse(**r.json())
        except requests.HTTPError as e:
            detail = e.response.text if e.response is not None else str(e)
            raise HTTPException(status_code=502, detail=f"pi-forward error: {detail}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"pi-forward error: {type(e).__name__}: {e}")
    if _POLICY is None or _DEVICE is None:
        raise HTTPException(status_code=503, detail="pi0 not loaded")
    t0 = time.perf_counter()
    try:
        batch = _build_pi0_batch(req)
        with torch.no_grad():
            action = _POLICY.select_action(batch).squeeze().detach().cpu().tolist()
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    action = action[:_ACTION_DIM] if _ACTION_DIM else action
    return PredictResponse(
        action=action,
        queue_size_after=len(_POLICY._action_queue),
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    if _HLP is None:
        raise HTTPException(status_code=503, detail="HLP (Qwen) not loaded")
    t0 = time.perf_counter()
    try:
        obs_pil = _decode_image_pil(req.image)
        prompt = make_detect_prompt(
            task_text=req.task,
            memory=req.memory,
            n_images=1,
            event_list=req.event_list,
        )
        if os.environ.get("HLP_DEBUG"):
            logger.info("[/detect] image size=%s mode=%s", obs_pil.size, obs_pil.mode)
            logger.info("[/detect] prompt (first 600 chars):\n%s", prompt[:600])
        batch = create_hlp_detect_batch(_HLP.processor, obs_pil, prompt)
        batch = _device_move(batch, _HLP.model.device)
        raw = _HLP._generate_text(batch, _HLP.max_new_tokens_detect)
        if os.environ.get("HLP_DEBUG"):
            logger.info("[/detect] raw output: %s", raw)
        from evaluate.eval_HLP_LLP.eval_real_time_qwen import parse_detect_yaml
        detected, event = parse_detect_yaml(raw)
    except Exception as e:
        logger.exception("detect failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return DetectResponse(
        detected=bool(detected),
        event=str(event),
        raw=raw,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


@app.post("/update", response_model=UpdateResponse)
def update(req: UpdateRequest) -> UpdateResponse:
    if _HLP is None:
        raise HTTPException(status_code=503, detail="HLP (Qwen) not loaded")
    t0 = time.perf_counter()
    try:
        obs_pil = _decode_image_pil(req.image)
        prompt = make_update_prompt(
            task_text=req.task,
            prev_memory=req.prev_memory,
            n_images=1,
            llp_commands=req.allowed_actions,
            event=req.event,
        )
        if os.environ.get("HLP_DEBUG"):
            logger.info("[/update] image size=%s mode=%s", obs_pil.size, obs_pil.mode)
            logger.info("[/update] prompt (first 600 chars):\n%s", prompt[:600])
        batch = create_hlp_update_batch(_HLP.processor, obs_pil, prompt)
        batch = _device_move(batch, _HLP.model.device)
        raw = _HLP._generate_text(batch, _HLP.max_new_tokens_update)
        if os.environ.get("HLP_DEBUG"):
            logger.info("[/update] raw output: %s", raw)
        from evaluate.eval_HLP_LLP.eval_real_time_qwen import parse_update_yaml
        mem = parse_update_yaml(raw)
    except Exception as e:
        logger.exception("update failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return UpdateResponse(
        memory=mem,
        raw=raw,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


# ---------- Boot ----------

def _fill_identity_for_inf_stats(model: torch.nn.Module) -> int:
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
        logger.warning("[stats] filled identity for inf entry: %s", name)
    return n_fixed


def _load_pi0(ckpt: str, device: str) -> None:
    global _POLICY, _DEVICE, _IMAGE_KEYS, _STATE_DIM, _ACTION_DIM
    ckpt = os.path.expanduser(ckpt)
    logger.info("Loading PI0Policy from %s on %s ...", ckpt, device)
    t0 = time.time()
    _DEVICE = torch.device(device)
    policy = PI0Policy.from_pretrained(ckpt)
    n = _fill_identity_for_inf_stats(policy)
    if n:
        logger.warning("[stats] %d entries set to identity", n)
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
    logger.info("pi0 loaded in %.1fs. image_keys=%s state_dim=%d action_dim=%d",
                time.time() - t0, _IMAGE_KEYS, _STATE_DIM, _ACTION_DIM)


def _load_hlp(qwen_base: str, qwen_adapter: Optional[str], device: str, load_in_4bit: bool) -> None:
    global _HLP
    qwen_base = os.path.expanduser(qwen_base)
    qwen_adapter = os.path.expanduser(qwen_adapter) if qwen_adapter else None
    logger.info("Loading HLPQwenV2 base=%s adapter=%s on %s (4bit=%s) ...",
                qwen_base, qwen_adapter, device, load_in_4bit)
    _HLP = HLPQwenV2(
        base_model_path=qwen_base,
        adapter_path=qwen_adapter,
        device=device,
        attn_impl="sdpa",
        load_in_4bit=load_in_4bit,
    )


def main() -> None:
    global _PI_FORWARD_URL, _PI_CKPT_PATH, _QWEN_BASE_PATH, _QWEN_ADAPTER_PATH
    p = argparse.ArgumentParser()
    p.add_argument("--pi0-ckpt", default=os.path.expanduser("~/ckpt/pi0_base"))
    p.add_argument("--pi-forward-url", default=None,
                   help="If set, /predict, /reset, and pi-side of /health are "
                        "HTTP-forwarded to this URL (e.g. http://localhost:8001 "
                        "for a pi05 server in a separate conda env). pi0 is then "
                        "NOT loaded locally; only Qwen runs in this process.")
    p.add_argument("--qwen-base", default=os.path.expanduser("~/ckpt/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--qwen-adapter", default=None,
                   help="Optional LoRA/QLoRA adapter path. Omit for smoke test.")
    p.add_argument("--device", default="cuda:0",
                   help="Single device for both pi0 and Qwen. With 4bit Qwen + bf16 pi0 fits ~25GB.")
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable Qwen 4bit quantization (uses bf16; needs more VRAM).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    _QWEN_BASE_PATH = os.path.expanduser(args.qwen_base) if args.qwen_base else None
    _QWEN_ADAPTER_PATH = os.path.expanduser(args.qwen_adapter) if args.qwen_adapter else None
    if not args.pi_forward_url:
        _PI_CKPT_PATH = os.path.expanduser(args.pi0_ckpt) if args.pi0_ckpt else None

    if args.pi_forward_url:
        _PI_FORWARD_URL = args.pi_forward_url.rstrip("/")
        logger.info("[pi-forward] /predict, /reset will be proxied to %s", _PI_FORWARD_URL)
        # Sanity check: target must be reachable before we accept client traffic
        try:
            r = requests.get(f"{_PI_FORWARD_URL}/health", timeout=5)
            r.raise_for_status()
            logger.info("[pi-forward] target /health: %s", r.json())
        except Exception as e:
            logger.warning("[pi-forward] target unreachable at startup (%s); continuing anyway", e)
    else:
        _load_pi0(args.pi0_ckpt, args.device)

    _load_hlp(args.qwen_base, args.qwen_adapter, args.device, load_in_4bit=not args.no_4bit)

    logger.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
