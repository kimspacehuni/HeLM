# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from common.optim.optimizers import AdamWConfig
from common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from configs.policies import PreTrainedConfig
from configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("pi0")
@dataclass(eq=False)
class PI0Config(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Add empty images. Used by pi0_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # If true, transform the action chunk into chunk-relative cumulative form
    # at training time (target = cumsum(deltas)) and invert back to per-step
    # deltas at inference (delta_t = cum_t - cum_{t-1}). Helps when the raw
    # data is bang-bang discrete delta (e.g., LIBERO-Mem teleop) — pi0 then
    # learns continuous trajectory shapes instead of collapsing to majority
    # delta on OOD inputs.
    action_chunk_relative: bool = False

    # If true, the dataset's action[:, 0:3] has been pre-transformed offline to
    # absolute EEF xyz target (= state[t+1, 0:3] from observation.state). pi0
    # learns continuous trajectories in robot-base coords instead of bang-bang
    # delta. At inference, select_action converts predicted abs xyz back to per-
    # step delta using the current observation EEF as the chunk anchor:
    #   delta[0] = abs[0] - current_eef
    #   delta[i] = abs[i] - abs[i-1]   for i >= 1
    # action[:, 3:7] (rotation delta + gripper) is unchanged in the dataset and
    # passed through at inference.
    action_abs_xyz: bool = False

    # Gain applied to the per-step abs-xyz diff at inference to match LIBERO's
    # OSC_POSE controller scale. OSC interprets `action ∈ [-1, +1]` where 1 maps
    # to a 0.05 m position offset target, and the PD loop achieves ~25% of that
    # per 20 Hz step → measured per-step state diff in demos is ~0.005 m while
    # the commanded action saturated at ±0.4 (= ±0.02 m offset). Empirical
    # ratio commanded / achieved ≈ 80, so multiply by 80 then clip to ±1 to
    # reproduce demo speed when sending diffs back through OSC. Override at the
    # CLI for tuning.
    action_abs_xyz_gain: float = 80.0

    # Tokenizer
    tokenizer_max_length: int = 48

    # Projector
    proj_width: int = 1024

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # or fa2, flex

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 2.5e-6
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1.5e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6
    optimizer_grad_clip_norm: float = 1.0

    # ----- Fields stored by the upstream lerobot/openpi pi0 checkpoint format. -----
    # We keep them on the dataclass so draccus accepts the checkpoint's config.json
    # without raising on unknown keys. They are not used by HeLM's training/eval
    # paths (which read num_steps / resize_imgs_with_padding instead), but loading
    # a pretrained ckpt requires the schema to match.

    # HuggingFace Hub publishing metadata
    push_to_hub: bool = True
    repo_id: str | None = None
    private: bool | None = None
    tags: list[str] | None = None
    license: str | None = None

    # Architecture variants
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"

    # Flow-matching inference parameters (mirror of num_steps)
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    min_period: float = 0.004
    max_period: float = 4.0

    # Image input (mirror of resize_imgs_with_padding)
    image_resolution: tuple[int, int] = (224, 224)

    # Runtime
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    # TODO: Add EMA

    def __post_init__(self):
        super().__post_init__()

        self._attn_implementation_internal = self.attention_implementation

        # TODO(Steven): Validate device and amp? in all policy configs?
        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by pi0 for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        # TODO: implement value error
        # if not self.image_features and not self.env_state_feature:
        #     raise ValueError("You must provide at least one image or the environment state among the inputs.")

        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
