# HeLM: Hierarchical Explicit Language Memory for VLM-Guided Non-Markovian Manipulation

**[Project Page](https://kimspacehuni.github.io/HeLM/)**

Most robot policies operate under a **Markovian assumption** — mapping only the current observation to an action — which causes them to fail at tasks requiring working memory (e.g., counting button presses, maintaining order of actions across episodes). **HeLM** addresses this by incorporating memory as a natural language into the robot's control loop, inspired by how humans use an *inner monologue* to track progress.

HeLM decouples memory management from action generation through a two-level hierarchy:
- **HLP (High-Level Policy)**: a fine-tuned VLM that maintains an explicit language memory buffer and monitors for task-relevant events.
- **LLP (Low-Level Policy)**: a VLA-based action policy (e.g., π0) that executes motor actions guided by HLP commands.

**Key results on 10 real-world non-Markovian tasks:**
- **+13%** average success rate over baselines
- **78%** zero-shot success rate on unseen task variants
- **4.5x** more computationally efficient than image-based alternatives
- **2.5x** lower inference latency than keyframe-based methods

---

## Architecture

```
              Global Task I
                    │
         ┌──────────▼──────────┐
         │   High-Level Policy  │
         │  (VLM + Language Mem)│
         │                      │
         │  Detect Mode         │  ── monitors observations for events e_t
         │  Update Mode         │  ── updates memory M_t upon event detection
         └──────────┬──────────┘
                    │ Action Command z_t
         ┌──────────▼──────────┐
         │   Low-Level Policy   │
         │      (VLA / π0)      │
         └──────────┬──────────┘
                    │ Action a_t
                    ▼
                 Robot
```

### Structured Language Memory

The HLP maintains a structured memory `M_t = ⟨m_t^work, m_t^epi⟩`:

| Component | Scope | Role | Example |
|---|---|---|---|
| **Working Memory** `m_t^work` | Intra-episode | Tracks active progress within current task | `Count: 2 (Goal: 3)` |
| **Episodic Context** `m_t^epi` | Inter-episode | Preserves critical info across task boundaries | `Previous Pressed: 5, Object Origin: red plate` |

### Two-Stage HLP Inference

1. **Detect Mode** (`π_det`): At every step, the VLM classifies whether the current observation is a task-relevant event (e.g., `"pressed red button"`, `"wiped the top window"`) or `"none"`.
2. **Update Mode** (`π_upd`): Triggered only when an event is detected or the global instruction changes. Updates `M_t` and generates a new action command `z_t` for the LLP.

This event-triggered design prevents redundant memory updates from visually similar frames and keeps the memory compact and interpretable.

---

## Task Suite

HeLM is evaluated on 10 real-world non-Markovian tasks across two categories:

### Intra-Episode Tasks (Working Memory)
The agent must track sequential logic and progress within a single instruction.

| ID | Task | Memory Requirement |
|---|---|---|
| A1 | Press N Times | Count repeated actions precisely |
| A2 | Press in Order | Track completed steps in a sequence |
| A3 | Wipe the Window | Remember which regions have been wiped |
| A4 | Find the Object | Recall object locations from past observations |
| A5 | Pick Place and Press | Multi-step sequential manipulation |

### Inter-Episode Tasks (Episodic Context)
The agent must retain information across task boundaries and episode resets.

| ID | Task | Memory Requirement |
|---|---|---|
| E1 | Press N Times → Press M Times Total | Recall count from prior episode |
| E2 | Press the Button in Human Order | Memorize human-demonstrated button sequence |
| E3 | Wipe the Remaining Window | Track which regions were wiped in a previous episode |
| E4 | Open the Drawer with Object | Remember which drawer contains a specific object |
| E5 | Pick Place to Original Position | Restore objects to their originally observed locations |

---

## Results

### Intra-Episode (Success Rate %)

| Method | A1 | A2 | A3 | A4 | A5 | **Avg** |
|---|---|---|---|---|---|---|
| **HeLM (Ours)** | **100** | **92.2** | **96.6** | **66.6** | **73.3** | **85.7** |
| Keyframe-HLP | 90* | 91.1 | 60 | 79.2* | 44.4 | 72.9 |
| PCMB-π0 | 100 | 24.4 | 33.3 | 32.9 | 6.7 | 39.5** |
| Naïve-π0 | 31.1 | 6.6 | 33.3 | 8.3 | 4.4 | 16.7** |

\* Keyframe-HLP fails to detect termination in A1 and A4.
\*\* PCMB-π0 and Naïve-π0 fail both sequencing and termination across all trials.

### Inter-Episode (Success Rate %)

| Method | E1 | E2 | E3 | E4 | E5 | **Avg** |
|---|---|---|---|---|---|---|
| **HeLM (Ours)** | **90.5** | **87.7** | **90** | **80** | **97.7** | **89.2** |
| Keyframe-HLP | 27.6* | 0 | 33.3 | 0 | 22.2 | 16.6 |

\* Keyframe-HLP fails to detect termination.

PCMB-π0 and Naïve-π0 are excluded from inter-episode comparison as they reset internal states at episode boundaries.

### Zero-Shot Generalization (Success Rate %)

| Method | A1' | A2' | A3' | A4' | A5' | **Avg** |
|---|---|---|---|---|---|---|
| **HeLM (Ours)** | **100** | **95** | **67** | **67** | **61** | **78** |
| Keyframe-HLP | 100* | 63.9 | 77.8 | 38.9* | 48.1* | 65.7 |

\* Keyframe-HLP fails to detect termination in A1', A4', and A5'.

### Computational Efficiency

| Metric | HeLM | Keyframe-HLP |
|---|---|---|
| VRAM (8 events) | ~6 GB | ~28 GB |
| Latency scaling | Stable | Grows sharply with task length |

---

## Directory Structure

```
HeLM/
├── common/                  # Shared infrastructure (based on LeRobot).
│                            # Includes model implementations (π0, SmolVLA) and dataset loaders.
│
├── helm_datasets/           # HeLM dataset pipeline (main)
│   ├── core/                # Spec, templates, data index, I/O
│   ├── taskspecs/           # Task specification JSONs
│   ├── libero_taskspecs/    # LIBERO task specs
│   ├── extract_frames.py    # Step 1: extract frames from LeRobot videos
│   ├── annotate_app.py      # Step 2: Streamlit annotation tool
│   ├── build_helm.py        # Step 3: generate DETECT/UPDATE JSONL
│   ├── merge_helm_data.py   # Step 4: merge JSONL across tasks
│   ├── validate_helm_data.py # Streamlit viewer for spot-checking
│
├── train/
│   ├── train_helm/          # HeLM HLP training (Qwen2.5-VL)
│   ├── train_helm_smolvlm/  # HeLM HLP training (SmolVLM)
│   └── train_llp/           # LLP training (π0)
│
├── evaluate/
│   ├── eval_helm/           # HLP validation (DETECT/UPDATE accuracy on held-out data)
│   └── eval_HLP_LLP/        # Real-time full-system evaluation: main script connects
│                            # HLP (event detection + memory update) and LLP (action
│                            # execution) in a closed loop on the physical robot.
│
├── configs/                 # Dataclass-based configs (dataset, train, eval)
└── scripts/                 # Automation scripts for pipeline steps
```

---

## Installation

```bash
git clone https://github.com/kimspacehuni/HeLM.git && cd HeLM

conda env create -f environment.yml
conda activate HeLM
```

> `flash-attn` is included in `environment.yml` but requires a matching CUDA version to build. If installation fails, install it separately following the [flash-attention installation guide](https://github.com/Dao-AILab/flash-attention).

> For real-robot evaluation: AgileX PiPER arm with Intel RealSense D455 (workspace) and L515 (wrist) cameras. See `common/robot_devices/` for device setup.

---

## Dataset Pipeline

The `helm_datasets/` package builds the HeLM training dataset from raw robot demonstration videos in LeRobot format. Training data consists of ~130k scene frames annotated with event labels and memory states.

### Step 1 — Extract frames

```bash
python -m helm_datasets.extract_frames \
  --lerobot_root /data/your_dataset/lerobot_5hz \
  --out_root     /data/helm_data/your_task
```

### Step 2 — Annotate event frames

```bash
export PYTHONPATH=$(pwd)
streamlit run helm_datasets/annotate_app.py -- \
  --out_root /data/helm_data/your_task
```

### Step 3 — Build DETECT/UPDATE JSONL

```bash
python -m helm_datasets.build_helm \
  --out_root /data/helm_data/your_task \
  --task_id  press_button_1 \
  --split    train,val \
  --seed     1234
```

Generates DETECT rows (every frame) and UPDATE rows (event frames only) as ShareGPT-format JSONL.

### Step 4 — Merge across tasks

```bash
python -m helm_datasets.merge_helm_data \
  --src_root /data/helm_data \
  --out_root /data/helm_data/merged \
  --tasks    press_button_1,press_button_2,wipe_top
```

### Validation

```bash
streamlit run helm_datasets/validate_helm_data.py -- --jsonl /data/helm_data/merged/all_val.jsonl
```

---

## Training

### HLP — Qwen2.5-VL-7B (QLoRA, rank 16)

```bash
export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master-port=29545 train/train_helm/train_helm.py \
  --model_name_or_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /data/helm_data/merged/all_train.jsonl \
  --val_jsonl   /data/helm_data/merged/all_val.jsonl \
  --output_dir  /results/HeLM_HLP \
  --num_images 1 \
  --batch_size 8 \
  --n_detect_pos 2 --n_detect_neg 2 \
  --n_update_intra 2 --n_update_transition 2 \
  --num_train_epochs 3 \
  --wandb_project HeLM
```

### HLP — SmolVLM (lightweight alternative)

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_helm_smolvlm/train_helm_smolvlm.py \
  --train_jsonl /data/helm_data/merged/all_train.jsonl \
  --val_jsonl   /data/helm_data/merged/all_val.jsonl \
  --output_dir  /results/HeLM_HLP_SmolVLM
```

### LLP — π0 (per-task fine-tuning, 150–200 demos)

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_llp/train_pi0.py \
  --policy.path=/path/to/pi0_base \
  --train_dataset.repo_id=<dataset_id> \
  --train_dataset.root=/data/lerobot/<dataset_id> \
  --test_dataset.repo_id=<dataset_id> \
  --test_dataset.root=/data/lerobot/<dataset_id> \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=5000 \
  --output_dir=/results/LLP_checkpoint
```

---

## Evaluation

### HLP Validation (DETECT/UPDATE accuracy)

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate/eval_helm/eval_helm_hlp.py \
  --jsonl       /data/helm_data/merged/all_val.jsonl \
  --base_model  /ckpt/Qwen2.5-VL-7B-Instruct \
  --adapter     /results/HeLM_HLP/checkpoint-3000 \
  --max_samples 200 \
  --out_jsonl   /results/eval_preds.jsonl
```

### Full system — real-time HLP + LLP

```bash
python evaluate/eval_HLP_LLP/eval_real_time_main.py \
  --llp_model_path /results/LLP_checkpoint \
  --hlp_base      /path/to/Qwen2.5-VL-7B-Instruct \
  --hlp_adapter   /results/HeLM_HLP/checkpoint-3000 \
  --taskspecs_dir helm_datasets/taskspecs \
  --task_group    press_button_N_times_M_times_total
```

---

## License

This project is released under the [Apache 2.0 License](LICENSE).

Portions of `common/policies/pi0/` and `common/policies/smolvla/` are derived from [LeRobot](https://github.com/huggingface/lerobot) (Apache 2.0).
