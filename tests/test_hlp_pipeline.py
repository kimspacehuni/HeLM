"""HLP data contracts and optional integration checks with local processors/data."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image
import torch
import yaml

from common.datasets.hlp_utils import load_hlp_images, mask_assistant_tokens
from helm_datasets.core.registry import load_all_taskspecs, load_taskspec
from helm_datasets.core.data_index import DataEpisodeV4, load_data_episode
from helm_datasets.build_helm import build_for_task, build_update_rows_for_episode_step
from helm_datasets import merge_helm_data
from train.train_helm.helm_dataset import DataCollatorForQwenVL, HelmJsonlDatasetV4, V4DatasetConfig
from train.train_helm_smolvlm.helm_dataset import DataCollatorForSmolVLM, HelmJsonlDatasetSmolVLM, SmolVLMDatasetConfig

ROOT = Path(__file__).resolve().parents[1]


class HlpContractTests(unittest.TestCase):
    def test_all_repository_specs_validate_within_their_task_groups(self):
        total = 0
        for directory in sorted((ROOT / "helm_datasets/taskspecs").iterdir()):
            if directory.is_dir():
                total += len(load_all_taskspecs(directory))
        total += len(load_all_taskspecs(ROOT / "helm_datasets/libero_taskspecs"))
        for example in (ROOT / "helm_datasets/taskspecs").glob("*.json"):
            load_taskspec(example)
            total += 1
        expected = sum(1 for base in [ROOT / "helm_datasets/taskspecs", ROOT / "helm_datasets/libero_taskspecs"]
                       for _ in base.rglob("*.json"))
        self.assertEqual(total, expected)

    def test_invalid_memory_and_command_are_rejected(self):
        spec = load_taskspec(ROOT / "helm_datasets/taskspecs/press_button_N_times/press_blue_button_1.json")
        for change in [{"Action_Command": "unknown"}, {"Working_Memory": None}]:
            broken = copy.deepcopy(spec.memory_grid)
            broken[0][0].update(change)
            with self.assertRaises(ValueError):
                replace(spec, memory_grid=broken).validate()

    def test_missing_and_corrupt_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "table.jpg"
            Image.new("RGB", (32, 32)).save(table)
            item = {"uid": "image-check", "images": {"table": str(table)}}
            self.assertEqual(len(load_hlp_images(item, 1)), 1)
            with self.assertRaisesRegex(ValueError, "wrist"):
                load_hlp_images(item, 2)
            table.write_text("not an image")
            with self.assertRaisesRegex(ValueError, "image-check"):
                load_hlp_images(item, 1)

    def test_empty_batches_are_rejected(self):
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, padding_side="left"))
        for collator in [DataCollatorForQwenVL(processor), DataCollatorForSmolVLM(processor)]:
            for batch in [[], [None]]:
                with self.assertRaises(ValueError):
                    collator(batch)

    def test_label_mask_uses_exact_prefix_and_rejects_invalid_samples(self):
        full = torch.tensor([1, 2, 1, 2, 3])
        prompt = torch.tensor([1, 2, 1])
        self.assertEqual(mask_assistant_tokens(full, prompt).tolist(), [-100, -100, -100, 2, 3])
        with self.assertRaises(ValueError):
            mask_assistant_tokens(full, torch.tensor([9]))
        with self.assertRaises(ValueError):
            mask_assistant_tokens(full, full)

    def test_missing_event_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "frames_5hz/chunk-000/episode_000000"
            (base / "table").mkdir(parents=True)
            Image.new("RGB", (32, 32)).save(base / "table/frame_000000.jpg")
            (base / "episode_000000.json").write_text('{"event_frame_idxs":[1]}')
            with self.assertRaisesRegex(ValueError, "event frames"):
                load_data_episode(Path(tmp), 5, "chunk-000", "episode_000000", False)

    def test_init_and_inter_episode_update_rows_preserve_gold_memory(self):
        specs = [spec for group in (ROOT / "helm_datasets/taskspecs").iterdir() if group.is_dir()
                 for spec in load_all_taskspecs(group).values() if spec.inter > 0]
        for spec in specs:
            episode = DataEpisodeV4("chunk-000", "episode_000000", {}, list(range(20)), [18])
            before = copy.deepcopy(spec.memory_grid)
            initial = build_update_rows_for_episode_step(spec, episode, ROOT, ROOT, 5, 1, "train", 0, 0)
            transition = build_update_rows_for_episode_step(spec, episode, ROOT, ROOT, 5, 1, "train", 1, 0)
            init_row = next(row for row in initial if row["kind"] == "init")
            transition_row = next(row for row in transition if row["kind"] == "transition")
            self.assertEqual(yaml.safe_load(init_row["gt_text"]), spec.memory_grid[0][0])
            self.assertEqual(yaml.safe_load(transition_row["gt_text"]), spec.memory_grid[1][0])
            self.assertIn(spec.memory_grid[0][-1]["Episodic_Context"], transition_row["user_prompt"])
            self.assertEqual(spec.memory_grid, before)

    def test_runtime_parsers_and_transition_reject_invalid_outputs(self):
        from evaluate.eval_HLP_LLP.eval_real_time_qwen import parse_detect_yaml, parse_update_yaml
        from evaluate.eval_HLP_LLP import eval_real_time_main as runtime
        self.assertEqual(parse_detect_yaml('Event_Detected: "False"\nEvent: pressed'), (False, "none"))
        self.assertEqual(parse_detect_yaml('Event_Detected: true\nEvent: pressed'), (True, "pressed"))
        self.assertEqual(parse_detect_yaml('```yaml\nEvent_Detected: true\nEvent: pressed\n```'), (True, "pressed"))
        with self.assertRaises(ValueError):
            parse_update_yaml("Action_Command: done")
        with self.assertRaises(ValueError):
            parse_update_yaml("Action_Command: done\nWorking_Memory: []\nEpisodic_Context: None")
        spec = next(iter(load_all_taskspecs(ROOT / "helm_datasets/taskspecs/press_button_N_times_M_times_total").values()))
        previous, expected = spec.transition_prev_curr()
        before = copy.deepcopy(previous)
        hlp = SimpleNamespace(processor=object(), model=SimpleNamespace(device=torch.device("cpu")),
                              update=lambda batch: parse_update_yaml(yaml.safe_dump(expected)))
        with mock.patch.object(runtime, "create_hlp_update_batch", return_value={}):
            memory, _ = runtime.run_hlp_update(hlp, None, spec.task_text[1], previous, spec.llp_commands, event="task changed")
            self.assertEqual(memory, expected)
            hlp.update = lambda batch: {**expected, "Action_Command": "unknown"}
            with self.assertRaises(ValueError):
                runtime.run_hlp_update(hlp, None, spec.task_text[1], previous, spec.llp_commands)
        self.assertEqual(previous, before)

    def test_runtime_loads_all_groups_without_device_dependencies(self):
        from evaluate.eval_HLP_LLP.eval_real_time_main import _load_taskspecs_from_group
        root = ROOT / "helm_datasets/taskspecs"
        for group in root.iterdir():
            if group.is_dir():
                self.assertEqual(len(_load_taskspecs_from_group(str(root), group.name)), len(list(group.glob("*.json"))))


class LocalProcessorTests(unittest.TestCase):
    def test_real_processors_mask_only_answer_and_collate_one_and_two_images(self):
        paths = {name: os.environ.get(variable) for name, variable in
                 [("qwen", "HELM_QWEN_PROCESSOR"), ("smol", "HELM_SMOL_PROCESSOR")]}
        if not all(paths.values()):
            self.skipTest("Set HELM_QWEN_PROCESSOR and HELM_SMOL_PROCESSOR to local processor directories")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            table, wrist = tmp / "table.jpg", tmp / "wrist.jpg"
            Image.new("RGB", (64, 48), "red").save(table)
            Image.new("RGB", (64, 48), "blue").save(wrist)
            expected = "Event_Detected: false\nEvent: none"
            for name, cls, cfg, collator in [
                ("qwen", HelmJsonlDatasetV4, V4DatasetConfig, DataCollatorForQwenVL),
                ("smol", HelmJsonlDatasetSmolVLM, SmolVLMDatasetConfig, DataCollatorForSmolVLM),
            ]:
                for count in (1, 2):
                    with self.subTest(processor=name, images=count):
                        row = {"uid": "processor-check", "images": {"table": str(table), "wrist": str(wrist)},
                               "user_prompt": f"Do not repeat this example: {expected}", "gt_text": expected, "label": "detect_neg"}
                        short_row = {**row, "uid": "processor-check-short", "user_prompt": "Detect the event."}
                        path = tmp / "rows.jsonl"
                        path.write_text(json.dumps(short_row) + "\n" + json.dumps(row) + "\n")
                        dataset = cls(cfg(str(path), paths[name], num_images=count))
                        sample = dataset[0]
                        labels = sample["labels"]
                        answer = dataset.processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=True)
                        self.assertEqual(answer.strip(), expected)
                        second = dataset[1]
                        second_answer = dataset.processor.tokenizer.decode(second["labels"][second["labels"] != -100], skip_special_tokens=True)
                        self.assertEqual(second_answer.strip(), expected)
                        batch = collator(dataset.processor)([sample, second])
                        self.assertEqual(batch["labels"].shape[0], 2)
                        self.assertTrue(torch.equal(batch["labels"][0, -labels.numel():], labels))
                        padding = batch["labels"].shape[1] - labels.numel()
                        self.assertGreater(padding, 0)
                        self.assertTrue((batch["labels"][0, :padding] == -100).all())
                        self.assertTrue((batch["attention_mask"][0, :padding] == 0).all())
                        if name == "qwen":
                            self.assertEqual(batch["image_grid_thw"].shape, (2 * count, 3))
                        else:
                            self.assertEqual(batch["pixel_values"].shape[0], 2)


class LocalLeRobotTests(unittest.TestCase):
    def test_actual_dataset_factory_batches_and_filtered_episode_boundaries(self):
        root = os.environ.get("HELM_SMOKE_DATASET")
        if not root:
            self.skipTest("Set HELM_SMOKE_DATASET to a normalized local smoke dataset")
        from common.datasets.factory import make_dataset
        from common.datasets.lerobot_dataset import LeRobotDataset
        from common.datasets.make_dataloader import make_dataloader
        from common.policies.pi0.configuration_pi0 import PI0Config
        from configs.default import DatasetConfig
        from configs.train import TrainPipelineConfig
        policy = PI0Config(device="cpu", n_action_steps=3, chunk_size=3)
        cfg = TrainPipelineConfig(train_dataset=DatasetConfig(repo_id="local/smoke", root=root, use_imagenet_stats=False),
                                  policy=policy, batch_size=2, num_workers=0, stream_stride=1)
        dataset = make_dataset(cfg, "train")
        self.assertEqual(dataset.num_frames, 330)
        self.assertEqual(dataset.num_episodes, 5)
        batch = next(iter(make_dataloader(cfg, dataset, torch.device("cpu"))))
        self.assertEqual(batch["action"].shape, (2, 3, 7))
        self.assertEqual(batch["observation.state"].shape, (2, 7))
        self.assertEqual(batch["observation.images.table"].shape, (2, 3, 480, 640))
        self.assertTrue(torch.isfinite(batch["action"]).all())
        filtered = LeRobotDataset("local/smoke", root=root, episodes=[4, 1], video_backend="pyav",
                                  delta_timestamps={"action": [0.0, 0.2, 0.4]})
        self.assertEqual(len(filtered), 71 + 45)
        last_first_episode = filtered[70]
        first_second_episode = filtered[71]
        self.assertEqual(last_first_episode["episode_index"].item(), 4)
        self.assertEqual(last_first_episode["action_is_pad"].tolist(), [False, True, True])
        self.assertEqual(first_second_episode["episode_index"].item(), 1)
        self.assertIn("two times", first_second_episode["task"])

    def test_real_frames_build_and_merge_without_train_val_leakage(self):
        frames = os.environ.get("HELM_SMOKE_FRAMES")
        if not frames:
            self.skipTest("Set HELM_SMOKE_FRAMES to extracted frames (synthetic event annotations are used)")
        frames = Path(frames)
        spec = load_taskspec(ROOT / "helm_datasets/taskspecs/press_button_N_times/press_blue_button_1.json")
        # Apply the same schema to all five recordings solely to exercise the
        # pipeline. Event frames are synthetic; these are not training labels.
        spec = replace(spec, task_id="synthetic_smoke", episode_filters=[[{}]])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "input"
            for episode in (frames / "frames_5hz/chunk-000").glob("episode_*"):
                target = source / "frames_5hz/chunk-000" / episode.name
                target.mkdir(parents=True)
                for camera in ("table", "wrist"):
                    (target / camera).symlink_to(episode / camera, target_is_directory=True)
                ids = sorted(int(path.stem.split("_")[-1]) for path in (episode / "table").glob("frame_*.jpg"))
                meta = json.loads((episode / f"{episode.name}.json").read_text())
                meta.update(event_frame_idxs=[ids[-1]], annotation_source="synthetic smoke fixture")
                (target / f"{episode.name}.json").write_text(json.dumps(meta))
            result = build_for_task(spec, tmp / "built", source, 5, 2, 0.4, 0, 50,
                                    ROOT / "helm_datasets/taskspecs/press_button_N_times")
            self.assertEqual(result["n_episodes_train"], 3)
            self.assertEqual(result["n_episodes_val"], 2)
            with mock.patch.object(sys, "argv", ["merge", "--jsonl_root", str(tmp / "built/jsonl_v4"),
                                                 "--out_dir", str(tmp / "merged")]):
                merge_helm_data.main()
            rows = {split: [json.loads(line) for line in (tmp / f"merged/all_{split}.jsonl").read_text().splitlines()]
                    for split in ("train", "val")}
            train_episodes = {(row["chunk"], row["episode"]) for row in rows["train"]}
            val_episodes = {(row["chunk"], row["episode"]) for row in rows["val"]}
            self.assertFalse(train_episodes & val_episodes)
            all_rows = rows["train"] + rows["val"]
            self.assertEqual(len(all_rows), 340)
            self.assertEqual(len({row["uid"] for row in all_rows}), 340)
            self.assertEqual({row["label"] for row in all_rows},
                             {"detect_pos", "detect_neg", "update_intra", "update_transition"})
            for row in all_rows:
                self.assertEqual(yaml.safe_load(row["gt_text"]), row["gt_yaml"])
                self.assertTrue(all(Path(path).is_file() for path in row["images"].values()))


class EntryPointTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("HELM_FULL_IMPORT_CHECK") == "1", "Set HELM_FULL_IMPORT_CHECK=1 for full entrypoint checks")
    def test_documented_entrypoints_support_module_and_script_help(self):
        modules = ["helm_datasets.extract_frames", "helm_datasets.build_helm", "helm_datasets.merge_helm_data",
                   "train.train_helm.train_helm", "train.train_helm_smolvlm.train_helm_smolvlm",
                   "train.train_llp.train_pi0", "evaluate.eval_helm.eval_helm_hlp",
                   "evaluate.eval_helm.eval_helm_hlp_SmolVLM", "evaluate.eval_helm.eval_helm_hlp_ablation",
                   "evaluate.eval_HLP_LLP.eval_real_time_main"]
        env = {**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        for module in modules:
            for command in [[sys.executable, "-m", module, "--help"],
                            [sys.executable, str(ROOT / (module.replace(".", "/") + ".py")), "--help"]]:
                with self.subTest(command=command):
                    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
