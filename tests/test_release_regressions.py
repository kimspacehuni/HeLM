"""CPU regressions; no checkpoints, Transformers, or robot hardware required.

Some entrypoints eagerly import their entire GPU/robot stack. Extract their
actual definitions for these focused tests rather than mocking that stack.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import logging
import os
from pathlib import Path
import random
import runpy
import sys
import tempfile
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase, main, mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def definitions(path, names, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    env = {} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), env)
    return env


def mixed_sampler(path):
    tree = ast.parse((ROOT / path).read_text())
    if any(isinstance(n, ast.ClassDef) and n.name == "MixedBatchSampler" for n in tree.body):
        return definitions(path, {"MixedBatchSampler"}, {"random": random, "dist": dist})["MixedBatchSampler"]
    from common.datasets.mixed_batch_sampler import MixedBatchSampler
    return MixedBatchSampler


class MixedSamplerTests(TestCase):
    paths = (
        "train/train_helm/train_helm.py",
        "train/train_helm_smolvlm/train_helm_smolvlm.py",
    )

    def batches(self, path, rank, world, steps=5, replacement=True, pool_size=40):
        with mock.patch.object(dist, "is_initialized", return_value=True), \
             mock.patch.object(dist, "get_rank", return_value=rank), \
             mock.patch.object(dist, "get_world_size", return_value=world):
            sampler = mixed_sampler(path)(
                {"detect_pos": list(range(pool_size)), "detect_neg": list(range(100, 140))},
                {"detect_pos": 2, "detect_neg": 1}, steps, with_replacement=replacement,
            )
            return sampler, list(sampler), len(sampler)

    def test_equal_steps_for_odd_global_batch_count(self):
        for path in self.paths:
            with self.subTest(path=path):
                counts = []
                for rank in range(2):
                    _, batches, length = self.batches(path, rank, 2)
                    self.assertEqual(len(batches), length)
                    self.assertTrue(all(len(b) == 3 for b in batches))
                    counts.append(len(batches))
                self.assertEqual(counts, [3, 3])

    def test_no_replacement_respects_pool_capacity_and_rank_balance(self):
        for path in self.paths:
            with self.subTest(path=path):
                combined = []
                counts = []
                for rank in range(2):
                    _, batches, length = self.batches(path, rank, 2, replacement=False, pool_size=10)
                    self.assertEqual(len(batches), length)
                    counts.append(len(batches))
                    combined.extend(i for b in batches for i in b)
                self.assertEqual(counts, [2, 2])
                self.assertEqual(len(combined), len(set(combined)))

    def test_epoch_changes_order_but_is_reproducible(self):
        for path in self.paths:
            with self.subTest(path=path):
                sampler, first, _ = self.batches(path, 0, 1)
                with mock.patch.object(dist, "is_initialized", return_value=False), mock.patch.dict(os.environ, {}, clear=True):
                    sampler.set_epoch(1)
                    second = list(sampler)
                    self.assertNotEqual(first, second)
                    sampler.set_epoch(1)
                    self.assertEqual(second, list(sampler))


class MergeTests(TestCase):
    def run_merge(self, root, output, *extra):
        argv = ["merge_helm_data.py", "--jsonl_root", str(root), "--out_dir", str(output), *extra]
        with mock.patch.object(sys, "argv", argv):
            runpy.run_path(str(ROOT / "helm_datasets/merge_helm_data.py"), run_name="__main__")
        return [json.loads(line) for line in (output / "all_train.jsonl").read_text().splitlines()]

    def test_upweight_keeps_duplicates_from_weight_but_removes_input_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            task = root / "task_a"
            task.mkdir(parents=True)
            (task / "train-00000.jsonl").write_text('\n'.join(json.dumps({"uid": uid}) for uid in ["a", "a", "b"]))
            rows = self.run_merge(root, Path(tmp) / "output", "--weights", "task_a=2")
            self.assertEqual(sorted(r["uid"] for r in rows), ["a", "a", "b", "b"])

    def test_default_weight_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            task = root / "task_a"
            task.mkdir(parents=True)
            (task / "train-00000.jsonl").write_text('{"uid":"a"}\n{"uid":"a"}\n')
            rows = self.run_merge(root, Path(tmp) / "output")
            self.assertEqual(rows, [{"uid": "a"}])


class CliTests(TestCase):
    def test_false_is_not_true_in_entrypoints(self):
        paths = [*MixedSamplerTests.paths,
                 "evaluate/eval_helm/eval_helm_hlp.py",
                 "evaluate/eval_helm/eval_helm_hlp_SmolVLM.py",
                 "evaluate/eval_helm/eval_helm_hlp_ablation.py"]
        for path in paths:
            tree = ast.parse((ROOT / path).read_text())
            # Execute the actual argument declarations, without loading a model.
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                     and n.value.func.attr == "add_argument"]
            parser = argparse.ArgumentParser()
            namespace = {"ap": parser, "argparse": argparse}
            try:
                from common.utils.argparse_utils import str_to_bool
                namespace["str_to_bool"] = str_to_bool
            except ImportError:
                pass
            for call in calls:
                if call.value.args and isinstance(call.value.args[0], ast.Constant) and call.value.args[0].value == "--bf16":
                    exec(compile(ast.Module(body=[call], type_ignores=[]), path, "exec"), namespace)
            with self.subTest(path=path):
                self.assertIs(parser.parse_args(["--bf16", "False"]).bf16, False)

    def test_explicit_boolean_values_and_invalid_input(self):
        from common.utils.argparse_utils import str_to_bool
        for value in ("false", "False", "0", "no", "off"):
            self.assertIs(str_to_bool(value), False)
        for value in ("true", "TRUE", "1", "yes", "on"):
            self.assertIs(str_to_bool(value), True)
        with self.assertRaises(argparse.ArgumentTypeError):
            str_to_bool("typo")


class EpisodeSamplerTests(TestCase):
    def sampler(self, **kwargs):
        cls = definitions("common/datasets/sampler.py", {"EpisodeAwareBatchSampler"}, {"torch": torch})["EpisodeAwareBatchSampler"]
        return cls({"from": torch.tensor([0, 3, 5]), "to": torch.tensor([3, 5, 10])},
                   batch_size=4, shuffle_episodes=False, stride=1, **kwargs)

    def test_cross_episode_fill_visits_every_frame_once_without_mutation(self):
        sampler = self.sampler(allow_cross_episode_fill=True)
        expected = list(range(10))
        for _ in range(2):
            batches = list(sampler)
            self.assertEqual([i for b in batches for i in b], expected)
            self.assertEqual(len(batches), len(sampler))

    def test_single_episode_batches_report_exact_length(self):
        sampler = self.sampler(allow_cross_episode_fill=False)
        self.assertEqual(len(sampler), len(list(sampler)))

    def test_drop_last_keeps_later_full_episodes(self):
        sampler = self.sampler(allow_cross_episode_fill=False, drop_last=True)
        self.assertEqual(list(sampler), [[5, 6, 7, 8]])
        self.assertEqual(len(sampler), 1)

    def test_invalid_stride_is_rejected(self):
        cls = definitions("common/datasets/sampler.py", {"EpisodeAwareBatchSampler"}, {"torch": torch})["EpisodeAwareBatchSampler"]
        with self.assertRaises(ValueError):
            cls({"from": torch.tensor([0]), "to": torch.tensor([10])}, batch_size=2, stride=0)


class LoaderTests(TestCase):
    def test_trainer_epoch_hook_reaches_sampler(self):
        from common.datasets.mixed_batch_sampler import EpochDataLoader, MixedBatchSampler
        sampler = MixedBatchSampler({"a": list(range(50))}, {"a": 2}, 5)
        loader = EpochDataLoader(list(range(50)), batch_sampler=sampler)
        first = [batch.tolist() for batch in loader]
        loader.set_epoch(1)
        self.assertNotEqual(first, [batch.tolist() for batch in loader])

    def test_cycle_advances_epoch_and_rejects_empty_data(self):
        from common.datasets.mixed_batch_sampler import EpochDataLoader, MixedBatchSampler, cycle_dataloader
        sampler = MixedBatchSampler({"a": list(range(50))}, {"a": 2}, 1)
        loader = EpochDataLoader(list(range(50)), batch_sampler=sampler)
        iterator = cycle_dataloader(loader)
        next(iterator)
        next(iterator)
        self.assertEqual(sampler.epoch, 1)
        with self.assertRaises(ValueError):
            next(cycle_dataloader(torch.utils.data.DataLoader([])))


class DebugImageTests(TestCase):
    def test_disabled_debug_does_not_import_matplotlib(self):
        function = definitions("evaluate/eval_HLP_LLP/eval_real_time_main.py", {"_show_debug_image"})["_show_debug_image"]
        with mock.patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            function(object(), 0)

    def test_enabled_debug_is_nonblocking_and_closes_figure(self):
        function = definitions("evaluate/eval_HLP_LLP/eval_real_time_main.py", {"_show_debug_image"})["_show_debug_image"]
        plot = mock.Mock()
        figure, axis = object(), mock.Mock()
        plot.subplots.return_value = (figure, axis)
        matplotlib = mock.Mock(pyplot=plot)
        with mock.patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": plot}):
            function(object(), 0, enabled=True)
        plot.show.assert_called_once_with(block=False)
        plot.close.assert_called_once_with(figure)


class FrameExtractionTests(TestCase):
    def test_fractional_second_tail_is_preserved(self):
        import cv2
        import numpy as np
        from helm_datasets.extract_frames import extract_frames_at_fps
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48))
            self.assertTrue(writer.isOpened())
            try:
                for i in range(7):
                    writer.write(np.full((48, 64, 3), i * 30, dtype=np.uint8))
            finally:
                writer.release()
            output = Path(tmp) / "frames"
            self.assertEqual(extract_frames_at_fps(source, output, False, 5.0), 7)
            self.assertEqual(len(list(output.glob("frame_*.jpg"))), 7)
            self.assertEqual(extract_frames_at_fps(source, output, False, 5.0), 7)
            tail = cv2.imread(str(output / "frame_000006.jpg"))
            self.assertAlmostEqual(float(tail.mean()), 180.0, delta=5.0)

    def test_invalid_fps_is_rejected(self):
        from helm_datasets.extract_frames import extract_frames_at_fps
        for fps in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                extract_frames_at_fps(Path("missing.mp4"), Path("unused"), False, fps)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.cache_cleared = 0

    def forward(self, batch):
        if isinstance(batch, dict):
            batch = batch["values"]
        loss = (self.weight * batch - 1).square().mean()
        return loss, {}

    def clear_cache(self):
        self.cache_cleared += 1


def ddp_worker(rank, init_method, output):
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=init_method,
                            timeout=datetime.timedelta(seconds=30))
    try:
        factory = definitions("common/policies/factory.py", {"dist_policy"},
                              {"torch": torch, "DDP": DDP, "FSDP": FSDP})
        raw = TinyPolicy()
        wrapped, _ = factory["dist_policy"](raw, "ddp", True, rank, torch.device("cpu"))
        assert isinstance(wrapped, DDP), "dist_policy must return the DDP wrapper"
        helpers = definitions("common/utils/model_utils.py", {"compute_param_norm"}, {"torch": torch})
        helpers.update(torch=torch, DDP=DDP, FSDP=FSDP, time=time, nullcontext=nullcontext,
                       get_device_from_parameters=lambda p: next(p.parameters()).device,
                       has_method=lambda p, name: callable(getattr(p, name, None)))
        functions = definitions("train/train_llp/train_pi0.py", {"update_policy", "test_policy"}, helpers)
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        functions["update_policy"](SimpleNamespace(), wrapped, torch.tensor([1.0 if rank == 0 else 3.0]),
                                     optimizer, 100.0, scaler)
        assert raw.cache_cleared == 1
        # Two different local gradients (-2, -6) must average to -4.
        assert abs(raw.weight.item() - 0.4) < 1e-6, raw.weight.item()
        (Path(output) / str(rank)).write_text(str(raw.weight.item()))
        functions["test_policy"](SimpleNamespace(), wrapped, torch.tensor([1.0]))
    finally:
        dist.destroy_process_group()


class DistributedTests(TestCase):
    def test_real_two_process_cpu_gradient_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(ddp_worker, args=(f"file://{tmp}/init", tmp), nprocs=2, join=True)
            self.assertEqual((Path(tmp) / "0").read_text(), (Path(tmp) / "1").read_text())


class TrainingIntegrationTests(TestCase):
    def test_transformers_trainer_advances_sampler_epoch(self):
        from transformers import Trainer, TrainingArguments
        from common.datasets.mixed_batch_sampler import EpochDataLoader, MixedBatchSampler

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.seen = []

            def forward(self, values):
                self.seen.append(values.tolist())
                return {"loss": (self.weight - values.float().mean()).square().mean()}

        for path in MixedSamplerTests.paths:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as tmp:
                trainer_class = definitions(path, {"MixedBatchTrainer"},
                                            {"Trainer": Trainer, "EpochDataLoader": EpochDataLoader})["MixedBatchTrainer"]
                model = Model()
                sampler = MixedBatchSampler({"a": list(range(50))}, {"a": 2}, 3)
                trainer = trainer_class(
                    model=model,
                    args=TrainingArguments(output_dir=tmp, use_cpu=True, num_train_epochs=2,
                                           per_device_train_batch_size=2, report_to=[],
                                           save_strategy="no", logging_strategy="no",
                                           disable_tqdm=True, remove_unused_columns=False),
                    train_dataset=list(range(50)),
                    data_collator=lambda features: {"values": torch.tensor(features)},
                    train_batch_sampler=sampler,
                )
                trainer.train()
                self.assertEqual(trainer.state.global_step, 6)
                self.assertEqual(sampler.epoch, 1)
                self.assertNotEqual(model.seen[:3], model.seen[3:])

    def test_llp_trainer_uses_wrapper_for_forward_and_raw_policy_for_checkpoint(self):
        from common.datasets.mixed_batch_sampler import cycle_dataloader

        raw = TinyPolicy()

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.module = raw
                self.forward_calls = 0

            def forward(self, batch):
                self.forward_calls += 1
                return self.module(batch)

            def clear_cache(self):
                self.module.clear_cache()

        class Dataset:
            meta = None
            num_frames = 1
            num_episodes = 1

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {"values": torch.tensor([1.0])}

        class Tracker(SimpleNamespace):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def step(self):
                pass

        wrapped = Wrapper()
        saved = []
        optimizer_inputs = []

        def make_optimizer(cfg, policy):
            optimizer_inputs.append(policy)
            return torch.optim.SGD(policy.parameters(), lr=0.1), None

        env = definitions("common/utils/model_utils.py", {"compute_param_norm"}, {"torch": torch})
        env.update(torch=torch, DDP=DDP, dist=dist, os=os, logging=logging, time=time,
                   datetime=datetime, pformat=lambda value: str(value), colored=lambda value, *args, **kwargs: value,
                   get_device_from_parameters=lambda p: next(p.parameters()).device,
                   has_method=lambda p, name: callable(getattr(p, name, None)), nullcontext=nullcontext,
                   get_safe_torch_device=lambda *args, **kwargs: torch.device("cpu"),
                   is_ddp_master=lambda distributed, rank: not distributed or rank == 0,
                   format_big_number=lambda value: str(value),
                   make_dataset=lambda *args, **kwargs: Dataset(),
                   make_dataloader=lambda cfg, dataset, device: torch.utils.data.DataLoader(dataset, batch_size=1),
                   cycle_dataloader=cycle_dataloader, make_policy=lambda **kwargs: raw,
                   wrap_policy=lambda **kwargs: (raw, ""), dist_policy=lambda **kwargs: (wrapped, ""),
                   make_optimizer_and_scheduler=make_optimizer, GradScaler=torch.amp.GradScaler,
                   set_seed=lambda value: torch.manual_seed(value), AverageMeter=lambda *args: None,
                   MetricsTracker=Tracker, batch_to_device=lambda batch, device: batch,
                   get_step_checkpoint_dir=lambda directory, steps, step: directory,
                   save_checkpoint=lambda directory, step, cfg, policy, *args: saved.append(policy),
                   update_last_checkpoint=lambda directory: None, log_wandb_tracker=lambda *args, **kwargs: None)
        functions = definitions("train/train_llp/train_pi0.py", {"train", "update_policy", "test_policy"}, env)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(validate=lambda: None, to_dict=lambda: {}, dist_mode="none",
                                  policy=SimpleNamespace(device="cpu", use_amp=False),
                                  seed=0, wandb=SimpleNamespace(enable=False, project=None),
                                  method=SimpleNamespace(core="vanilla"), gradient_checkpointing=False,
                                  resume=False, steps=1, batch_size=1, log_freq=0, save_freq=1,
                                  test_freq=1, k_plot_freq=0, save_checkpoint=True, output_dir=Path(tmp),
                                  optimizer=SimpleNamespace(grad_clip_norm=100.0))
            functions["train"](cfg)
        self.assertEqual(wrapped.forward_calls, 2)
        self.assertEqual(saved, [raw])
        self.assertEqual(optimizer_inputs, [raw])

    def test_fsdp_rejected_before_model_creation(self):
        function = definitions("train/train_llp/train_pi0.py", {"train"})["train"]
        with self.assertRaisesRegex(NotImplementedError, "FSDP"):
            function(SimpleNamespace(validate=lambda: None, dist_mode="fsdp"))


if __name__ == "__main__":
    main()
