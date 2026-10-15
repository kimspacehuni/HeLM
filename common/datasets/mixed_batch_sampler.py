"""Balanced label batches with equal iteration counts on every distributed rank."""
from __future__ import annotations

import os
import random

import torch.distributed as dist
from torch.utils.data import DataLoader


class MixedBatchSampler:
    """Sample a global batch sequence, then assign batches round-robin to ranks.

    With replacement, round the global step count up to a multiple of world size.
    Without replacement, cap it by pool capacity and drop an incomplete rank round.
    Both modes therefore expose the same number of batches on every rank.
    """

    def __init__(self, pools, per_batch, steps_per_epoch, seed=0,
                 with_replacement=True, shuffle_within_batch=True):
        self.pools = {key: list(indices) for key, indices in pools.items()}
        self.per_batch = dict(per_batch)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.with_replacement = bool(with_replacement)
        self.shuffle_within_batch = bool(shuffle_within_batch)
        self.epoch = 0
        if self.steps_per_epoch <= 0 or sum(self.per_batch.values()) <= 0:
            raise ValueError("steps_per_epoch and total batch size must be positive")
        for key, count in self.per_batch.items():
            if count < 0:
                raise ValueError(f"per_batch[{key}] must be >= 0")
            if count > 0 and not self.pools.get(key):
                raise ValueError(f"pool '{key}' is empty but per_batch[{key}]={count}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _context(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        # Trainer may query length before it initializes torch.distributed.
        return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))

    def _global_steps(self, world):
        if self.with_replacement:
            return ((self.steps_per_epoch + world - 1) // world) * world
        capacity = min(len(self.pools[key]) // count
                       for key, count in self.per_batch.items() if count > 0)
        steps = min(self.steps_per_epoch, capacity)
        steps -= steps % world
        if steps == 0:
            raise ValueError("Pools cannot supply one complete batch round across all ranks")
        return steps

    def __len__(self):
        _, world = self._context()
        return self._global_steps(world) // world

    def __iter__(self):
        rank, world = self._context()
        rng = random.Random(self.seed + self.epoch)
        pools = {key: list(indices) for key, indices in self.pools.items()}
        if not self.with_replacement:
            for indices in pools.values():
                rng.shuffle(indices)
        offsets = {key: 0 for key in pools}
        for step in range(self._global_steps(world)):
            batch = []
            for key, count in self.per_batch.items():
                if count == 0:
                    continue
                if self.with_replacement:
                    batch.extend(rng.choices(pools[key], k=count))
                else:
                    start = offsets[key]
                    batch.extend(pools[key][start:start + count])
                    offsets[key] += count
            if self.shuffle_within_batch:
                rng.shuffle(batch)
            if step % world == rank:
                yield batch


class EpochDataLoader(DataLoader):
    """Expose the epoch hook Trainer expects on its training dataloader."""

    def set_epoch(self, epoch):
        for sampler in (self.batch_sampler, self.sampler):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)


def cycle_dataloader(dataloader, start_epoch=0):
    """Repeat a loader, advancing sampler seeds and rejecting empty epochs."""
    epoch = start_epoch
    while True:
        for sampler in (dataloader.batch_sampler, dataloader.sampler):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        yielded = False
        for batch in dataloader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("Dataloader produced no batches")
        epoch += 1
