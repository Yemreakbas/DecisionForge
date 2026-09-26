"""Turns phase 2 run folders into flat arrays for training the world model.

Every shard of every run is concatenated into one step-indexed table. Episode
boundaries survive as ``window_starts``: a window ``t .. t + H`` is only ever
drawn from inside one episode, so a rollout never crosses a death or a respawn.

Probe labels are derived here, from the recorded self vector, grid and episode
table. ``pos_xz`` is read for one thing only: the display probes that turn an
imagined latent back into a point on the map, so the planner's rollouts can be
drawn as ghost trails. It is a probe *target*, never a model input, and no
score the planner optimises is built on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jevdata import Run

# Probe name -> short description. Order is the probe head's output order and,
# once exported, part of the model contract.
PROBES = {
    "health":        "Health01 now",
    "in_cover":      "InCover01 now",
    "line_of_sight": "HasLineOfSight now",
    "exposure":      "ThreatExposure under the agent (grid centre 2x2) now",
    "in_center":     "InCenterZone now",
    "hurt_soon":     "loses health or dies within the next 1.0 s",
    "death_soon":    "dies within the next 2.0 s",
    "round_won":     "the agent's team wins this round (masked on draws)",
    "pos_x":         "world x, -20..20 m mapped to 0..1 (display only)",
    "pos_z":         "world z, -20..20 m mapped to 0..1 (display only)",
}
DISPLAY_ONLY_PROBES = ("pos_x", "pos_z")
ARENA_HALF_EXTENT = 20.0   # metres; Octagon is 40 x 40
HURT_HORIZON = 10    # ticks
DEATH_HORIZON = 20   # ticks


@dataclass
class Dataset:
    grid: np.ndarray          # (N, C, S, S) float16
    self_vec: np.ndarray      # (N, F) float32
    intent: np.ndarray        # (N,) int64, executed
    labels: np.ndarray        # (N, P) float32
    label_mask: np.ndarray    # (N, P) float32, 0 where a label is undefined
    ep_start: np.ndarray      # (E,) int64, global
    ep_length: np.ndarray     # (E,) int64
    meta: dict                # run.json of the first run

    @property
    def steps(self) -> int:
        return int(self.intent.shape[0])

    def window_starts(self, horizon: int) -> np.ndarray:
        spans = [np.arange(s, s + n - horizon, dtype=np.int64)
                 for s, n in zip(self.ep_start.tolist(), self.ep_length.tolist()) if n > horizon]
        return np.concatenate(spans) if spans else np.zeros(0, dtype=np.int64)


def _labels(meta, grid, self_vec, pos_xz, ep_start, ep_length, ep_end, ep_won):
    fields = {name: i for i, name in enumerate(meta["self_fields"])}
    channels = meta["grid"]["channels"]
    n = self_vec.shape[0]
    c = grid.shape[-1] // 2

    health = self_vec[:, fields["Health01"]]
    exposure = grid[:, channels.index("ThreatExposure"), c - 1:c + 1, c - 1:c + 1].astype(np.float32).mean(axis=(1, 2))

    ep_of_step = np.repeat(np.arange(len(ep_start)), ep_length)
    last = (ep_start + ep_length - 1)[ep_of_step]
    to_end = last - np.arange(n)
    died = (ep_end == 1)[ep_of_step]

    # Minimum health over the next HURT_HORIZON ticks, clipped to the episode.
    future_min = health.copy()
    for k in range(1, HURT_HORIZON + 1):
        future_min = np.minimum(future_min, health[np.minimum(np.arange(n) + k, last)])
    hurt = (future_min < health - 1e-3) | (died & (to_end < HURT_HORIZON))

    won = ep_won[ep_of_step]
    columns = {
        "health": health,
        "in_cover": self_vec[:, fields["InCover01"]],
        "line_of_sight": self_vec[:, fields["HasLineOfSight"]],
        "exposure": exposure,
        "in_center": self_vec[:, fields["InCenterZone"]],
        "hurt_soon": hurt.astype(np.float32),
        "death_soon": (died & (to_end < DEATH_HORIZON)).astype(np.float32),
        "round_won": (won == 1).astype(np.float32),
        "pos_x": pos_xz[:, 0] / (2 * ARENA_HALF_EXTENT) + 0.5,
        "pos_z": pos_xz[:, 1] / (2 * ARENA_HALF_EXTENT) + 0.5,
    }
    labels = np.stack([np.clip(columns[name], 0, 1) for name in PROBES], axis=1).astype(np.float32)
    mask = np.ones_like(labels)
    mask[:, list(PROBES).index("round_won")] = (won >= 0).astype(np.float32)
    return labels, mask


def load(folders, verbose: bool = True, memmap_path=None, memmap_above_gb: float = 4.0) -> Dataset:
    """Concatenates runs. A grid larger than ``memmap_above_gb`` goes to a
    memory-mapped .npy at ``memmap_path`` when one is given: file-backed pages
    do not count against the Windows commit limit, which a 9 GB allocation hit
    on this 32 GB machine, and the OS still caches them in free RAM."""
    runs = [Run.open(folder) for folder in folders]
    meta = runs[0].meta
    contract = (meta["grid"]["channels"], meta["self_fields"], meta["intents"])
    for run in runs[1:]:
        if (run.channels, run.self_fields, run.intents) != contract:
            raise ValueError(f"{run.folder}: observation contract differs from {runs[0].folder.name}")

    # Count first and fill in place: concatenating per-shard arrays would hold two
    # copies of the grid for a moment -- about 11 GB once two arenas are loaded.
    n = 0
    for run in runs:
        for path in run.shard_paths:
            with np.load(path) as z:
                n += int(z["intent"].shape[0])

    c, s = len(meta["grid"]["channels"]), meta["grid"]["size"]
    nbytes = n * c * s * s * 2
    if memmap_path is not None and nbytes > memmap_above_gb * 1e9:
        memmap_path = Path(memmap_path)
        memmap_path.parent.mkdir(parents=True, exist_ok=True)
        grid = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.float16, shape=(n, c, s, s))
        if verbose:
            print(f"  grid {nbytes / 1e9:.1f} GB -> memory-mapped at {memmap_path}")
    else:
        grid = np.empty((n, c, s, s), dtype=np.float16)
    self_vec = np.empty((n, len(meta["self_fields"])), dtype=np.float32)
    intent = np.empty(n, dtype=np.int64)
    labels = np.empty((n, len(PROBES)), dtype=np.float32)
    label_mask = np.empty_like(labels)
    starts, lengths = [], []

    offset = 0
    for run in runs:
        for shard in run.shards():
            k = shard.steps
            lab, msk = _labels(run.meta, shard.grid, shard.self_vec, shard.pos_xz, shard.ep_start.astype(np.int64),
                               shard.ep_length.astype(np.int64), shard.ep_end, shard.ep_won)
            grid[offset:offset + k] = shard.grid
            self_vec[offset:offset + k] = shard.self_vec
            intent[offset:offset + k] = shard.intent
            labels[offset:offset + k] = lab
            label_mask[offset:offset + k] = msk
            starts.append(shard.ep_start.astype(np.int64) + offset)
            lengths.append(shard.ep_length.astype(np.int64))
            offset += k
        if verbose:
            print(f"  loaded {run.folder.name} ({run.meta.get('arena', '?')}): {offset:,} steps so far")

    return Dataset(grid=grid, self_vec=self_vec, intent=intent, labels=labels, label_mask=label_mask,
                   ep_start=np.concatenate(starts), ep_length=np.concatenate(lengths), meta=meta)
