"""Reader for the phase 2 dataset written by the Unity DatasetRecorder.

A run folder holds ``run.json`` plus ``shard_0000.npz``, ``shard_0001.npz``, ...
Each shard stores whole episodes -- one agent's life in one round -- as
contiguous runs of steps, so a transition ``(obs_t, intent_t, obs_t+1)`` is just
steps ``t`` and ``t + 1`` of one episode.

Model inputs are ``grid`` and ``self`` only. ``pos_xz`` is ground truth kept for
plots and probes: the brains are never told where they are, and a model that
reads it is cheating.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

SCHEMA = "jev-dataset/1"

STEP_ARRAYS = ("grid", "self", "intent", "policy_intent", "explored", "time", "pos_xz")
EPISODE_ARRAYS = ("ep_start", "ep_length", "ep_agent", "ep_team", "ep_captain",
                  "ep_round", "ep_end", "ep_won")


@dataclass
class Shard:
    path: Path
    grid: np.ndarray           # (N, C, S, S) float16, agent-centred, world-axis-aligned
    self_vec: np.ndarray       # (N, F) float32, named by run.json "self_fields"
    intent: np.ndarray         # (N,) uint8, what was executed (after exploration)
    policy_intent: np.ndarray  # (N,) uint8, what the baseline wanted
    explored: np.ndarray       # (N,) uint8, 1 when a detour overrode the baseline
    time: np.ndarray           # (N,) float32, game seconds
    pos_xz: np.ndarray         # (N, 2) float32, DEBUG ONLY -- never a model input
    ep_start: np.ndarray       # (E,) int64
    ep_length: np.ndarray      # (E,) int32
    ep_agent: np.ndarray       # (E,) uint8
    ep_team: np.ndarray        # (E,) uint8
    ep_captain: np.ndarray     # (E,) uint8
    ep_round: np.ndarray       # (E,) int32
    ep_end: np.ndarray         # (E,) uint8: 0 survived the round, 1 died, 2 cut short
    ep_won: np.ndarray         # (E,) int8: 1 won, 0 lost, -1 draw or unknown

    @property
    def steps(self) -> int:
        return int(self.intent.shape[0])

    @property
    def episodes(self) -> int:
        return int(self.ep_start.shape[0])

    def episode_index(self) -> np.ndarray:
        """Episode number of every step."""
        return np.repeat(np.arange(self.episodes), self.ep_length.astype(np.int64))

    def window_starts(self, horizon: int = 1) -> np.ndarray:
        """Steps ``t`` whose window ``t .. t + horizon`` lies inside one episode.

        ``horizon=1`` gives every one-step transition; longer horizons give the
        multi-step rollouts the latent predictor learns to imagine.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        spans = [np.arange(s, s + n - horizon, dtype=np.int64)
                 for s, n in zip(self.ep_start.tolist(), self.ep_length.tolist()) if n > horizon]
        return np.concatenate(spans) if spans else np.zeros(0, dtype=np.int64)


def load_shard(path) -> Shard:
    path = Path(path)
    with np.load(path) as z:
        missing = [name for name in STEP_ARRAYS + EPISODE_ARRAYS if name not in z.files]
        if missing:
            raise ValueError(f"{path.name}: missing arrays {missing}")
        a = {name: z[name] for name in STEP_ARRAYS + EPISODE_ARRAYS}

    return Shard(path=path, grid=a["grid"], self_vec=a["self"], intent=a["intent"],
                 policy_intent=a["policy_intent"], explored=a["explored"], time=a["time"],
                 pos_xz=a["pos_xz"], ep_start=a["ep_start"], ep_length=a["ep_length"],
                 ep_agent=a["ep_agent"], ep_team=a["ep_team"], ep_captain=a["ep_captain"],
                 ep_round=a["ep_round"], ep_end=a["ep_end"], ep_won=a["ep_won"])


@dataclass
class Run:
    folder: Path
    meta: dict
    shard_paths: list

    @classmethod
    def open(cls, folder) -> "Run":
        folder = Path(folder)
        meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if meta.get("schema") != SCHEMA:
            raise ValueError(f"{folder}: schema {meta.get('schema')!r}, expected {SCHEMA!r}")
        return cls(folder=folder, meta=meta, shard_paths=sorted(folder.glob("shard_*.npz")))

    @property
    def channels(self) -> list:
        return self.meta["grid"]["channels"]

    @property
    def self_fields(self) -> list:
        return self.meta["self_fields"]

    @property
    def intents(self) -> list:
        return self.meta["intents"]

    def shards(self) -> Iterator[Shard]:
        for path in self.shard_paths:
            yield load_shard(path)


def find_runs(root) -> list:
    """Every run folder under ``root``, i.e. every folder holding a run.json."""
    return sorted(path.parent for path in Path(root).glob("*/run.json"))


def find_run_by_name(root, name):
    """A run folder by name, up to two levels under ``root`` (Datasets/warehouse/...).

    For finding the runs a checkpoint recorded -- never for choosing training
    data, which stays explicit (``find_runs`` on the folders given).
    """
    for pattern in ("*/run.json", "*/*/run.json"):
        for path in Path(root).glob(pattern):
            if path.parent.name == name:
                return path.parent
    return None
