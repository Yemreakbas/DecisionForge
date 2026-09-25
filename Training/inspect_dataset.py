"""Summarise and sanity-check a phase 2 dataset run.

    Training/.venv/Scripts/python Training/inspect_dataset.py              (newest run)
    Training/.venv/Scripts/python Training/inspect_dataset.py <run folder>

Exits 1 if any check fails, so it can gate a training job.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from jevdata import Run, find_runs

DATASETS = Path(__file__).resolve().parent.parent / "Datasets"


def pick_run(argument: str | None) -> Path | None:
    if argument:
        return Path(argument)
    runs = find_runs(DATASETS)
    return max(runs, key=lambda p: (p / "run.json").stat().st_mtime) if runs else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise and sanity-check a phase 2 dataset run.")
    parser.add_argument("run", nargs="?", help="run folder (default: newest under Datasets/)")
    folder = pick_run(parser.parse_args().run)
    if folder is None:
        print(f"No runs under {DATASETS}")
        return 1

    run = Run.open(folder)
    meta = run.meta
    channels, fields, intents = run.channels, run.self_fields, run.intents
    size = meta["grid"]["size"]
    tick = 1.0 / float(meta["tick_hz"])

    problems: list[str] = []

    def check(ok, message: str) -> None:
        if not ok:
            problems.append(message)

    steps = episodes = explored = 0
    executed = np.zeros(len(intents), np.int64)
    wanted = np.zeros(len(intents), np.int64)
    ends = np.zeros(3, np.int64)
    wins = np.zeros(2, np.int64)
    draws = 0
    ch_min = np.full(len(channels), np.inf)
    ch_max = np.full(len(channels), -np.inf)
    ch_sum = np.zeros(len(channels))
    ch_nonzero = np.zeros(len(channels))
    self_min = np.full(len(fields), np.inf)
    self_max = np.full(len(fields), -np.inf)
    obstacle_values: set = set()
    visibility_values: set = set()
    dts, lengths_all = [], []

    for shard in run.shards():
        n, e, name = shard.steps, shard.episodes, shard.path.name

        check(shard.grid.shape == (n, len(channels), size, size), f"{name}: grid shape {shard.grid.shape}")
        check(shard.self_vec.shape == (n, len(fields)), f"{name}: self shape {shard.self_vec.shape}")
        check(shard.pos_xz.shape == (n, 2), f"{name}: pos_xz shape {shard.pos_xz.shape}")
        for array in ("intent", "policy_intent", "explored", "time"):
            check(getattr(shard, array).shape == (n,), f"{name}: {array} shape {getattr(shard, array).shape}")

        # Episodes must tile the shard exactly, in order, with nothing empty.
        starts = shard.ep_start.astype(np.int64)
        lengths = shard.ep_length.astype(np.int64)
        check(e > 0 and starts[0] == 0, f"{name}: first episode does not start at step 0")
        check(np.array_equal(starts[1:], starts[:-1] + lengths[:-1]), f"{name}: episodes are not contiguous")
        check(lengths.sum() == n, f"{name}: episode lengths sum to {lengths.sum()}, shard has {n} steps")
        check((lengths > 0).all(), f"{name}: empty episode")
        lengths_all.append(lengths)

        grid = shard.grid.astype(np.float32)
        check(np.isfinite(grid).all(), f"{name}: non-finite grid values")
        check((grid >= 0).all(), f"{name}: negative grid values")
        check(np.isfinite(shard.self_vec).all(), f"{name}: non-finite self values")
        check(int(shard.intent.max(initial=0)) < len(intents), f"{name}: intent out of range")
        check(int(shard.policy_intent.max(initial=0)) < len(intents), f"{name}: policy intent out of range")

        per_channel = grid.transpose(1, 0, 2, 3).reshape(len(channels), -1)
        ch_min = np.minimum(ch_min, per_channel.min(axis=1))
        ch_max = np.maximum(ch_max, per_channel.max(axis=1))
        ch_sum += per_channel.sum(axis=1, dtype=np.float64)
        ch_nonzero += (per_channel != 0).sum(axis=1)
        obstacle_values.update(np.unique(per_channel[channels.index("ObstacleHeight")]).tolist())
        visibility_values.update(np.unique(per_channel[channels.index("Visibility")]).tolist())
        self_min = np.minimum(self_min, shard.self_vec.min(axis=0))
        self_max = np.maximum(self_max, shard.self_vec.max(axis=0))

        # Consecutive steps of one episode should be exactly one tactical tick apart.
        index = shard.episode_index()
        dts.append(np.diff(shard.time)[index[1:] == index[:-1]])

        executed += np.bincount(shard.intent, minlength=len(intents))[:len(intents)]
        wanted += np.bincount(shard.policy_intent, minlength=len(intents))[:len(intents)]
        explored += int(shard.explored.sum())
        ends += np.bincount(shard.ep_end, minlength=3)[:3]

        # A round counts once per team, however many of its agents won.
        rounds = set(shard.ep_round.tolist())
        winners = {(r, t) for r, t, w in zip(shard.ep_round.tolist(), shard.ep_team.tolist(),
                                             shard.ep_won.tolist()) if w == 1}
        for _, team in winners:
            wins[team] += 1
        draws += len(rounds - {r for r, _ in winners})

        steps += n
        episodes += e

    dt = np.concatenate(dts) if dts else np.zeros(0)
    lengths = np.concatenate(lengths_all) if lengths_all else np.zeros(0)

    check(steps > 0, "no steps recorded")
    check(dt.size == 0 or (dt > 0).all(), "time does not increase inside an episode")
    on_tick = float(np.mean(np.abs(dt - tick) < 0.25 * tick)) if dt.size else 0.0
    check(dt.size == 0 or on_tick > 0.95, f"only {on_tick:.1%} of steps are one tick ({tick:.3f} s) apart")
    check(obstacle_values <= {0.0, 0.5, 1.0}, f"ObstacleHeight holds {sorted(obstacle_values)[:6]}")
    check(visibility_values <= {0.0, 1.0}, f"Visibility holds {sorted(visibility_values)[:6]}")
    for bounded in ("CoverQuality", "ThreatExposure"):
        check(ch_max[channels.index(bounded)] <= 1.0 + 1e-3, f"{bounded} exceeds 1")
    if meta.get("complete"):
        totals = meta["totals"]
        check(totals["steps"] == steps, f"run.json says {totals['steps']} steps, shards hold {steps}")
        check(totals["episodes"] == episodes, f"run.json says {totals['episodes']} episodes, shards hold {episodes}")
        check(totals["shards"] == len(run.shard_paths), f"run.json says {totals['shards']} shards, found {len(run.shard_paths)}")

    cells = max(steps, 1) * size * size
    print(f"run      {folder}")
    print(f"match    {meta['arena']}, {meta['format']}, {' vs '.join(meta['brains'])}, seed {meta['seed']}, "
          f"{'complete' if meta.get('complete') else 'INCOMPLETE'}")
    print(f"data     {len(run.shard_paths)} shard(s), {meta['totals']['rounds']} rounds, {episodes} episodes, "
          f"{steps:,} steps = {steps * tick / 60:.1f} min of agent experience")
    if lengths.size:
        print(f"episode  mean {lengths.mean() * tick:.1f} s, median {np.median(lengths) * tick:.1f} s, "
              f"longest {lengths.max() * tick:.1f} s; ended: survived {ends[0]}, died {ends[1]}, cut {ends[2]}")
    print(f"rounds   team A won {wins[0]}, team B won {wins[1]}, no winner {draws}")
    if dt.size:
        print(f"tick     dt median {np.median(dt):.4f} s, {on_tick:.1%} within 25% of {tick:.3f} s")
    print(f"explore  {explored / max(steps, 1):.1%} of steps were detours")

    print("\nintent                 executed   baseline")
    for i, intent in enumerate(intents):
        print(f"  {intent:<20} {executed[i] / max(steps, 1):>8.1%}   {wanted[i] / max(steps, 1):>8.1%}")

    print("\nchannel                  min      max     mean   nonzero")
    for i, channel in enumerate(channels):
        print(f"  {channel:<20} {ch_min[i]:>6.2f} {ch_max[i]:>8.2f} {ch_sum[i] / cells:>8.3f} "
              f"{ch_nonzero[i] / cells:>8.1%}")

    print("\nself field                       min      max")
    for i, field in enumerate(fields):
        print(f"  {field:<28} {self_min[i]:>8.3f} {self_max[i]:>8.3f}")

    if problems:
        print("\nFAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nchecks   all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
