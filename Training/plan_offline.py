"""What would the planner choose? Runs JevBrain's scoring offline on held-out states.

    Training\\.venv\\Scripts\\python Training\\plan_offline.py [checkpoint folder] [--states 5000]

For each sampled state: encode, roll all ten constant intents H ticks forward,
read the probes at every imagined step, and score them exactly as JevBrain does
(weights below mirror ``JevBrain.Weights`` -- keep them in step). Prints the
chosen-intent distribution next to what the baseline did in the same states,
and how far apart the candidates' scores are, which says whether the choice is
driven by the model or by noise.

This cannot say whether a choice is *good* -- only a match can. It catches a
degenerate scoring (one intent always, or all candidates tied) before a match
is spent on it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jepa_data  # noqa: E402
from jevdata import find_run_by_name  # noqa: E402
from jevmodel import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Mirrors JevBrain.Weights defaults.
WEIGHTS = {
    "death_soon": -1.5,
    "hurt_soon": -0.75,
    "exposure": -0.25,
    "in_cover": 0.25,
    "round_won": 1.0,
    "in_center": 0.1,
}
DISCOUNT = 0.93


def score(probes: np.ndarray, names, weights=WEIGHTS, discount=DISCOUNT) -> np.ndarray:
    """probes (..., H, P) -> score (...). Discounted mean over the imagined steps."""
    h = probes.shape[-2]
    w_t = discount ** np.arange(h)
    w_t /= w_t.sum()
    per_step = sum(weights[n] * probes[..., names.index(n)] for n in weights)
    return (per_step * w_t).sum(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?")
    p.add_argument("--states", type=int, default=5000)
    p.add_argument("--horizon", type=int)
    args = p.parse_args()

    ckpts = sorted((ROOT / "Training" / "checkpoints").glob("*/model.pt"), key=lambda q: q.stat().st_mtime)
    path = Path(args.checkpoint) / "model.pt" if args.checkpoint else ckpts[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(path, device)
    info, cfg = ckpt["info"], model.cfg
    H = args.horizon or info["horizon"]
    run = find_run_by_name(ROOT / "Datasets", info["val_runs"][0])
    ds = jepa_data.load([run], verbose=False)
    fields = {n: i for i, n in enumerate(cfg.self_fields)}

    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(ds.steps, min(args.states, ds.steps), replace=False))
    A = len(cfg.intents)
    with torch.no_grad():
        z = model.encoder(torch.from_numpy(ds.grid[idx]).to(device),
                          torch.from_numpy(ds.self_vec[idx]).to(device)).float()
        zz = z.repeat_interleave(A, 0)
        cand = torch.arange(A, device=device).repeat(len(idx))
        steps = []
        for _ in range(H):
            zz = model.predictor.step(zz, cand)
            steps.append(model.probes(zz))
        probes = torch.stack(steps, 1).view(len(idx), A, H, -1).cpu().numpy()

    s = score(probes, cfg.probes)                                   # (states, A)
    # Duel: no allies, so RegroupAlly is vetoed exactly as ConstraintFilter does.
    no_ally = ds.self_vec[idx, fields["AlliesAlive01"]] <= 0
    s[no_ally, cfg.intents.index("RegroupAlly")] = -np.inf
    choice = s.argmax(1)
    finite = np.where(np.isfinite(s), s, np.nan)
    spread = np.nanmax(finite, 1) - np.nanmin(finite, 1)
    top2 = np.sort(np.where(np.isfinite(s), s, -1e9), 1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]

    baseline = ds.intent[idx]
    print(f"{len(idx):,} held-out states from {run.name}, horizon {H}, weights {WEIGHTS}")
    print(f"score spread across candidates: median {np.median(spread):.3f}, "
          f"top-2 margin median {np.median(margin):.4f}")
    print(f"agreement with the executed intent: {(choice == baseline).mean():.3f}")
    print(f"{'intent':18s} planner  baseline")
    for a in range(A):
        print(f"  {cfg.intents[a]:18s} {np.mean(choice == a):6.3f}  {np.mean(baseline == a):6.3f}")

    # Which probe drives the choice: mean of each probe for the chosen vs the average candidate.
    names = [n for n in WEIGHTS]
    chosen = probes[np.arange(len(idx)), choice]                   # (states, H, P)
    print("probe means over the horizon:  chosen   all candidates")
    for n in names:
        j = cfg.probes.index(n)
        print(f"  {n:12s} {chosen[..., j].mean():.3f}    {probes[..., j].mean():.3f}")


if __name__ == "__main__":
    main()
