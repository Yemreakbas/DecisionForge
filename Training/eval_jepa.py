"""Measures a trained world model on a held-out run.

    Training\\.venv\\Scripts\\python Training\\eval_jepa.py [checkpoint folder] [--run RUN]

Three questions, each against a baseline that knows nothing:

rollout    Is the imagined latent at t+k closer to the real one than simply
           assuming nothing changes? ``ratio_to_copy`` < 1 means yes.
intent_id  Does the predictor actually listen to the intent? On windows where one
           intent was held for the whole horizon, roll out all ten and check
           whether the true one lands nearest the real future. Chance is 0.1.
probes     Can the heads read the future off an imagined latent? Scored at t+H
           from the imagined latent, from the real one (ceiling), and from the
           latent at t (the "nothing changes" floor).
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

CONTINUOUS_PROBES = {"health", "exposure", "in_cover", "pos_x", "pos_z"}


def planner_score(probes: np.ndarray, names) -> np.ndarray:
    """JevBrain's score of rollouts (B, H, P) -> (B,), with its default weights."""
    from plan_offline import score
    return score(probes, names)
METRES_PER_POS_UNIT = 2 * jepa_data.ARENA_HALF_EXTENT


@torch.no_grad()
def latents(encoder, ds, device, batch: int = 2048) -> torch.Tensor:
    out = []
    for i in range(0, ds.steps, batch):
        g = torch.from_numpy(ds.grid[i:i + batch]).to(device)
        s = torch.from_numpy(ds.self_vec[i:i + batch]).to(device)
        out.append(encoder(g, s).float())
    return torch.cat(out)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = label > 0.5, label <= 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = np.empty(len(score))
    ranks[np.argsort(score, kind="mergesort")] = np.arange(1, len(score) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _probe_scores(pred: np.ndarray, label: np.ndarray, mask: np.ndarray, names) -> dict:
    out = {}
    for j, name in enumerate(names):
        keep = mask[:, j] > 0
        p, y = pred[keep, j], label[keep, j]
        out[name] = {"mae": float(np.abs(p - y).mean()), "auc": auc(p, y)}
    return out


@torch.no_grad()
def evaluate(model, ds, horizon: int, device, probes: bool = True, max_windows: int = 20000) -> dict:
    model.eval()
    H = horizon
    starts = ds.window_starts(H)
    if max_windows and len(starts) > max_windows:
        starts = np.sort(np.random.default_rng(0).choice(starts, max_windows, replace=False))

    zt_all = latents(model.target_encoder, ds, device)          # (N, D) target latents of every step
    names = model.cfg.probes
    offsets = np.arange(H + 1)
    se_pred = torch.zeros(H, device=device)
    se_copy = torch.zeros(H, device=device)
    probe_rows = {"imagined": [], "real": [], "copy": []}
    probe_labels, probe_masks = [], []
    score_imagined, score_real = [], []

    for i in range(0, len(starts), 1024):
        st = starts[i:i + 1024]
        frames = st[:, None] + offsets
        g = torch.from_numpy(ds.grid[st]).to(device)
        s = torch.from_numpy(ds.self_vec[st]).to(device)
        a = torch.from_numpy(ds.intent[frames[:, :H]]).to(device)
        z0 = model.encoder(g, s).float()
        pred = model.predictor.rollout(z0, a).float()             # (b, H, D)
        real = zt_all[torch.from_numpy(frames[:, 1:]).to(device)]  # (b, H, D)
        now = zt_all[torch.from_numpy(st).to(device)]
        se_pred += (pred - real).pow(2).mean(-1).sum(0)
        se_copy += (now[:, None] - real).pow(2).mean(-1).sum(0)
        if probes:
            # The planner's own score of the executed intents, from the imagined
            # future and from the real one -- what the planner actually uses.
            score_imagined.append(planner_score(model.probes(pred).cpu().numpy(), names))
            score_real.append(planner_score(model.probes(real).cpu().numpy(), names))
            probe_rows["imagined"].append(model.probes(pred[:, -1]).cpu().numpy())
            probe_rows["real"].append(model.probes(real[:, -1]).cpu().numpy())
            probe_rows["copy"].append(model.probes(now).cpu().numpy())
            probe_labels.append(ds.labels[frames[:, -1]])
            probe_masks.append(ds.label_mask[frames[:, -1]])

    n = len(starts)
    report = {"windows": int(n), "horizon": H, "rollout": {}}
    for k in range(1, H + 1):
        p, c = (se_pred[k - 1] / n).item(), (se_copy[k - 1] / n).item()
        report["rollout"][str(k)] = {"mse": p, "copy_mse": c, "ratio_to_copy": p / max(c, 1e-12)}

    report["intent_id"] = _intent_identification(model, ds, zt_all, H, device)

    if probes:
        y, m = np.concatenate(probe_labels), np.concatenate(probe_masks)
        report["probes_at_horizon"] = {kind: _probe_scores(np.concatenate(rows), y, m, names)
                                       for kind, rows in probe_rows.items()}
        now_pred = model.probes(zt_all).cpu().numpy()
        report["probes_now"] = _probe_scores(now_pred, ds.labels, ds.label_mask, names)
        si, sr = np.concatenate(score_imagined), np.concatenate(score_real)
        report["score_fidelity"] = {"spearman": spearman(si, sr), "pearson": float(np.corrcoef(si, sr)[0, 1])}
    return report


@torch.no_grad()
def _intent_identification(model, ds, zt_all, H, device, max_windows: int = 8000) -> dict:
    starts = ds.window_starts(H)
    frames = starts[:, None] + np.arange(H)
    constant = (ds.intent[frames] == ds.intent[starts][:, None]).all(axis=1)
    starts = starts[constant]
    if len(starts) > max_windows:
        starts = np.sort(np.random.default_rng(1).choice(starts, max_windows, replace=False))
    num = model.predictor.num_intents
    truth, guess = [], []
    for i in range(0, len(starts), 512):
        st = starts[i:i + 512]
        b = len(st)
        z0 = model.encoder(torch.from_numpy(ds.grid[st]).to(device),
                           torch.from_numpy(ds.self_vec[st]).to(device)).float()
        real = zt_all[torch.from_numpy(st[:, None] + np.arange(1, H + 1)).to(device)]   # (b, H, D)
        z = z0.repeat_interleave(num, 0)
        cand = torch.arange(num, device=device).repeat(b)
        err = torch.zeros(b * num, device=device)
        target = real.repeat_interleave(num, 0)
        for k in range(H):
            z = model.predictor.step(z, cand)
            err += (z - target[:, k]).pow(2).mean(-1)
        guess.append(err.view(b, num).argmin(1).cpu().numpy())
        truth.append(ds.intent[st])
    truth, guess = np.concatenate(truth), np.concatenate(guess)
    per_intent = {}
    for c in range(num):
        sel = truth == c
        if sel.any():
            per_intent[str(c)] = {"n": int(sel.sum()), "recall": float((guess[sel] == c).mean())}
    recalls = [v["recall"] for v in per_intent.values()]
    majority = np.bincount(truth, minlength=num).max() / len(truth)
    return {"windows": int(len(truth)), "accuracy": float((truth == guess).mean()),
            "balanced_accuracy": float(np.mean(recalls)), "chance": 1.0 / num,
            "majority_rate": float(majority), "per_intent": per_intent}


def print_report(r: dict, cfg) -> None:
    print(f"\nheld-out evaluation, {r['windows']:,} windows, horizon {r['horizon']} ticks")
    print("rollout   k   pred mse   copy mse   ratio")
    for k, v in r["rollout"].items():
        if int(k) in (1, 2, 5, 10, 15, 20, 30) or int(k) == r["horizon"]:
            print(f"        {int(k):3d}   {v['mse']:.4f}     {v['copy_mse']:.4f}     {v['ratio_to_copy']:.3f}")
    ii = r["intent_id"]
    print(f"intent id  accuracy {ii['accuracy']:.3f}  balanced {ii['balanced_accuracy']:.3f}  "
          f"(chance {ii['chance']:.2f}, majority {ii['majority_rate']:.2f}, {ii['windows']:,} windows)")
    for c, v in ii["per_intent"].items():
        print(f"    {cfg.intents[int(c)]:18s} recall {v['recall']:.3f}  n={v['n']}")
    if "score_fidelity" in r:
        sf = r["score_fidelity"]
        print(f"score fidelity  planner score, imagined vs real future: spearman {sf['spearman']:.3f}  "
              f"pearson {sf['pearson']:.3f}")
    if "probes_at_horizon" in r:
        ph, pn = r["probes_at_horizon"], r["probes_now"]
        print(f"probes      now      | at t+{r['horizon']}: imagined   real(ceiling)  copy(floor)")
        for name in cfg.probes:
            key = "mae" if name in CONTINUOUS_PROBES else "auc"
            scale, unit = (METRES_PER_POS_UNIT, "m  ") if name in jepa_data.DISPLAY_ONLY_PROBES else (1.0, key)
            vals = [pn[name][key], ph["imagined"][name][key], ph["real"][name][key], ph["copy"][name][key]]
            vals = [v * scale for v in vals]
            print(f"  {name:14s} {unit} {vals[0]:.3f}  |  {vals[1]:.3f}      {vals[2]:.3f}          {vals[3]:.3f}")


def main():
    from jevmodel import load_checkpoint

    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?", help="checkpoint folder (default: newest)")
    p.add_argument("--run", help="run folder to evaluate on (default: the checkpoint's val run)")
    p.add_argument("--horizon", type=int, help="default: the training horizon")
    args = p.parse_args()

    ckpts = sorted((root / "Training" / "checkpoints").glob("*/model.pt"), key=lambda q: q.stat().st_mtime)
    path = Path(args.checkpoint) / "model.pt" if args.checkpoint else (ckpts[-1] if ckpts else None)
    if path is None or not path.exists():
        raise SystemExit("no checkpoint found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(path, device)
    info = ckpt["info"]
    if args.run:
        runs = [Path(args.run)]
    else:
        runs = [find_run_by_name(root / "Datasets", name) for name in info["val_runs"]]
        if any(r is None for r in runs):
            raise SystemExit(f"val runs {info['val_runs']} not all found under Datasets/")
    for run in runs:
        ds = jepa_data.load([run], verbose=False)
        print(f"\ncheckpoint {path.parent.name}, run {run.name} ({ds.meta.get('arena', '?')})")
        if (ds.meta["grid"]["channels"], ds.meta["self_fields"], ds.meta["intents"]) != \
                (model.cfg.channels, model.cfg.self_fields, model.cfg.intents):
            raise SystemExit("run's observation contract does not match the checkpoint")
        print_report(evaluate(model, ds, args.horizon or info["horizon"], device), model.cfg)


if __name__ == "__main__":
    main()
