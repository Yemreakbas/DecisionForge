"""Phase 3: train the JEPA world model on phase 2 runs.

    Training\\.venv\\Scripts\\python Training\\train_jepa.py [--steps 12000] [--horizon 15]

Stage 1, self-supervised. Online encoder reads obs_t; the predictor rolls it
forward through the executed intents for H ticks; each imagined latent is pulled
toward the EMA target encoder's latent of the real obs_t+k. No labels, no reward.
VICReg variance/covariance terms on the online latents rule out collapse.

Stage 2, probes. The encoders are frozen and small heads learn to read health,
cover, exposure, imminent damage and death, and round outcome off *target*
latents -- the space the predictor is trained to land in, so the same heads can
score an imagined future in the planner.

The held-out run (default: seed 1004) never touches either stage.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jepa_data  # noqa: E402
from eval_jepa import evaluate, latents, print_report  # noqa: E402
from jevdata import find_run_by_name, find_runs  # noqa: E402
from jevmodel import ModelConfig, WorldModel, load_checkpoint, save_checkpoint, vicreg_regulariser  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=[str(ROOT / "Datasets")],
                   help="folders whose run folders are used (one level down each), e.g. "
                        "Datasets Datasets/warehouse for a two-arena model")
    p.add_argument("--val-seed", type=int, nargs="+", default=[1004],
                   help="runs held out for evaluation, one seed each -- one per arena")
    p.add_argument("--gpu-data-gb", type=float, default=4.0,
                   help="keep the grid on the GPU only below this size; above it, gather per batch from RAM")
    p.add_argument("--horizon", type=int, default=15, help="rollout length in 0.1 s ticks")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--var-weight", type=float, default=1.0)
    p.add_argument("--cov-weight", type=float, default=0.04)
    p.add_argument("--ema-start", type=float, default=0.99)
    p.add_argument("--ema-end", type=float, default=0.999)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--probe-epochs", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="checkpoint folder name (default: timestamp)")
    p.add_argument("--refit-probes", metavar="CHECKPOINT_FOLDER",
                   help="keep that checkpoint's world model and refit only the probes")
    return p.parse_args()


def split_runs(folders, val_seeds):
    runs = [run for folder in folders for run in find_runs(Path(folder))]
    val = []
    for seed in val_seeds:
        matches = [r for r in runs if r.name.endswith(f"_seed{seed}")]
        if len(matches) != 1:
            raise SystemExit(f"need exactly one run with seed {seed} under {folders}, found {len(matches)}")
        val += matches
    train = [r for r in runs if r not in val]
    if not train:
        raise SystemExit(f"no training runs left under {folders}")
    return train, val


def cosine(step, total, start, end):
    """start at step 0, end at step total, half-cosine in between."""
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * min(step / total, 1.0)))


def train_world_model(model, train, args, device, val):
    H, B = args.horizon, args.batch
    # One arena's grid (3.3 GB fp16) fits the 8 GB laptop GPU next to the model;
    # two do not, and then each batch is gathered from RAM instead (~60 MB a step).
    on_gpu = device.type == "cuda" and train.grid.nbytes < args.gpu_data_gb * 1e9
    grid = torch.from_numpy(train.grid)
    if on_gpu:
        grid = grid.to(device)
    selfv = torch.from_numpy(train.self_vec).to(device)
    intent = torch.from_numpy(train.intent).to(device)
    starts = torch.from_numpy(train.window_starts(H)).to(device)
    offsets = torch.arange(H + 1, device=device)
    print(f"stage 1: {len(starts):,} windows of {H} ticks, batch {B}, {args.steps} steps "
          f"(~{args.steps * B / len(starts):.1f} epochs), grid {train.grid.nbytes / 1e9:.1f} GB "
          f"{'on the GPU' if on_gpu else 'in RAM'}")

    params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warmup = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * cosine(s, args.steps, 1.0, 0.05))

    log, t0, acc = [], time.time(), None
    for step in range(1, args.steps + 1):
        frames = starts[torch.randint(len(starts), (B,), device=device)][:, None] + offsets
        g = grid[frames] if on_gpu else grid[frames.cpu()].to(device, non_blocking=True)
        s, a = selfv[frames], intent[frames[:, :H]]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z0 = model.encoder(g[:, 0], s[:, 0])
            with torch.no_grad():
                zt = model.target_encoder(g[:, 1:].flatten(0, 1), s[:, 1:].flatten(0, 1)).view(B, H, -1)
            pred = model.predictor.rollout(z0, a)
        pred_loss = F.mse_loss(pred.float(), zt.float())
        var_loss, cov_loss, std = vicreg_regulariser(z0)
        loss = pred_loss + args.var_weight * var_loss + args.cov_weight * cov_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        model.update_target(cosine(step, args.steps, args.ema_start, args.ema_end))

        row = torch.stack([pred_loss.detach(), var_loss.detach(), cov_loss.detach(), std.detach()])
        acc = row if acc is None else acc + row
        if step % 100 == 0:
            m = (acc / 100).tolist()
            acc = None
            print(f"  step {step:6d}  pred {m[0]:.4f}  var {m[1]:.4f}  cov {m[2]:.4f}  "
                  f"std {m[3]:.3f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:5.0f}s", flush=True)
            log.append({"step": step, "pred": m[0], "var": m[1], "cov": m[2], "std": m[3]})
        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate(model, val, H, device, probes=False, max_windows=4096)
            ks = [k for k in (1, 5, 10, H) if k <= H]
            print("  val  " + "  ".join(f"k={k}: {r['rollout'][str(k)]['ratio_to_copy']:.3f}" for k in ks)
                  + f"  intent-id {r['intent_id']['accuracy']:.3f}", flush=True)
            log[-1]["val"] = r
            model.train()
    del grid, selfv, intent
    return log


def train_probes(model, train, args, device):
    print(f"stage 2: probes {list(jepa_data.PROBES)}")
    z = latents(model.target_encoder, train, device)
    y = torch.from_numpy(train.labels).to(device)
    m = torch.from_numpy(train.label_mask).to(device)
    opt = torch.optim.AdamW(model.probes.parameters(), lr=1e-3, weight_decay=1e-4)
    n = z.shape[0]
    for epoch in range(args.probe_epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, 2048):
            idx = perm[i:i + 2048]
            loss = (F.binary_cross_entropy_with_logits(model.probes.logits(z[idx]), y[idx], reduction="none")
                    * m[idx]).sum() / m[idx].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  epoch {epoch + 1}  bce {total / n:.4f}")


def refit_probes(args, device):
    """Keep a trained encoder and predictor, refit only the probe heads.

    The probe list is part of how the planner scores, so it will change more often
    than the world model does; this avoids retraining the world model each time.
    """
    out = Path(args.refit_probes)
    old, ckpt = load_checkpoint(out / "model.pt", device)
    info = ckpt["info"]
    cfg = old.cfg
    cfg.probes = list(jepa_data.PROBES)
    model = WorldModel(cfg).to(device)
    missing, unexpected = model.load_state_dict(
        {k: v for k, v in ckpt["state"].items() if not k.startswith("probes.")}, strict=False)
    if unexpected or any(not k.startswith("probes.") for k in missing):
        raise SystemExit(f"checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
    model.eval()

    def locate(name):
        run = find_run_by_name(ROOT / "Datasets", name)
        if run is None:
            raise SystemExit(f"run {name} recorded by the checkpoint is not under Datasets/")
        return run

    print("train runs:")
    train = jepa_data.load([locate(name) for name in info["train_runs"]])
    train_probes(model, train, args, device)
    reports = evaluate_each(model, [locate(name) for name in info["val_runs"]], info["horizon"], device)

    prior = out / "report.json"
    log = json.loads(prior.read_text(encoding="utf-8")).get("log", []) if prior.exists() else []
    return model, info, reports, log, out


def evaluate_each(model, val_runs, horizon, device) -> dict:
    """One full report per held-out run: arenas are reported apart, never pooled."""
    reports = {}
    for run in val_runs:
        ds = jepa_data.load([run], verbose=False)
        print(f"\n=== {run.name} ({ds.meta.get('arena', '?')}) ===")
        reports[run.name] = evaluate(model, ds, horizon, device, probes=True)
        print_report(reports[run.name], model.cfg)
    return reports


def write_report(out: Path, info: dict, reports: dict, log: list) -> None:
    # "report" stays the first held-out run's, for readers written before there
    # were several (export_onnx); "reports" has one per run.
    first = next(iter(reports.values()))
    (out / "report.json").write_text(json.dumps({"info": info, "report": first, "reports": reports, "log": log},
                                                indent=1), encoding="utf-8")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("warning: no CUDA device, this will be slow")

    if args.refit_probes:
        model, info, reports, log, out = refit_probes(args, device)
        save_checkpoint(out / "model.pt", model, {"info": info})
        write_report(out, info, reports, log)
        print(f"saved {out}")
        return

    name = args.name or time.strftime("jepa_%Y%m%d-%H%M%S")
    train_runs, val_runs = split_runs(args.data, args.val_seed)
    print("train runs:")
    train = jepa_data.load(train_runs, memmap_path=ROOT / "Training" / "cache" / f"{name}_grid.npy")
    print(f"held out: {[r.name for r in val_runs]}; the first is scored during training")
    val = jepa_data.load(val_runs[:1], verbose=False)

    meta = train.meta
    cfg = ModelConfig(channels=meta["grid"]["channels"], self_fields=meta["self_fields"],
                      intents=meta["intents"], probes=list(jepa_data.PROBES),
                      grid_size=meta["grid"]["size"], latent_dim=args.latent)
    model = WorldModel(cfg).to(device)
    count = lambda mod: sum(p.numel() for p in mod.parameters())  # noqa: E731
    print(f"params: encoder {count(model.encoder):,}  predictor {count(model.predictor):,}  probes {count(model.probes):,}")

    out = ROOT / "Training" / "checkpoints" / name
    out.mkdir(parents=True, exist_ok=True)

    model.train()
    log = train_world_model(model, train, args, device, val)
    model.eval()
    train_probes(model, train, args, device)
    del val
    reports = evaluate_each(model, val_runs, args.horizon, device)

    info = {"args": vars(args), "train_runs": [r.name for r in train_runs],
            "val_runs": [r.name for r in val_runs], "train_steps": train.steps,
            "dataset_schema": meta["schema"], "horizon": args.horizon}
    save_checkpoint(out / "model.pt", model, {"info": info})
    write_report(out, info, reports, log)
    print(f"saved {out}")

    # The memory-mapped grid is only a training cache (8.6 GB for jepa_v3).
    # Windows refuses to delete a file that is still mapped, so drop the
    # mapping first, and leave the file if that is not enough.
    cache = getattr(train.grid, "filename", None)
    if cache:
        del train
        gc.collect()
        try:
            os.remove(cache)
            print(f"removed grid cache {cache}")
        except OSError as e:
            print(f"left grid cache {cache}: {e}")


if __name__ == "__main__":
    main()
