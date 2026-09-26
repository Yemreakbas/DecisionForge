"""Exports a trained world model to ONNX for Unity's Inference Engine.

    Training\\.venv\\Scripts\\python Training\\export_onnx.py [checkpoint folder] [--out DIR]

Four graphs, all taking raw observations exactly as TacticalObservation fills
them -- log1p on the accumulating channels and the echo-field mask are inside
the encoder, so Unity does no preprocessing:

    encoder.onnx    grid (B,7,32,32), self (B,24)      -> latent (B,D)
    predictor.onnx  latent (B,D), intent (B,10) one-hot -> next_latent (B,D)
    probes.onnx     latent (B,D)                         -> probes (B,P)
    imagine.onnx    latent (1,D), intents (N,H,10)       -> probes (N,H,P)

``imagine`` is the planner's whole per-tick job in one call: N candidate intent
sequences rolled H ticks forward from one latent, every imagined step read by
the probes (including the display-only position probes that draw the ghost
trails). H is fixed at export time; N is free.

``model.json`` beside them is the contract: channel, self-field, intent and probe
names in model order. JevBrain should refuse to load a model whose names differ
from its enums rather than feed it a silently reordered observation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jepa_data  # noqa: E402
from jevdata import Run, find_run_by_name  # noqa: E402
from jevmodel import ACCUMULATING_CHANNELS, ECHO_FIELDS, load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OPSET = 15   # Inference Engine imports opsets 7..15 without surprises


class Imagine(nn.Module):
    def __init__(self, model, horizon: int):
        super().__init__()
        self.predictor, self.probes, self.horizon = model.predictor, model.probes, horizon

    def forward(self, latent, intents):
        z = latent.expand(intents.shape[0], -1)
        out = []
        for k in range(self.horizon):
            z = self.predictor(z, intents[:, k])
            out.append(self.probes(z))
        return torch.stack(out, dim=1)


def export(module, args, path, inputs, outputs, dynamic):
    torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs,
                      dynamic_axes=dynamic, opset_version=OPSET, dynamo=False)


def check(path, feeds: dict, expected: np.ndarray, label: str, runs: int = 200) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)[0]
    err = float(np.abs(got - expected).max())
    for _ in range(10):
        sess.run(None, feeds)
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, feeds)
    ms = (time.perf_counter() - t0) * 1000 / runs
    status = "ok" if err < 1e-3 else "MISMATCH"
    print(f"  {label:34s} max |onnx - torch| {err:.2e}  {status}   onnxruntime cpu {ms:.3f} ms")
    if err >= 1e-3:
        raise SystemExit(f"{path.name}: ONNX output differs from torch")
    return {"max_abs_error": err, "ort_cpu_ms": ms}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="?", help="checkpoint folder (default: newest)")
    p.add_argument("--out", help="output folder (default: Training/exports/<checkpoint>)")
    p.add_argument("--horizon", type=int, help="imagine.onnx rollout length (default: training horizon)")
    p.add_argument("--candidates", type=int, default=10, help="N used for the check and the timing")
    args = p.parse_args()

    ckpts = sorted((ROOT / "Training" / "checkpoints").glob("*/model.pt"), key=lambda q: q.stat().st_mtime)
    path = Path(args.checkpoint) / "model.pt" if args.checkpoint else (ckpts[-1] if ckpts else None)
    if path is None or not path.exists():
        raise SystemExit("no checkpoint found")
    model, ckpt = load_checkpoint(path, "cpu")
    model.float().eval()
    cfg, info = model.cfg, ckpt["info"]
    horizon = args.horizon or info["horizon"]
    out = Path(args.out) if args.out else ROOT / "Training" / "exports" / path.parent.name
    out.mkdir(parents=True, exist_ok=True)

    # Real observations from the held-out run, for the equivalence check.
    run = Run.open(find_run_by_name(ROOT / "Datasets", info["val_runs"][0]))
    shard = next(run.shards())
    pick = np.linspace(0, shard.steps - 1, 64).astype(int)
    grid = shard.grid[pick].astype(np.float32)
    selfv = shard.self_vec[pick].astype(np.float32)
    D, A, P = cfg.latent_dim, len(cfg.intents), len(cfg.probes)
    rng = np.random.default_rng(0)
    onehot = np.eye(A, dtype=np.float32)[rng.integers(0, A, 64)]
    seqs = np.eye(A, dtype=np.float32)[rng.integers(0, A, (args.candidates, horizon))]

    print(f"exporting {path.parent.name} -> {out}  (opset {OPSET}, imagine horizon {horizon})")
    with torch.no_grad():
        g, s = torch.from_numpy(grid), torch.from_numpy(selfv)
        z = model.encoder(g, s)
        z_np = z.numpy()
        imagine = Imagine(model, horizon).eval()

        export(model.encoder, (g, s), out / "encoder.onnx", ["grid", "self"], ["latent"],
               {"grid": {0: "batch"}, "self": {0: "batch"}, "latent": {0: "batch"}})
        export(model.predictor, (z, torch.from_numpy(onehot)), out / "predictor.onnx",
               ["latent", "intent"], ["next_latent"],
               {"latent": {0: "batch"}, "intent": {0: "batch"}, "next_latent": {0: "batch"}})
        export(model.probes, (z,), out / "probes.onnx", ["latent"], ["probes"],
               {"latent": {0: "batch"}, "probes": {0: "batch"}})
        export(imagine, (z[:1], torch.from_numpy(seqs)), out / "imagine.onnx", ["latent", "intents"],
               ["probes"], {"intents": {0: "candidates"}, "probes": {0: "candidates"}})

        checks = {
            "encoder": check(out / "encoder.onnx", {"grid": grid[:1], "self": selfv[:1]},
                             model.encoder(g[:1], s[:1]).numpy(), "encoder (batch 1)"),
            "predictor": check(out / "predictor.onnx", {"latent": z_np, "intent": onehot},
                               model.predictor(z, torch.from_numpy(onehot)).numpy(), "predictor (batch 64)"),
            "probes": check(out / "probes.onnx", {"latent": z_np}, model.probes(z).numpy(), "probes (batch 64)"),
            "imagine": check(out / "imagine.onnx", {"latent": z_np[:1], "intents": seqs},
                             imagine(z[:1], torch.from_numpy(seqs)).numpy(),
                             f"imagine ({args.candidates} candidates x {horizon} ticks)"),
        }

    report_path = path.parent / "report.json"
    evaluation = {}
    if report_path.exists():
        r = json.loads(report_path.read_text(encoding="utf-8"))["report"]
        evaluation = {"rollout_ratio_to_copy": {k: v["ratio_to_copy"] for k, v in r["rollout"].items()},
                      "intent_id_accuracy": r["intent_id"]["accuracy"],
                      "intent_id_balanced_accuracy": r["intent_id"]["balanced_accuracy"],
                      "windows": r["windows"]}

    contract = {
        "schema": "jev-model/1",
        "checkpoint": path.parent.name,
        "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_schema": info["dataset_schema"],
        "train_runs": info["train_runs"],
        "val_runs": info["val_runs"],
        "tick_seconds": 1.0 / run.meta["tick_hz"],
        "grid": {"channels": cfg.channels, "size": cfg.grid_size, "cell_size": run.meta["grid"]["cell_size"],
                 "frame": run.meta["grid"]["frame"], "log1p_channels": list(ACCUMULATING_CHANNELS)},
        "self_fields": cfg.self_fields,
        "masked_self_fields": list(ECHO_FIELDS),
        "intents": cfg.intents,
        "latent_dim": D,
        "probes": [{"name": n, "description": jepa_data.PROBES.get(n, ""),
                    "display_only": n in jepa_data.DISPLAY_ONLY_PROBES} for n in cfg.probes],
        "position_probes": {"x": "pos_x", "z": "pos_z", "world_metres": "(p - 0.5) * 2 * half_extent",
                            "half_extent": jepa_data.ARENA_HALF_EXTENT},
        "opset": OPSET,
        "graphs": {
            "encoder": {"file": "encoder.onnx", "inputs": {"grid": [-1, len(cfg.channels), cfg.grid_size, cfg.grid_size],
                                                           "self": [-1, len(cfg.self_fields)]},
                        "outputs": {"latent": [-1, D]}},
            "predictor": {"file": "predictor.onnx", "inputs": {"latent": [-1, D], "intent": [-1, A]},
                          "outputs": {"next_latent": [-1, D]}},
            "probes": {"file": "probes.onnx", "inputs": {"latent": [-1, D]}, "outputs": {"probes": [-1, P]}},
            "imagine": {"file": "imagine.onnx", "horizon": horizon,
                        "inputs": {"latent": [1, D], "intents": [-1, horizon, A]},
                        "outputs": {"probes": [-1, horizon, P]}},
        },
        "checks": checks,
        "evaluation": evaluation,
    }
    (out / "model.json").write_text(json.dumps(contract, indent=1), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.glob("*.onnx")) / 1e6
    print(f"wrote 4 graphs ({size:.1f} MB) and model.json to {out}")


if __name__ == "__main__":
    main()
