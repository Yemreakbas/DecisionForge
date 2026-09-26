"""Aggregates match CSVs (MatchMetrics.ToCsv) into one A/B result per brain.

    Training\\.venv\\Scripts\\python Training\\summarize_matches.py [CSV or glob ...] [--since EPOCH]

Default input: every ``*_eval_*.csv`` under Builds/Collector/MatchLogs. Rounds a
brain won are pooled over files; the win rate is over decided rounds, with a
Wilson 95% interval and a z-score against the 50% a fair rig gives two equal
brains (see the mirror-match calibration in CLAUDE.md). Each file is also split
by which team slot the brain played, so a slot bias would show.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMED = ("rounds_won", "kills", "deaths", "first_contact", "dumb_moments", "exposure_s", "center_s", "safety_vetoes")


def wilson(wins: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return float("nan"), float("nan")
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", help="CSV files or glob patterns")
    p.add_argument("--since", type=float, default=0.0, help="only files modified after this epoch time")
    args = p.parse_args()

    patterns = args.inputs or [str(ROOT / "Builds" / "Collector" / "MatchLogs" / "*_eval_*.csv")]
    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    files = [f for f in files if Path(f).stat().st_mtime > args.since]
    if not files:
        raise SystemExit("no match CSVs found")

    totals = defaultdict(lambda: defaultdict(float))
    by_slot = defaultdict(lambda: [0, 0])          # (brain, slot) -> [won, decided]
    draws = rounds = 0
    ms = defaultdict(list)
    peak = defaultdict(float)
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != 2:
            print(f"skipping {f}: expected two team rows")
            continue
        drawn = int(rows[0]["rounds_drawn"])
        decided = sum(int(r["rounds_won"]) for r in rows)
        draws += drawn
        rounds += decided + drawn
        mirror = rows[0]["brain"] == rows[1]["brain"]
        for r in rows:
            # In a mirror match both rows carry the same label; keep the sides
            # apart, since A against B *is* the calibration number.
            brain = f"{r['brain']} (team {'A' if r['team'] == '0' else 'B'})" if mirror else r["brain"]
            for key in SUMMED:
                if key in r and r[key] != "":
                    totals[brain][key] += float(r[key])
            totals[brain]["files"] += 1
            totals[brain]["decided"] += decided
            ms[brain].append(float(r["avg_decision_ms"]))
            peak[brain] = max(peak[brain], float(r["peak_decision_ms"]))
            slot = by_slot[(brain, "A" if r["team"] == "0" else "B")]
            slot[0] += int(r["rounds_won"])
            slot[1] += decided

    print(f"{len(files)} match file(s), {rounds} rounds, {draws} drawn")
    for brain, t in sorted(totals.items()):
        n, won = int(t["decided"]), int(t["rounds_won"])
        lo, hi = wilson(won, n)
        z = (won - n / 2) / math.sqrt(n / 4) if n else float("nan")
        per_round = lambda key: t[key] / max(rounds, 1)  # noqa: E731
        print(f"\n{brain}")
        print(f"  rounds won      {won}/{n} decided = {won / max(n, 1):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  z = {z:+.2f}")
        print(f"  K/D             {int(t['kills'])}/{int(t['deaths'])}   first contact {int(t['first_contact'])}")
        print(f"  per round       dumb moments {per_round('dumb_moments'):.2f}   exposure {per_round('exposure_s'):.1f} s"
              f"   centre {per_round('center_s'):.1f} s")
        print(f"  decision cost   mean of file averages {sum(ms[brain]) / len(ms[brain]):.4f} ms, peak {peak[brain]:.3f} ms")
        print(f"  safety vetoes   {int(t['safety_vetoes'])}")
        for slot in ("A", "B"):
            w, d = by_slot.get((brain, slot), (0, 0))
            if d:
                print(f"  as team {slot}       {w}/{d} = {w / d:.3f}")


if __name__ == "__main__":
    main()
