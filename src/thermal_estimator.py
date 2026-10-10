#!/usr/bin/env python3
"""
Thermal-aware steady-state throughput estimator (G12 proposed method).

PROBLEM (identified from the reproduced baseline)
--------------------------------------------------
Husom et al. and short-run benchmarks generally characterise a configuration
from a brief measurement on a cool device. On a thermally constrained consumer
GPU the device throttles under continuous load, so that figure overstates what
the configuration actually sustains -- here by up to 43% -- and the size of the
error depends on the quantization level. Deployment decisions taken from those
numbers can therefore be wrong.

PROPOSED MODULE
---------------
Instead of reporting the mean of a short measurement window, fit the observed
decay and extrapolate to the steady state:

    tok/s(t) = a + b * exp(-t / tau)        predicted steady state = a

For a fixed tau the model is linear in (a, b), so we grid-search tau and solve
the 2-parameter least squares in closed form. No SciPy dependency.

ABLATION
--------
    baseline      mean of the first W seconds          (what the paper reports)
    +A            longer measurement, last-60s mean    (measure for LONG_S)
    +B            predictor on the first W seconds     (cheap extrapolation)
    +A+B          predictor on the first LONG_S seconds

Ground truth is the mean throughput after STEADY_FROM seconds of continuous
load. Error is absolute percentage error against that value.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

SHORT_W = 60.0      # short measurement window (s) - the baseline's regime
LONG_S = 300.0      # "measure for longer" component A (s)
STEADY_FROM = 700.0  # ground-truth steady state starts here (s)

CONFIGS = {
    "s05": "qwen2.5-0.5b-q4_K_M",
    "q4": "qwen2.5-1.5b-q4_K_M",
    "q8": "qwen2.5-1.5b-q8_0",
    "fp16": "qwen2.5-1.5b-fp16 (CPU-offloaded)",
}


def load(tag: str) -> list[tuple[float, float]]:
    """Return [(elapsed_s, tok_s)] for one sustained-load run."""
    with open(RAW / f"tc_{tag}_reqs.csv", encoding="utf8") as f:
        return [(float(r["elapsed"]), float(r["tok_s"])) for r in csv.DictReader(f)]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def fit_steady_state(series, tau_grid=None):
    """Fit y = a + b*exp(-t/tau); return predicted steady state a.

    For fixed tau this is ordinary least squares in (a, b), solved in closed
    form. We scan tau and keep the fit with the lowest squared error.
    """
    if len(series) < 4:
        return float("nan")
    if tau_grid is None:
        tau_grid = [20.0 * k for k in range(1, 31)]  # 20..600 s

    t0 = series[0][0]
    best = (float("inf"), float("nan"))
    for tau in tau_grid:
        xs = [math.exp(-(t - t0) / tau) for t, _ in series]
        ys = [y for _, y in series]
        n = len(xs)
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        den = n * sxx - sx * sx
        if abs(den) < 1e-12:
            continue
        b = (n * sxy - sx * sy) / den
        a = (sy - b * sx) / n
        sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        if sse < best[0]:
            best = (sse, a)
    return best[1]


def ape(pred, truth):
    return abs(pred - truth) / truth * 100.0 if truth else float("nan")


def main():
    rows = []
    for tag, name in CONFIGS.items():
        s = load(tag)
        truth = mean(y for t, y in s if t >= STEADY_FROM)
        if math.isnan(truth):
            print(f"[skip] {tag}: no steady-state samples")
            continue

        short = [(t, y) for t, y in s if t <= SHORT_W]
        long_ = [(t, y) for t, y in s if t <= LONG_S]

        est = {
            "baseline": (mean(y for _, y in short), SHORT_W),
            "+A": (mean(y for t, y in long_ if t >= LONG_S - 60), LONG_S),
            "+B": (fit_steady_state(short), SHORT_W),
            "+A+B": (fit_steady_state(long_), LONG_S),
        }
        for method, (pred, cost) in est.items():
            rows.append({
                "config": name, "tag": tag, "method": method,
                "predicted_tok_s": round(pred, 1),
                "actual_tok_s": round(truth, 1),
                "abs_pct_error": round(ape(pred, truth), 1),
                "measurement_cost_s": int(cost),
            })

    with open(OUT / "ablation.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary over the GPU-resident configs (fp16 is CPU-offloaded) ----
    print(f"{'method':<10}{'MAPE %':>9}{'max err %':>11}{'cost s':>9}")
    summary = []
    for method in ("baseline", "+A", "+B", "+A+B"):
        errs = [r["abs_pct_error"] for r in rows
                if r["method"] == method and r["tag"] != "fp16"]
        cost = next(r["measurement_cost_s"] for r in rows if r["method"] == method)
        print(f"{method:<10}{mean(errs):>9.1f}{max(errs):>11.1f}{cost:>9}")
        summary.append({"method": method, "mape_pct": round(mean(errs), 1),
                        "max_err_pct": round(max(errs), 1),
                        "measurement_cost_s": cost})

    with open(OUT / "ablation_summary.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\nper-configuration detail")
    print(f"{'config':<34}{'actual':>8}" + "".join(f"{m:>10}" for m in ("baseline", "+A", "+B", "+A+B")))
    for tag, name in CONFIGS.items():
        rs = {r["method"]: r for r in rows if r["tag"] == tag}
        if not rs:
            continue
        line = f"{name:<34}{rs['baseline']['actual_tok_s']:>8.1f}"
        for m in ("baseline", "+A", "+B", "+A+B"):
            line += f"{rs[m]['predicted_tok_s']:>10.1f}"
        print(line)

    print(f"\nWrote {OUT/'ablation.csv'} and {OUT/'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
