#!/usr/bin/env python3
"""
Headroom analysis for the thermal-aware estimator (G12).

Asks three questions before committing further effort to this direction:

  Q1  How much of the remaining error is reducible?  Compare the deployed
      estimator against an ORACLE that fits the same functional form to the
      FULL run. The oracle is the best this model family can do, so the gap
      between estimator and oracle is recoverable headroom, while the oracle's
      own error is an irreducible floor for this form.

  Q2  Is a better functional form available?  Compare single-exponential
      against double-exponential and power-law decay on the full runs.

  Q3  How does error trade against measurement cost?  Sweep the window length
      and find the knee.

No SciPy: every form is linear in its amplitude parameters once the shape
parameters are fixed, so we grid-search the shape and solve least squares in
closed form.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

STEADY_FROM = 700.0
GPU_RESIDENT = ["s05", "q4", "q8"]          # fp16 is CPU-offloaded
NAMES = {"s05": "0.5b-q4", "q4": "1.5b-q4", "q8": "1.5b-q8", "fp16": "1.5b-fp16"}


def load(tag):
    with open(RAW / f"tc_{tag}_reqs.csv", encoding="utf8") as f:
        return [(float(r["elapsed"]), float(r["tok_s"])) for r in csv.DictReader(f)]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _lstsq(cols, ys):
    """Solve ys ~ sum(beta_k * cols[k]) by normal equations (Gaussian elim)."""
    k = len(cols)
    A = [[sum(cols[i][n] * cols[j][n] for n in range(len(ys))) for j in range(k)]
         + [sum(cols[i][n] * ys[n] for n in range(len(ys)))] for i in range(k)]
    for i in range(k):                       # elimination with partial pivot
        p = max(range(i, k), key=lambda r: abs(A[r][i]))
        if abs(A[p][i]) < 1e-12:
            return None
        A[i], A[p] = A[p], A[i]
        for r in range(i + 1, k):
            f = A[r][i] / A[i][i]
            for c in range(i, k + 1):
                A[r][c] -= f * A[i][c]
    beta = [0.0] * k
    for i in reversed(range(k)):
        beta[i] = (A[i][k] - sum(A[i][j] * beta[j] for j in range(i + 1, k))) / A[i][i]
    return beta


def fit(series, form="exp"):
    """Return (asymptote, sse) for the chosen decay form."""
    if len(series) < 5:
        return float("nan"), float("inf")
    t0 = series[0][0]
    ts = [t - t0 for t, _ in series]
    ys = [y for _, y in series]
    ones = [1.0] * len(ys)
    best = (float("inf"), float("nan"))

    if form == "exp":
        shapes = [(20.0 * k,) for k in range(1, 31)]
    elif form == "exp2":
        taus = [20.0, 45.0, 90.0, 150.0, 240.0, 360.0, 500.0, 650.0]
        shapes = [(a, b) for i, a in enumerate(taus) for b in taus[i + 1:]]
    elif form == "power":
        shapes = [(0.1 * k,) for k in range(1, 21)]
    else:
        raise ValueError(form)

    for sh in shapes:
        if form == "exp":
            cols = [ones, [math.exp(-t / sh[0]) for t in ts]]
        elif form == "exp2":
            cols = [ones, [math.exp(-t / sh[0]) for t in ts],
                    [math.exp(-t / sh[1]) for t in ts]]
        else:
            cols = [ones, [(1.0 + t) ** (-sh[0]) for t in ts]]
        beta = _lstsq(cols, ys)
        if beta is None:
            continue
        pred = [sum(b * c[n] for b, c in zip(beta, cols)) for n in range(len(ys))]
        sse = sum((y - p) ** 2 for y, p in zip(ys, pred))
        if sse < best[0]:
            best = (sse, beta[0])
    return best[1], best[0]


def ape(pred, truth):
    return abs(pred - truth) / truth * 100.0


def main():
    data = {t: load(t) for t in NAMES}
    truth = {t: mean(y for tt, y in s if tt >= STEADY_FROM) for t, s in data.items()}

    # ---------- Q3: error vs measurement cost ----------
    windows = [30, 45, 60, 90, 120, 180, 240, 300, 450, 600]
    sweep_rows = []
    print("Q3  estimator error vs measurement window (GPU-resident configs)")
    print(f"{'window s':>9}{'MAPE %':>9}{'worst %':>9}")
    for W in windows:
        errs = []
        for tag in GPU_RESIDENT:
            sub = [(t, y) for t, y in data[tag] if t <= W]
            pred, _ = fit(sub, "exp")
            errs.append(ape(pred, truth[tag]) if pred == pred else float("nan"))
        sweep_rows.append({"window_s": W, "mape_pct": round(mean(errs), 1),
                           "worst_pct": round(max(errs), 1)})
        print(f"{W:>9}{mean(errs):>9.1f}{max(errs):>9.1f}")

    # ---------- Q1: oracle headroom ----------
    print("\nQ1  deployed estimator vs oracle (same form fitted to the full run)")
    print(f"{'config':<12}{'actual':>8}{'W=60':>9}{'W=300':>9}{'oracle':>9}{'floor %':>9}")
    head_rows = []
    for tag in GPU_RESIDENT:
        p60, _ = fit([(t, y) for t, y in data[tag] if t <= 60], "exp")
        p300, _ = fit([(t, y) for t, y in data[tag] if t <= 300], "exp")
        porc, _ = fit(data[tag], "exp")
        head_rows.append({
            "config": NAMES[tag], "actual": round(truth[tag], 1),
            "w60_err_pct": round(ape(p60, truth[tag]), 1),
            "w300_err_pct": round(ape(p300, truth[tag]), 1),
            "oracle_err_pct": round(ape(porc, truth[tag]), 1),
        })
        print(f"{NAMES[tag]:<12}{truth[tag]:>8.1f}{p60:>9.1f}{p300:>9.1f}{porc:>9.1f}"
              f"{ape(porc, truth[tag]):>9.1f}")
    orc = mean(r["oracle_err_pct"] for r in head_rows)
    w300 = mean(r["w300_err_pct"] for r in head_rows)
    print(f"\n    mean oracle error (irreducible for single-exp): {orc:.1f}%")
    print(f"    mean deployed error at W=300:                   {w300:.1f}%")
    print(f"    recoverable headroom:                           {w300-orc:.1f} points")

    # ---------- Q2: is a better functional form available? ----------
    print("\nQ2  functional form, fitted to full runs (lower SSE = better fit)")
    print(f"{'config':<12}{'exp SSE':>12}{'exp2 SSE':>12}{'power SSE':>12}"
          f"{'exp err%':>10}{'exp2 err%':>11}{'power err%':>12}")
    form_rows = []
    for tag in GPU_RESIDENT:
        out = {}
        for form in ("exp", "exp2", "power"):
            a, sse = fit(data[tag], form)
            out[form] = (a, sse)
        form_rows.append({
            "config": NAMES[tag],
            "exp_sse": round(out["exp"][1], 1), "exp2_sse": round(out["exp2"][1], 1),
            "power_sse": round(out["power"][1], 1),
            "exp_err_pct": round(ape(out["exp"][0], truth[tag]), 1),
            "exp2_err_pct": round(ape(out["exp2"][0], truth[tag]), 1),
            "power_err_pct": round(ape(out["power"][0], truth[tag]), 1),
        })
        r = form_rows[-1]
        print(f"{NAMES[tag]:<12}{r['exp_sse']:>12.1f}{r['exp2_sse']:>12.1f}"
              f"{r['power_sse']:>12.1f}{r['exp_err_pct']:>10.1f}"
              f"{r['exp2_err_pct']:>11.1f}{r['power_err_pct']:>12.1f}")

    for name, rows in (("headroom_window_sweep", sweep_rows),
                       ("headroom_oracle", head_rows),
                       ("headroom_forms", form_rows)):
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote headroom_*.csv to {OUT}")


if __name__ == "__main__":
    main()
