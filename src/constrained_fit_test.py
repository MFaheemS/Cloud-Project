#!/usr/bin/env python3
"""
Test a constrained variant of the thermal-aware estimator (G12).

The unconstrained least-squares fit of y(t) = a + b*exp(-t/tau) is ill-posed on
short windows: with only partial decay visible it can return a negative or
absurd asymptote (observed: -61 tok/s at a 120 s window). Throughput is
physically bounded, so we impose two constraints during the tau grid search and
keep the best feasible fit:

    0 < a <= min(observed y)      the steady state cannot exceed the lowest
                                  throughput already seen under load
    b >= 0                        the curve decays, it does not grow

This script quantifies what those constraints buy across window lengths.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw"
OUT = ROOT / "results"
STEADY_FROM = 700.0
CONFIGS = {"s05": "0.5b-q4", "q4": "1.5b-q4", "q8": "1.5b-q8"}


def load(tag):
    with open(RAW / f"tc_{tag}_reqs.csv", encoding="utf8") as f:
        return [(float(r["elapsed"]), float(r["tok_s"])) for r in csv.DictReader(f)]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def fit(series, constrained):
    if len(series) < 5:
        return float("nan")
    t0 = series[0][0]
    ts = [t - t0 for t, _ in series]
    ys = [y for _, y in series]
    lo = min(ys)
    n = len(ys)
    best = (float("inf"), float("nan"))
    for k in range(1, 31):
        tau = 20.0 * k
        xs = [math.exp(-t / tau) for t in ts]
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        den = n * sxx - sx * sx
        if abs(den) < 1e-12:
            continue
        b = (n * sxy - sx * sy) / den
        a = (sy - b * sx) / n
        if constrained:
            if b < 0:                      # must decay, not grow
                continue
            if not (0 < a <= lo):          # clamp to the feasible band
                a = min(max(a, 1e-6), lo)
                b = (sy - n * a) / sx if abs(sx) > 1e-12 else b
                if b < 0:
                    continue
        sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        if sse < best[0]:
            best = (sse, a)
    if best[1] != best[1]:
        # No feasible fit: too little decay in the window for any tau to give a
        # non-increasing curve. Fall back to the most conservative feasible
        # estimate, the lowest throughput already observed. Returning NaN would
        # leave the caller with no answer at all.
        return lo
    return best[1]


def main():
    data = {t: load(t) for t in CONFIGS}
    truth = {t: mean(y for tt, y in s if tt >= STEADY_FROM) for t, s in data.items()}
    windows = [30, 45, 60, 90, 120, 180, 240, 300, 450, 600]

    rows = []
    print(f"{'window s':>9}{'uncon MAPE':>12}{'uncon worst':>13}"
          f"{'CON MAPE':>11}{'CON worst':>11}")
    for W in windows:
        u, c = [], []
        for tag in CONFIGS:
            sub = [(t, y) for t, y in data[tag] if t <= W]
            u.append(abs(fit(sub, False) - truth[tag]) / truth[tag] * 100)
            c.append(abs(fit(sub, True) - truth[tag]) / truth[tag] * 100)
        rows.append({"window_s": W,
                     "unconstrained_mape": round(mean(u), 1),
                     "unconstrained_worst": round(max(u), 1),
                     "constrained_mape": round(mean(c), 1),
                     "constrained_worst": round(max(c), 1)})
        print(f"{W:>9}{mean(u):>12.1f}{max(u):>13.1f}{mean(c):>11.1f}{max(c):>11.1f}")

    um = mean(r["unconstrained_mape"] for r in rows)
    cm = mean(r["constrained_mape"] for r in rows)
    uw = max(r["unconstrained_worst"] for r in rows)
    cw = max(r["constrained_worst"] for r in rows)
    print(f"\naveraged over all windows   unconstrained {um:.1f}%   constrained {cm:.1f}%")
    print(f"worst case across windows   unconstrained {uw:.1f}%   constrained {cw:.1f}%")

    with open(OUT / "constrained_fit_test.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT/'constrained_fit_test.csv'}")


if __name__ == "__main__":
    main()
