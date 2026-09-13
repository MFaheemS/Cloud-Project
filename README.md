# When Quantization Stops Paying: Thermal Limits on Sustained Edge LLM Inference

Measurement harness, raw telemetry, and analysis for a study of sustained
LLM inference on thermally-constrained consumer GPUs.

Group G12, Cloud Computing 100-Day Research Assignment. Approved direction:
Edge–Cloud Intelligence. Faheem (23i-0728, Primary Researcher) and
Irtaza Kazmi (23i-6001, Co-Researcher).

## Research question

How does quantization level affect the throughput a consumer GPU can sustain
under continuous LLM inference, and can steady-state performance be predicted
from a short measurement well enough to make correct edge–cloud placement
decisions?

Objectives:

1. Characterise peak and sustained throughput, energy per token, and heating
   and cooling behaviour across model sizes, quantization levels and two GPUs.
2. Explain why the advantage of aggressive quantization shrinks under heat, by
   separating reduced clock speed from heat itself.
3. Build and validate a predictor of sustained throughput from a short
   cold-start measurement.
4. Evaluate a placement policy driven by that predictor against policies that
   assume constant performance or react to temperature.

## Motivation

Edge–cloud placement policies decide where inference runs using a *service
rate* (tokens/sec) measured at profiling time. On thermally-constrained
consumer hardware that number decays under sustained load, so the decision is
made from a value that silently becomes wrong.

## Findings so far

RTX 3050 Laptop (4 GB), Qwen2.5-1.5B-Instruct, Ollama, cold start (<60 deg C),
15 min sustained load + 15 min idle, NVML telemetry at 20 Hz. Both models are
100 % GPU-resident.

| Quantity | Q4_K_M | Q8_0 |
|---|---|---|
| Peak throughput (0-60 s) | **115.8** +/- 1.40 tok/s | **95.4** +/- 0.39 tok/s |
| Steady throughput (>700 s) | **66.16** +/- 4.01 (n=90) | **66.77** +/- 4.05 (n=91) |
| Throughput loss | 42.9 % | 30.0 % |
| Temperature | 61.6 -> 88.8 deg C | 56.2 -> 88.4 deg C |
| `tau_heat` (temperature) | 25 s | 30 s |
| `tau_cool` (temperature) | 105 s | 101 s |
| `tau_cool / tau_heat` | 4.16 | 3.36 |
| `tau_heat` (throughput) | 239 s | 269 s |
| Minimum fleet size | N >= 5.2 | N >= 4.4 |

Three results follow.

**1. Quantization's throughput advantage is transient.**
At cold start Q4 is 21.4 % faster than Q8. At thermal equilibrium the two are
statistically indistinguishable (66.16 vs 66.77 tok/s, a 0.6 tok/s difference
against a standard deviation of ~4.0 across ~90 requests each). Under sustained
load the device converges to a thermally-limited throughput floor that is
independent of quantization level: it delivers what its power and thermal
envelope allow, regardless of the arithmetic cost per token.

The methodological consequence is that **short-run benchmarks systematically
overstate quantization speedup** — here by the entire effect, a 21 % advantage
that decays to zero within roughly ten minutes. Q4 also heats faster
(`tau_heat` 25 s vs 30 s) precisely because it does more work per unit time
early on: it buys its speed by spending thermal headroom faster.

**2. Temperature is an information-free control signal.**
Temperature equilibrates 8.9-9.5x faster than throughput degrades (25-30 s vs
239-269 s). The GPU sits at a flat ~88 deg C, well below NVML's 97 deg C
slowdown threshold, while throughput continues to collapse for another four
minutes. Any policy that controls on temperature is blind for the entire window
in which the performance is lost. `SwThermalSlowdown` was active in 96.7 % of
samples in the uncontrolled pilot run.

**3. The floor is GPU-bound, not a serving artefact.**
A control run with Qwen2.5-0.5B-Q4 (397 MB) peaks at 243.6 tok/s and sustains
179.9 tok/s, far above the ~66 tok/s floor shared by both 1.5B variants. The
floor therefore scales with the work done per token and is not a fixed
per-token cost in the serving stack. The 0.5B run also holds ~1181 MHz at steady
state against ~835 MHz for the 1.5B runs, so the throttle floor follows power
draw rather than a fixed clock.

A likely mechanism: Q4_K_M reads fewer bytes per token than Q8 but spends more
arithmetic unpacking block-quantized weights. Throttling removes arithmetic
capacity, so the compute-dependent advantage erodes first. This is a hypothesis
to be tested (Objective 2), not yet a result.

### Exploratory: a fleet-size heuristic (not a claim)
Balancing heat in against heat out for an N-device rotation gives

    N >= 1 + tau_cool / tau_heat

which gives 5.2 devices for Q4 and 4.4 for Q8. This is a back-of-envelope
heuristic, not a result. In a simple lumped thermal model heating and cooling
share one time constant, so the measured asymmetry most likely reflects active
control (fan ramp-up, power reduction under throttling) rather than a property
of the silicon, and the two constants were measured towards different
asymptotes. It cannot be validated with the hardware available and is kept here
only as a direction for discussion.

### FP16 — a different regime, and confounded

FP16 (3.7 GB) does not fit in 4 GB of VRAM and runs at a 23 %/77 % CPU/GPU
split. It behaves completely differently:

| Quantity | FP16 (partly CPU-bound) |
|---|---|
| Throughput | 25.3 -> 24.6 tok/s (**2.9 % loss**) |
| Temperature | 54.9 -> 81.5 deg C |
| `tau_heat` / `tau_cool` | 103 s / 110 s |
| `tau_cool / tau_heat` | **1.07** (symmetric) |
| Minimum fleet size | N >= 2.1 |

Throughput is nearly flat and the heat/cool asymmetry disappears. The likely
explanation is that once a quarter of the work moves to the CPU, the GPU is no
longer the bottleneck: it is driven well below its thermal envelope (81.5 vs
88.8 deg C), so throttling barely affects end-to-end throughput.

**This measurement is confounded and should not be reported as a quantization
effect.** Two variables changed at once — numerical precision *and* the
CPU/GPU split — and this experiment cannot separate them. Disentangling them
requires a card with enough VRAM to hold FP16 entirely (e.g. 8 GB), which is
pending.

Taken cautiously, it suggests the thermal bound applies specifically to
*GPU-bound* inference, and that the bound weakens when the accelerator is not
the constraint.

## Repository layout

    src/thermal_tc.py    cold-start-controlled heat/cool characterisation harness
    src/analyze_tc.py    time-constant fitting and summary statistics
    experiments/raw/     raw telemetry and per-request CSVs
    proposal/            Cutoff 1 proposal (PDF)
    paper/               LaTeX manuscript source for Overleaf
    CONTRIBUTIONS.md     contribution log

## Reproducing

Requires an NVIDIA GPU, [Ollama](https://ollama.com), and Python 3.9+.

    pip install -r requirements.txt
    ollama pull qwen2.5:1.5b-instruct-q4_K_M

    # cold-start wait, 900 s load, 900 s cooldown, tag "q4", cold threshold 60 C
    python src/thermal_tc.py qwen2.5:1.5b-instruct-q4_K_M 900 900 q4 60
    python src/analyze_tc.py q4

Outputs `tc_<tag>_tel.csv` (telemetry) and `tc_<tag>_reqs.csv` (per-request).

## Methodological notes

- **Sample with NVML, not `nvidia-smi`.** Subprocess spawn caps `nvidia-smi`
  polling at ~3.7 Hz, too coarse to integrate power. NVML via `nvidia-ml-py`
  sustains ~84 kHz; we log at 20 Hz.
- **Control the cold start.** An uncontrolled pilot began at 88 °C already
  throttling and measured only 22.8 % degradation — roughly half the true
  figure, because it never observed peak throughput.
- **Check the offload split.** `ollama ps` reports the CPU/GPU split; a model
  that does not fit in VRAM measures a different regime. Llama-3.2-3B-Q4 spills
  10 % to CPU on a 4 GB card; Qwen2.5-1.5B-Q4 is 100 % GPU-resident.
- `nvmlDeviceGetUtilizationRates` is unreliable on hybrid-graphics laptops;
  clock and power are stable. Enforced power limit is unavailable on this part.

## Limitations

All runs so far are single repetitions on one device (RTX 3050 Laptop, 4 GB)
at uncontrolled ambient temperature. Planned: five repetitions per
configuration, a second GPU (RTX 4070 Laptop, 8 GB, which also allows a clean
FP16 run), a 3B model, recorded ambient temperature, and clock-locked runs to
test the mechanism.

## Status

Work in progress — 100-day cloud computing research project.

## License

MIT
