# Thermally Mandated Offloading: Heat–Recovery Asymmetry as a Hard Bound on Edge LLM Serving

Measurement harness, raw telemetry, and analysis for a study of sustained
LLM inference on thermally-constrained consumer GPUs.

## Research question

Given measured heat-up and cool-down time constants of consumer GPUs under
sustained LLM inference, what is the minimum edge fleet size at which local
rotation can sustain throughput — and below that bound, what offload policy
follows from those constants?

## Motivation

Edge–cloud placement policies decide where inference runs using a *service
rate* (tokens/sec) measured at profiling time. On thermally-constrained
consumer hardware that number decays under sustained load, so the decision is
made from a value that silently becomes wrong.

## Findings so far

RTX 3050 Laptop (4 GB), Qwen2.5-1.5B-Instruct-Q4_K_M, Ollama, cold start
(<60 °C), 15 min sustained load + 15 min idle, NVML telemetry at 20 Hz.

| Quantity | Value |
|---|---|
| Throughput | 116.8 -> 66.4 tok/s (**43.1 % loss**) |
| Temperature | 61.6 -> 88.8 °C |
| `tau_heat` (temperature) | 25 s |
| `tau_cool` (temperature) | 105 s |
| `tau_cool / tau_heat` | **4.16** |
| `tau_heat` (throughput) | 239 s |
| Minimum fleet size | **N >= 5.2 devices** |

Two results follow.

**1. Temperature is an information-free control signal.**
Temperature equilibrates 9.5x faster than throughput degrades (25 s vs 239 s).
The GPU sits at a flat ~88 °C, well below NVML's 97 °C slowdown threshold,
while throughput continues to collapse for another four minutes. Any policy
that controls on temperature is blind for the entire window in which the
performance is lost. `SwThermalSlowdown` was active in 96.7 % of samples in the
uncontrolled pilot run.

**2. A fleet-size bound on thermal rotation.**
Balancing heat in against heat out for an N-device rotation gives

    N >= 1 + tau_cool / tau_heat  ~= 5.2 devices

Below this, rotating work between edge devices cannot reach thermal
equilibrium and the fleet throttles regardless of scheduling. Cloud offload is
therefore a *thermal necessity*, not merely a cost or latency optimisation.

## Repository layout

    src/thermal_tc.py    cold-start-controlled heat/cool characterisation harness
    src/analyze_tc.py    time-constant fitting and fleet-bound derivation
    experiments/raw/     raw telemetry and per-request CSVs
    results/             derived tables and figures
    docs/                assignment guide and notes

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

Results are from a single device, single quantization level, single run, at
uncontrolled ambient temperature. The duty-cycle bound assumes a first-order
lumped thermal model and has not yet been validated against a real multi-device
alternation experiment. The quantization sweep (Q8, FP16) and a second GPU
(RTX 4070, 8 GB) are in progress.

## Status

Work in progress — 100-day cloud computing research project.

## License

MIT
