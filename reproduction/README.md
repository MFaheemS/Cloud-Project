# Baseline Reproduction — Husom et al. (ACM TIOT, 2025)

Reproduction of *Sustainable LLM Inference for Edge AI: Evaluating Quantized
LLMs for Energy Efficiency, Output Accuracy, and Inference Latency*
(ACM Transactions on Internet of Things 6(4), doi:10.1145/3767742,
preprint arXiv:2504.03360) on a 4 GB consumer laptop GPU.

Full write-up: `Reproduction_Report_G12.pdf`.

## Result

Accuracy is hardware-independent, so these cells are directly comparable with
the paper's Table 8 (p. 17). Each cell is 200 prompts x `--repeat 5` =
1,000 generations, scored with the authors' own evaluation scripts.

| Model | Dataset | Paper | Reproduced | Diff | p |
|---|---|---|---|---|---|
| qwen2.5-0.5b-q4_K_M | CommonsenseQA | 0.32 | 0.291 | -0.029 | 0.420 |
| qwen2.5-1.5b-q4_K_M | CommonsenseQA | 0.62 | 0.593 | -0.027 | 0.474 |
| qwen2.5-0.5b-q4_K_M | TruthfulQA | 0.26 | 0.275 | +0.015 | 0.660 |
| qwen2.5-1.5b-q4_K_M | TruthfulQA | 0.24 | 0.249 | +0.009 | 0.786 |

All four agree with the published values; no difference is statistically
significant (two-proportion z-test).

Two further cells are **new data, not reproductions** — the paper's Raspberry
Pi 4 could not hold the 1.5B model above q4:

| Model | CommonsenseQA | TruthfulQA |
|---|---|---|
| qwen2.5-1.5b-q8_0 | 0.698 | 0.239 |
| qwen2.5-1.5b-fp16 | 0.705 | 0.245 |

## Layout

    run_repro.py       protocol driver (models x datasets, --repeat 5)
    finalize.py        completeness check, evaluation, comparison table
    results/           comparison_table.csv, reproduction_accuracy.csv
    patches/           changes made to upstream code (see below)

## Reproducing

Requires an NVIDIA GPU, [Ollama](https://ollama.com), and Python 3.9+.

    git clone https://github.com/ejhusom/MELODI.git
    git clone https://github.com/ejhusom/neurips-edge-llm-challenge-sampled.git data
    pip install ollama jsonlines pandas scikit-learn rouge_score python-Levenshtein nltk
    python run_repro.py 5        # generate
    python finalize.py           # evaluate + build comparison table

Commits used: MELODI `d55dae9`, data `c11fe13`.

## Changes required to the upstream code

`patches/husom_inference_py_windows.patch`

1. `signal.SIGALRM` / `signal.alarm` are Unix-only and raise on Windows. Now
   guarded with `hasattr(signal, "SIGALRM")`; Linux behaviour is unchanged.

Handled without editing upstream:

2. Undocumented dependencies — the data repository ships no requirements file.
   `scikit-learn`, `rouge_score`, `python-Levenshtein` and `nltk` are each
   needed by the evaluators and surface one at a time.
3. `inference.py` prints raw model output to stdout, which raises
   `UnicodeEncodeError` under the Windows cp1252 default when a model emits
   non-Latin-1 text (the 0.5B models emit CJK often enough that one run died at
   501/1000). `run_repro.py` sets `PYTHONIOENCODING=utf-8` in the child process.
4. **Silent data contamination.** Output is opened with
   `jsonlines.open(..., mode='a')`, so re-running a configuration appends
   instead of overwriting. After the crash in (3) the retry produced a
   1,501-line file mixing a partial and a complete run, and the evaluator
   scored it without complaint. `finalize.py` asserts the expected generation
   count before evaluating.

Not reproduced: energy and latency. The paper uses a Raspberry Pi 4 with an
external hardware power meter; this setup is a laptop GPU with NVML sampling.
Those numbers are not comparable and are not claimed.

## Rejected candidate baseline

`patches/bench360_portability.patch`, `patches/launch_engine_ollama.bat`,
`patches/config_g12_4gb_single.yaml`

Bench360 (arXiv:2511.16682) was set up and run before being rejected: its
experiment configs assume a 24 GB VRAM budget and 7B-32B models, so no
published cell is reproducible on available hardware, and it is an unrefereed
preprint. The patches add a GGUF/Ollama backend — none of its four backends
(vLLM, TGI, SGLang, LMDeploy) run on a 4 GB consumer GPU despite the paper
targeting consumer hardware — plus Windows portability fixes. Retained as
evidence for the reproducibility verification step.
