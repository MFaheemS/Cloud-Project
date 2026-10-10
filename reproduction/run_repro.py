#!/usr/bin/env python3
"""
G12 reproduction driver for Husom et al. (ACM TIOT 2025), "Sustainable LLM
Inference for Edge AI".

Runs the authors' own inference.py and evaluate_*.py over the models we can
host locally, following their exp.sh protocol (--repeat 5 on the 200-sample
files), and collects accuracy into one CSV for comparison against Table 8.

Deviation from the original: the paper runs on a Raspberry Pi 4 (CPU) with a
Joulescope hardware power meter. We run on an RTX 3050 Laptop (4 GB) with
Ollama. Accuracy is hardware-independent and therefore directly comparable;
energy and latency are not, and are not claimed here.
"""
import csv, os, re, subprocess, sys, time
from pathlib import Path

# inference.py prints each raw model response to stdout. On Windows the default
# stdout encoding is cp1252, so any non-Latin-1 character the model emits (the
# 0.5B models produce CJK text fairly often) raises UnicodeEncodeError and kills
# the whole run. Forcing UTF-8 I/O in the child avoids patching their script.
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# dataset -> (input file relative to DATA, --dataset_type, evaluator script)
DATASETS = {
    "commonsenseqa": ("CommonsenseQA/commonsenseqa_200_samples.jsonl",
                      "CommonSenseQA", "evaluate_commonsenseQA.py"),
    "truthfulqa":    ("TruthfulQA/truthfulQA_MC_200_samples.jsonl",
                      "TruthfulQA", "evaluate_TruthfulQA.py"),
}

MODELS = [
    "qwen2.5:0.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q8_0",
    "qwen2.5:1.5b-instruct-fp16",
]

REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ACC_RE = re.compile(r"Accuracy:\s*([\d.]+)%")
CORRECT_RE = re.compile(r"Correct:\s*(\d+)\s*/\s*(\d+)")

summary_path = OUT / "reproduction_accuracy.csv"
rows = []
if summary_path.exists():  # resume support
    rows = list(csv.DictReader(open(summary_path, newline="", encoding="utf8")))
done = {(r["model"], r["dataset"]) for r in rows}

for ds, (infile, dtype, evaluator) in DATASETS.items():
    for model in MODELS:
        if (model, ds) in done:
            print(f"[skip] {model} / {ds} already done", flush=True)
            continue
        tag = f"{ds}_{model.replace(':', '_')}"
        responses = OUT / f"{tag}.jsonl"
        print(f"\n[run] {model} / {ds} (repeat={REPEAT})", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [PY, "src/inference.py", "--input", infile, "--output", str(responses),
             "--instruction", "--dataset_type", dtype, "--model", model,
             "--repeat", str(REPEAT), "--timeout", "60"],
            cwd=DATA, capture_output=True, text=True, errors="replace",
            env=CHILD_ENV,
        )
        elapsed = time.time() - t0
        if r.returncode != 0:
            print(f"[FAIL] inference: {r.stderr[-400:]}", flush=True)
            continue

        e = subprocess.run([PY, f"src/{evaluator}", str(responses)],
                           cwd=DATA, capture_output=True, text=True, errors="replace",
                           env=CHILD_ENV)
        m, c = ACC_RE.search(e.stdout or ""), CORRECT_RE.search(e.stdout or "")
        if not m:
            print(f"[FAIL] evaluate: {(e.stdout or '')[-300:]} {(e.stderr or '')[-300:]}", flush=True)
            continue

        row = {
            "model": model, "dataset": ds, "repeat": REPEAT,
            "accuracy": float(m.group(1)) / 100.0,
            "correct": c.group(1) if c else "", "total": c.group(2) if c else "",
            "wall_seconds": round(elapsed, 1),
        }
        rows.append(row)
        print(f"[ok] accuracy={row['accuracy']:.3f} ({row['correct']}/{row['total']}) "
              f"in {elapsed/60:.1f} min", flush=True)

        with open(summary_path, "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

print(f"\nDONE -> {summary_path}", flush=True)
