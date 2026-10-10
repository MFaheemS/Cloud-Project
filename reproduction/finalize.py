#!/usr/bin/env python3
"""
Finalise the G12 reproduction of Husom et al. (ACM TIOT 2025).

Evaluates every completed response file, re-runs only the cells whose
generation is incomplete, and writes one comparison table against the
accuracy figures in the paper's Table 8.
"""
import csv, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "results"
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
EXPECTED = 1000  # 200 prompts x --repeat 5

DATASETS = {
    "commonsenseqa": ("CommonsenseQA/commonsenseqa_200_samples.jsonl",
                      "CommonSenseQA", "evaluate_commonsenseQA.py"),
    "truthfulqa": ("TruthfulQA/truthfulQA_MC_200_samples.jsonl",
                   "TruthfulQA", "evaluate_TruthfulQA.py"),
}
MODELS = ["qwen2.5:0.5b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q4_K_M",
          "qwen2.5:1.5b-instruct-q8_0", "qwen2.5:1.5b-instruct-fp16"]

# Accuracy reported in Table 8 (p.17). None = not reported: the 1.5B model was
# only run at q3/q4 levels, since q8_0 (1.6 GB) and fp16 (3.1 GB) do not fit
# alongside the OS on the paper's 4 GB Raspberry Pi 4.
PAPER = {
    ("qwen2.5:0.5b-instruct-q4_K_M", "commonsenseqa"): 0.32,
    ("qwen2.5:0.5b-instruct-q4_K_M", "truthfulqa"): 0.26,
    ("qwen2.5:1.5b-instruct-q4_K_M", "commonsenseqa"): 0.62,
    ("qwen2.5:1.5b-instruct-q4_K_M", "truthfulqa"): 0.24,
    ("qwen2.5:1.5b-instruct-q8_0", "commonsenseqa"): None,
    ("qwen2.5:1.5b-instruct-q8_0", "truthfulqa"): None,
    ("qwen2.5:1.5b-instruct-fp16", "commonsenseqa"): None,
    ("qwen2.5:1.5b-instruct-fp16", "truthfulqa"): None,
}

ACC = re.compile(r"Accuracy:\s*([\d.]+)%")
COR = re.compile(r"Correct:\s*(\d+)")


def path_for(ds, model):
    return OUT / f"{ds}_{model.replace(':', '_')}.jsonl"


def nlines(p):
    if not p.exists():
        return 0
    with open(p, encoding="utf8", errors="replace") as f:
        return sum(1 for _ in f)


def generate(ds, model):
    infile, dtype, _ = DATASETS[ds]
    print(f"  [gen] {model} / {ds}", flush=True)
    t0 = time.time()
    r = subprocess.run(
        [PY, "src/inference.py", "--input", infile, "--output", str(path_for(ds, model)),
         "--instruction", "--dataset_type", dtype, "--model", model,
         "--repeat", "5", "--timeout", "60"],
        cwd=DATA, capture_output=True, text=True, errors="replace", env=ENV)
    print(f"  [gen] rc={r.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
    return r.returncode == 0


def evaluate(ds, model):
    _, _, ev = DATASETS[ds]
    r = subprocess.run([PY, f"src/{ev}", str(path_for(ds, model))],
                       cwd=DATA, capture_output=True, text=True,
                       errors="replace", env=ENV)
    m = ACC.search(r.stdout or "")
    if not m:
        print(f"  [eval-FAIL] {model}/{ds}: {(r.stdout or '')[-200:]}{(r.stderr or '')[-200:]}", flush=True)
        return None, None
    c = COR.search(r.stdout or "")
    return float(m.group(1)) / 100.0, int(c.group(1)) if c else None


def main():
    rows = []
    for ds in DATASETS:
        for model in MODELS:
            p = path_for(ds, model)
            n = nlines(p)
            if n < EXPECTED:
                print(f"[incomplete] {model}/{ds}: {n}/{EXPECTED} -> regenerating", flush=True)
                if not generate(ds, model):
                    print(f"[skip] {model}/{ds} generation failed", flush=True)
                    continue
                n = nlines(p)
            acc, correct = evaluate(ds, model)
            if acc is None:
                continue
            paper = PAPER.get((model, ds))
            rows.append({
                "model": model, "dataset": ds, "generations": n,
                "accuracy_ours": round(acc, 3),
                "accuracy_paper": "" if paper is None else paper,
                "difference": "" if paper is None else round(acc - paper, 3),
                "status": "new data point (not in Table 8)" if paper is None else "reproduction",
            })
            print(f"[ok] {model}/{ds} ours={acc:.3f} paper={paper}", flush=True)

    out = OUT / "comparison_table.csv"
    with open(out, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
