"""Builds the self-contained Kaggle notebook that trains the final model and writes submission.csv.

The notebook embeds the source of the modules it needs (taken from src/ at build time, so it never
drifts from the repository), runs the pipeline on the competition data and writes
/kaggle/working/submission.csv. It also writes kernel-metadata.json for `kaggle kernels push`.

Usage:
    python kaggle/build_kaggle_notebook.py --user <kaggle-username>
"""
import argparse
import json
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
COMPETITION = "fds-pokemon-battles-prediction-2026"
SLUG = "pokemon-fds-final"
TITLE = "Pokemon FDS Final"  # Kaggle derives the slug from the title: it must match SLUG
NOTEBOOK = f"{SLUG}.ipynb"

# module -> one-line description, in dependency order
MODULES = {
    "compute.py": "Hardware configuration: cores, CUDA, fork/spawn, fold-level parallelism.",
    "features.py": "Battle-state feature engineering, Gen 2 damage engine, label-free context (pokédex, "
                   "Hidden Power types, co-occurrence, species moves).",
    "train.py": "Feature cache (turn 30 + training snapshots at turns 20-28) and bag-of-words columns.",
    "simulate.py": "Monte Carlo simulator: plays every battle to the end 100 times from the last observed state.",
    "recursive.py": "Recursive model: LightGBM turn-to-turn transition model and roll-outs (RNN-like, no neural nets).",
    "stacking.py": "Base model zoo, out-of-fold predictions with snapshot augmentation, duplicate override.",
    "final_model.py": "Final 10-model stack with a logistic-regression meta-learner.",
}

INTRO = """# FDS Challenge: Pokémon Battle Prediction — final model

Classical machine learning only (no neural networks). Given the first 30 turns of a Gen 2 OU battle,
the notebook predicts whether player 1 wins.

**Pipeline**

1. **Features:** about 280 battle-state features from a Gen 2 damage engine, plus bag-of-words of
   Pokémon and moves.
2. **Monte Carlo features:** 100 simulated continuations of every battle.
3. **Recursive features:** turn-by-turn roll-outs of a gradient-boosted transition model.
4. **Snapshot augmentation:** training battles truncated at turns 24/26/28 are added inside the
   training folds, with weight 0.5.
5. **Stack of 10 diverse base models:** LightGBM, XGBoost and CatBoost with augmentation,
   ExtraTrees, Random Forest, LDA and two logistic regressions. A logistic-regression meta-learner
   combines their out-of-fold logits.

**Expected accuracy:** about 0.696 in nested cross-validation (0.699 with plain 5-fold OOF).

**Settings:**
- Accelerator: **None**. The CPU run takes about 30–40 minutes on Kaggle's 4 cores. With a GPU,
  XGBoost and CatBoost switch to CUDA and the predictions change slightly.
- Internet: not needed.

Source code and full documentation: https://github.com/MarcoValerioQuinti/pokemon-fds"""

SETUP = '''import glob
import os
import subprocess
import sys
import time

import pandas as pd


def find_data_dir():
    """Competition data: mounted under /kaggle/input on Kaggle, POKEMON_DATA or ./data elsewhere."""
    hits = sorted(glob.glob("/kaggle/input/**/train.jsonl", recursive=True))
    if hits:
        return os.path.dirname(hits[0])
    return os.environ.get("POKEMON_DATA", os.path.abspath("data"))


DATA_DIR = find_data_dir()
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
WORK_DIR = os.path.join(OUT_DIR, "outputs")  # feature caches, simulations, out-of-fold predictions
os.environ.update(POKEMON_DATA=DATA_DIR, POKEMON_OUT=WORK_DIR)
os.makedirs("src", exist_ok=True)

import catboost, lightgbm, sklearn, xgboost
print(f"data: {DATA_DIR} -> {sorted(os.listdir(DATA_DIR))}")
print(f"outputs: {WORK_DIR}; cores: {os.cpu_count()}")
print(f"scikit-learn {sklearn.__version__}, lightgbm {lightgbm.__version__}, "
      f"xgboost {xgboost.__version__}, catboost {catboost.__version__}")'''

RUN_HELPER = '''def run(script, *args):
    """Runs src/<script> in a subprocess, streams its log and stops the notebook if it fails."""
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-u", f"src/{script}", *args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if "Warning" not in line and "warnings.warn" not in line:  # hide library warnings
            print(line, end="")
    if proc.wait():
        raise RuntimeError(f"{script} failed with exit code {proc.returncode}")
    print(f"--- {script}: {(time.time() - t0) / 60:.1f} min")'''

SUBMISSION = '''pred = pd.read_csv(os.path.join(WORK_DIR, "submission_final.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
submission = sample[["battle_id"]].merge(pred, on="battle_id", how="left")
assert len(submission) == len(sample) and submission["player_won"].notna().all()
submission["player_won"] = submission["player_won"].astype(int)
submission.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
print(f"submission.csv written: {len(submission)} rows, share predicted as player 1 win: "
      f"{submission['player_won'].mean():.3f}")
submission.head()'''

SUBMIT_MD = f"""### 4. Submitting

**From the Kaggle website**

1. Click **Save Version** → **Save & Run All (Commit)**.
2. When the run finishes, open the notebook viewer.
3. Under **Output**, use **Submit to Competition**. This submits `submission.csv`.

**From the command line** (Kaggle CLI, after `kaggle kernels push` has run this notebook)

```bash
kaggle kernels status <user>/{SLUG}
kaggle competitions submit {COMPETITION} -k <user>/{SLUG} -f submission.csv -v <version> -m "final 10-model stack"
kaggle competitions submissions {COMPETITION}
```"""


def notebook():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(INTRO), md("### 0. Setup"), code(SETUP),
             md("### 1. Source code\n\nEach cell writes one module to `src/`. The modules are copied "
                "verbatim from the repository; comments are in Italian.")]
    for name, desc in MODULES.items():
        with open(os.path.join(SRC, name), encoding="utf-8") as f:
            source = f.read().rstrip("\n")
        cells += [md(f"**`src/{name}`**: {desc}"), code(f"%%writefile src/{name}\n{source}")]
    cells += [
        md("### 2. Pipeline\n\nEvery step caches its results in `outputs/`."),
        code(RUN_HELPER),
        md("**Monte Carlo features.** 15,000 battles + 30,000 training snapshots × 100 simulations."),
        code('run("simulate.py", "--n-sims", "100")'),
        md("**Recursive roll-out features.**"),
        code('run("recursive.py")'),
        md("**Features, out-of-fold predictions of the 10 base models, meta-learner.** The battle-state "
           "features are built here on first use. The log shows each model's out-of-fold accuracy "
           "and the cross-validated accuracy of the meta-learner."),
        code('run("final_model.py", "--out", "submission_final.csv")'),
        md("### 3. Submission file"),
        code(SUBMISSION),
        md(SUBMIT_MD),
    ]
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = cells
    return nb


def metadata(user):
    return {
        "id": f"{user}/{SLUG}",
        "title": TITLE,
        "code_file": NOTEBOOK,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME", "YOUR_KAGGLE_USERNAME"),
                    help="Kaggle username (owner of the notebook)")
    args = ap.parse_args()
    nbf.write(notebook(), os.path.join(HERE, NOTEBOOK))
    with open(os.path.join(HERE, "kernel-metadata.json"), "w") as f:
        json.dump(metadata(args.user), f, indent=2)
        f.write("\n")
    print(f"written kaggle/{NOTEBOOK} and kaggle/kernel-metadata.json (id {args.user}/{SLUG})")


if __name__ == "__main__":
    main()
