"""Builds the Kaggle notebook that clones this repository at a fixed commit and generates the
3 selected final submissions. It also writes kernel-metadata.json for `kaggle kernels push`.

The notebook contains no project code: it clones the GitHub repository (as required by the
competition rules), runs the pipeline on the competition data and writes one CSV per selected
submission to /kaggle/working, each preceded by a short markdown cell explaining the approach.

Usage:
    python kaggle/build_kaggle_notebook.py --user <kaggle-username> [--commit <sha>]
The default commit is the current HEAD, which must already be pushed to GitHub.
"""
import argparse
import json
import os
import subprocess

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/MarcoValerioQuinti/pokemon-fds.git"
COMPETITION = "fds-pokemon-battles-prediction-2026"
SLUG = "pokemon-fds-final"
TITLE = "Pokemon FDS Final"  # Kaggle derives the slug from the title: it must match SLUG
NOTEBOOK = f"{SLUG}.ipynb"

# the 3 selected submissions: (config name in src/final_model.py, explanation shown before it)
SUBMISSIONS = [
    ("stack10", """### Submission 1: `stack10`, the final 10-model stack

The main solution. Ten base models were chosen by a cost/benefit analysis of a 22-model zoo, keeping
the ones that make *different* errors:
- LightGBM, XGBoost and CatBoost trained with snapshot augmentation (on the full feature set and on the
  dense set without bag-of-words);
- ExtraTrees and Random Forest;
- LDA and two logistic regressions.

A logistic regression on the logits of their out-of-fold predictions combines them."""),
    ("stack22", """### Submission 2: `stack22`, the full 22-model stack

The same features and meta-learner, applied to the whole model zoo:
- the 10 models of submission 1;
- boosting without augmentation and Gaussian-noise variants;
- SVM, kNN, QDA and Gaussian-mixture classifiers.

It is a more redundant but larger ensemble: a hedge in case the pruning of submission 1 was
tuned too closely to the training folds."""),
    ("stack10_mean", """### Submission 3: `stack10_mean`, the 10 models averaged

The 10 base models of submission 1, combined by a plain average of their logits instead of learned
weights. With no meta-learner to fit, it is the most robust to overfitting of the stacking layer."""),
]

INTRO = """# FDS Challenge: Pokémon Battle Prediction — final submissions

Classical machine learning only (no neural networks). Given the first 30 turns of a Gen 2 OU battle,
we predict whether player 1 wins. This notebook clones the
[GitHub repository](https://github.com/MarcoValerioQuinti/pokemon-fds) at a fixed commit and
regenerates our **3 selected final submissions**.

**Pipeline** (code in `src/`; methodology in the repository README):

1. **Battle-state features** (`features.py`): about 280 features from a Gen 2 damage engine (HP, KOs,
   status, trajectory, damage matrix between the alive Pokémon, Spikes, sleep, win conditions), plus
   bag-of-words of Pokémon and moves.
2. **Monte Carlo features** (`simulate.py`): every battle is played to the end 100 times from turn 30.
3. **Recursive features** (`recursive.py`): roll-outs of a gradient-boosted turn-to-turn transition model.
4. **Snapshot augmentation** (`stacking.py`): training battles truncated at turns 24/26/28 are added
   inside the training folds, with weight 0.5.
5. **Stacking** (`final_model.py`): out-of-fold predictions of the base models (stratified 5-fold,
   seed 42), combined by a meta-learner.

**No test information is learned.** Pokédex, Hidden Power types, co-occurrence, move statistics,
bag-of-words vocabulary, the Monte Carlo context and the transition model are fitted on the training
battles only; the test battles are only transformed.

**Settings:**
- Accelerator **None**: about 1 hour on Kaggle's 4 CPU cores. The run is pinned to the CPU (`DEVICE`
  below), so the selected accelerator never changes the predictions. A GPU would only speed up
  XGBoost and CatBoost, with slightly different results; a TPU is not used by any of these libraries.
- Internet **on**, needed to clone the repository."""

SETUP = '''import glob
import os
import subprocess
import sys
import time

import pandas as pd

REPO = "{repo}"
COMMIT = "{commit}"  # exact version of the code that produced the selected submissions
DEVICE = "cpu"  # "cpu" reproduces the selected submissions; "auto" uses an NVIDIA GPU for XGBoost/CatBoost


def find_data_dir():
    """Competition data: mounted under /kaggle/input on Kaggle, POKEMON_DATA or ./data elsewhere."""
    hits = sorted(glob.glob("/kaggle/input/**/train.jsonl", recursive=True))
    if hits:
        return os.path.dirname(hits[0])
    return os.environ.get("POKEMON_DATA", os.path.abspath("data"))


DATA_DIR = find_data_dir()
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
CODE_DIR = os.path.join(OUT_DIR, "pokemon-fds")
WORK_DIR = os.path.join(OUT_DIR, "outputs")  # feature caches, simulations, out-of-fold predictions
if not os.path.isdir(CODE_DIR):
    subprocess.run(["git", "clone", "--quiet", REPO, CODE_DIR], check=True)
subprocess.run(["git", "-C", CODE_DIR, "checkout", "--quiet", COMMIT], check=True)
os.environ.update(POKEMON_DATA=DATA_DIR, POKEMON_OUT=WORK_DIR, POKEMON_DEVICE=DEVICE)

import catboost, lightgbm, numpy, sklearn, xgboost
print("code:", subprocess.run(["git", "-C", CODE_DIR, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip())
print(f"data: {{DATA_DIR}} -> {{sorted(os.listdir(DATA_DIR))}}")
print(f"device: {{DEVICE}}; cores: {{os.cpu_count()}}; numpy {{numpy.__version__}}, scikit-learn {{sklearn.__version__}}, "
      f"lightgbm {{lightgbm.__version__}}, xgboost {{xgboost.__version__}}, catboost {{catboost.__version__}}")'''

RUN_HELPER = '''def run(script, *args):
    """Runs src/<script> of the cloned repository, streams its log and stops the notebook if it fails."""
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-u", os.path.join(CODE_DIR, "src", script), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if "Warning" not in line and "warnings.warn" not in line:  # hide library warnings
            print(line, end="")
    if proc.wait():
        raise RuntimeError(f"{script} failed with exit code {proc.returncode}")
    print(f"--- {script}: {(time.time() - t0) / 60:.1f} min")


def write_submission(config, n):
    """Aligns the predictions of `config` to sample_submission.csv and writes submission_<n>_<config>.csv."""
    pred = pd.read_csv(os.path.join(WORK_DIR, f"submission_{config}.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    sub = sample[["battle_id"]].merge(pred, on="battle_id", how="left")
    assert len(sub) == len(sample) and sub["player_won"].notna().all()
    sub["player_won"] = sub["player_won"].astype(int)
    path = os.path.join(OUT_DIR, f"submission_{n}_{config}.csv")
    sub.to_csv(path, index=False)
    print(f"{os.path.basename(path)}: {len(sub)} rows, player 1 predicted to win in {sub.player_won.mean():.1%}")
    return sub'''

SUBMIT_MD = f"""### Submitting

On the Kaggle website, open this notebook's **Output** and use **Submit to Competition** on each of
the three `submission_<n>_<config>.csv` files.

From the Kaggle CLI, with the version number of this notebook run:

```bash
kaggle competitions submit {COMPETITION} -k <user>/{SLUG} -v <version> -f submission_1_stack10.csv -m "stack10"
```"""


def notebook(commit):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    configs = ",".join(c for c, _ in SUBMISSIONS)
    cells = [
        md(INTRO),
        md("### Setup: clone the repository at the fixed commit"),
        code(SETUP.format(repo=REPO, commit=commit)),
        code(RUN_HELPER),
        md("### Pipeline\n\nEvery step caches its results in `outputs/`. The log shows the out-of-fold "
           "accuracy of each base model and the cross-validated accuracy of each configuration."),
        md("**Monte Carlo features** (15,000 battles + 30,000 training snapshots, 100 simulations each)."),
        code('run("simulate.py", "--n-sims", "100")'),
        md("**Recursive roll-out features** (transition model trained on the training battles only)."),
        code('run("recursive.py")'),
        md("**Battle-state features, out-of-fold predictions of all base models, the 3 configurations.**"),
        code(f'run("final_model.py", "--configs", "{configs}")'),
    ]
    for n, (config, text) in enumerate(SUBMISSIONS, 1):
        cells += [md(text), code(f'write_submission("{config}", {n}).head()')]
    cells.append(md(SUBMIT_MD))
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
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME", "YOUR_KAGGLE_USERNAME"),
                    help="Kaggle username (owner of the notebook)")
    ap.add_argument("--commit", default=None, help="commit to clone (default: current HEAD)")
    args = ap.parse_args()
    commit = args.commit or subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True,
                                           text=True, check=True).stdout.strip()
    nbf.write(notebook(commit), os.path.join(HERE, NOTEBOOK))
    with open(os.path.join(HERE, "kernel-metadata.json"), "w") as f:
        json.dump(metadata(args.user), f, indent=2)
        f.write("\n")
    print(f"written kaggle/{NOTEBOOK} (commit {commit[:7]}) and kaggle/kernel-metadata.json (id {args.user}/{SLUG})")


if __name__ == "__main__":
    main()
