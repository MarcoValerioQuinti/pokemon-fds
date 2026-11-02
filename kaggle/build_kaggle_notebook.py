"""Build a repository-cloning notebook for the unchanged original final model."""
import argparse
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPOSITORY = 'https://github.com/MarcoValerioQuinti/pokemon-fds.git'


def build(commit='', repository=REPOSITORY):
    if commit and not re.fullmatch(r'[0-9a-fA-F]{40}', commit):
        raise ValueError('Use a full 40-character Git commit SHA')
    cells = []

    def add(kind, source):
        cell = {'cell_type': kind, 'id': f'cell-{len(cells):02d}', 'metadata': {},
                'source': source.splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# Pokémon FDS — original final model

This notebook clones the repository and executes the original Monte Carlo, recursive
features and ten-model stacking pipeline. Model source, features, hyperparameters and
seeds are preserved. No project modules are embedded in cells.

Completed local reproduction: repeated meta-CV accuracy **0.69958**, fold-score SD
**0.01009**. The historical **0.6957** whole-stack nested estimate remains unverified.
These are different evaluation protocols.

Public scores supplied by the team (not independently retrieved):
- Marco, `submission_final.csv`: **0.70133**.
- Marco, `submission_stacking_mc_rec.csv`: **0.70066**.
- Luca, notebook Version 1: **0.69466**.

This notebook reproduces the original final configuration only. Exact reconstruction
of the other two entries requires their original configurations/code; they are not
replaced with newly invented alternatives. No claim is made that the local regenerated
CSV is byte-identical to Marco's uploaded file.

CPU; Internet enabled for Git and dependency installation. Attach the four official
files after accepting the rules. If Internet is prohibited, clarify the cloning
requirement with the organizers. Check mechanics provenance and team-sharing eligibility
in `COMPLIANCE.md`; a public score does not certify rule compliance.
''')
    add('code', f'''import os
from pathlib import Path
REPO_URL = os.environ.get("POKEMON_REPO_URL", {repository!r})
REPO_COMMIT = os.environ.get("POKEMON_COMMIT", {commit!r})
DATA_DIR = os.environ.get("POKEMON_DATA", "")
REPO = Path("/kaggle/working/pokemon-fds")
OUT = Path("/kaggle/working/original-model")
N_JOBS = 2
''')
    add('code', '''import re
import subprocess
import sys
assert re.fullmatch(r"[0-9a-fA-F]{40}", REPO_COMMIT), "Set a full published Git SHA containing this workflow"
if not REPO.exists():
    subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(REPO)], check=True)
else:
    assert subprocess.check_output(["git", "-C", str(REPO), "remote", "get-url", "origin"], text=True).strip() == REPO_URL
    assert not subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip(), "Checkout has local changes"
subprocess.run(["git", "-C", str(REPO), "fetch", "origin", REPO_COMMIT], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", REPO_COMMIT], check=True)
assert subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip().lower() == REPO_COMMIT.lower()
assert (REPO / "src/final_model.py").is_file()
subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REPO / "requirements-final.txt")], check=True)
''')
    add('code', '''required = ("train.jsonl", "test.jsonl", "type_chart_gen2.csv", "sample_submission.csv")
if DATA_DIR:
    DATA = Path(DATA_DIR)
else:
    matches = sorted({p.parent for p in Path("/kaggle/input").rglob("train.jsonl")
                      if all((p.parent / name).is_file() for name in required)})
    assert len(matches) == 1, f"Set DATA_DIR explicitly; matching folders: {matches}"
    DATA = matches[0]
assert all((DATA / name).is_file() for name in required)
assert not OUT.exists(), "Use a fresh output directory; original caches have no fingerprints"
OUT.mkdir(parents=True)
env = dict(os.environ, POKEMON_DATA=str(DATA), POKEMON_OUT=str(OUT),
           POKEMON_DEVICE="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
''')
    add('markdown', '''## Original simulation and recursive features

Run the original modules with their defaults: 100 Monte Carlo simulations, 30 recursive
rollouts, original training snapshots and seed policies. All computation runs in fresh
Python subprocesses using the pinned repository. No additional nested evaluation is run.
''')
    add('code', '''for module in ("simulate.py", "recursive.py"):
    subprocess.run([sys.executable, str(REPO / "src" / module), "--n-jobs", str(N_JOBS)],
                   cwd=REPO, env=env, check=True)
''')
    add('markdown', '''## Submission — original ten-model stack

The original final model combines augmented LightGBM/XGBoost/CatBoost, dense boosted
variants, ExtraTrees, Random Forest, LDA and logistic models. A logistic meta-learner
fits base OOF logits. This is Marco's final-model configuration, associated with the
team-reported public score 0.70133; that score does not describe this unexecuted Kaggle
rerun. The original module also prints repeated meta-CV diagnostics and retains its
seven-match duplicate-label policy. Rule questions remain documented.
''')
    add('code', '''subprocess.run([sys.executable, str(REPO / "src/final_model.py"),
                "--n-jobs", str(N_JOBS), "--device", "cpu", "--out", "submission_final.csv"],
               cwd=REPO, env=env, check=True)
import shutil
submission = Path("/kaggle/working/submission_final.csv")
shutil.copy2(OUT / "submission_final.csv", submission)
print(submission)
''')
    add('markdown', '''## Remaining delivery

This creates one original-model CSV, not three selected Kaggle entries. Recover the
exact configurations for `submission_stacking_mc_rec.csv` and Luca's Version 1 before
claiming all three can be reproduced here. No replacement candidates are fabricated.

No upload or selection is performed. Per the task, select three entries before
November 5, 2026, 23:59 (confirm official timezone), at most three uploads daily.
Make the notebook public after the competition and submit notebook/report by
November 12, 2026. The report is deferred. This revised notebook has not been run on Kaggle.
''')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
                'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
                'nbformat': 4, 'nbformat_minor': 5}
    (HERE / 'pokemon-fds-final.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
    return notebook


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--commit', default='')
    ap.add_argument('--repository', default=REPOSITORY)
    args = ap.parse_args()
    build(args.commit, args.repository)
