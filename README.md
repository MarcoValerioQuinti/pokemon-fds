# Pokémon Gen 2 OU battle outcome prediction

Classical machine learning (no neural networks) for the Kaggle competition FDS Pokémon Battles
Prediction 2026: given the first 30 turns of a competitive Gen 2 OU battle, predict whether player 1
wins.

The solution stacks diverse base models trained on four kinds of input:
- hand-crafted battle-state features built on a Gen 2 damage engine;
- Monte Carlo simulations of the rest of each battle;
- roll-outs of a learned turn-to-turn transition model;
- snapshot augmentation of the training battles.

| Selected submission | Base models | Meta-learner | 5-fold CV accuracy |
|---|---|---|---|
| `stack10` | 10 diverse models | logistic regression | **0.6956 ± 0.011** |
| `stack22` | all 22 models of the zoo | logistic regression | 0.6935 ± 0.010 |
| `stack10_mean` | the same 10 models | mean of logits | 0.6942 ± 0.011 |

The meta-learner is evaluated with 5-fold CV repeated 5 times; ± is the standard deviation over the
folds. The 95% confidence interval of an accuracy measured on the 10,000 training battles is about
±0.9 points.

## Contents

1. [Data](#1-data)
2. [Repository layout and usage](#2-repository-layout-and-usage)
3. [Feature engineering](#3-feature-engineering)
4. [Simulation-based features](#4-simulation-based-features)
5. [Models and stacking](#5-models-and-stacking)
6. [Validation](#6-validation)
7. [Results and what did not work](#7-results-and-what-did-not-work)
8. [Data sources and domain knowledge](#8-data-sources-and-domain-knowledge)
9. [Hardware and portability](#9-hardware-and-portability)

## 1. Data

The competition files (`train.jsonl` with 10,000 labeled battles, `test.jsonl` with 5,000, the
Gen 2 type chart and the sample submission) go in `data/`; see [`data/README.md`](data/README.md).

For every battle we get:
- **Player 1:** the full team (base stats, types, level, item), without moves.
- **Player 2:** only the lead Pokémon.
- **The timeline:** exactly 30 turns, with each active Pokémon's state (HP, status, boosts,
  effects) and the move it used (name, type, category, power, accuracy, priority).

Facts that shape the design:

- **Most of the outcome happens after turn 30.** In turns 1–30 player 1 suffers about 3× more KOs
  than player 2, yet wins 50% of the games; only 0.6% of the battles are already over.
- **Reported moves are sticky.** The move field repeats a Pokémon's last move even on turns where
  it did not act (about 25% of the moves). Filtering them out lowers the CV by 0.3–0.5 points,
  because the repetitions carry information (time on the field, re-entries), so the raw moves are
  kept.
- **Duplicates.** 27 groups of battles have identical timelines, and their labels always agree.
  Cross-validation keeps them in the same fold. The 7 test battles identical to a training battle
  receive its label (`stacking.duplicate_override`).
- **No leakage and no shift.** `battle_id` has AUC 0.505 for the label, and a train-vs-test
  classifier reaches AUC 0.501. Train and test `battle_id`s overlap, so rows are keyed by
  (`battle_id`, `is_test`).

## 2. Repository layout and usage

```
src/
  dataset.py          paths, reference context learned from the training set, feature cache, bag-of-words
  features.py         battle-state features and the Gen 2 damage engine
  simulate.py         Monte Carlo simulation of the rest of each battle
  recursive.py        turn-to-turn transition model and recursive roll-outs
  stacking.py         base model zoo and out-of-fold predictions with snapshot augmentation
  final_model.py      the stacking configurations and their submissions
  evaluate_nested.py  nested cross-validation of the whole stacking procedure, paired comparisons
  strict_cv.py        cross-validation that re-learns every data-derived component inside each fold
  compute.py          cores, GPU and multiprocessing settings (macOS, Linux, Windows)
tests/test_damage.py  damage engine checked against Pokémon Showdown's Gen 2 formula
kaggle/               the competition notebook (clones this repository) and its guide
data/                 competition data, not versioned
outputs/              caches, out-of-fold predictions and submissions, not versioned
```

```bash
conda create -n pokemon-fds python=3.13 && conda activate pokemon-fds
pip install -r requirements.txt
conda install -c conda-forge llvm-openmp        # macOS only: OpenMP for LightGBM

python src/simulate.py      # Monte Carlo features, ~3 min
python src/recursive.py     # recursive roll-out features, ~2 min
python src/final_model.py   # features + 22 base models + 3 submissions in outputs/, ~12 min
python -m pytest tests -q
```

- **Timings** are on an Apple M5 with 10 cores.
- **Folders:** `POKEMON_DATA` and `POKEMON_OUT` override `data/` and `outputs/`.
- **Caches:** features, simulations and out-of-fold predictions are cached in `outputs/`. Delete
  them after changing the corresponding code.
- **Kaggle:** the notebook in [`kaggle/`](kaggle/README.md) clones this repository at a fixed
  commit and regenerates the 3 selected submissions.

## 3. Feature engineering

Every battle becomes one row describing the state at the last observed turn: about 280 dense
features plus about 990 bag-of-words columns (`features.py`).

- **Side state** (`p1_`, `p2_`, `diff_`):
  - seen, fainted and alive Pokémon;
  - total HP, with the unseen Pokémon at 100%, and mean HP of the alive ones;
  - status counts and status-weighted effective HP.
  - Player 2's unseen Pokémon enter with their expected stats, from team co-occurrence.
- **Trajectory:** HP and KO differences over the turns, checkpoints 5–25 turns before the end,
  slope, minimum and maximum, first and last KO.
- **Team and active Pokémon:**
  - base-stat aggregates of the teams and of the alive Pokémon;
  - for the actives: HP, status, boosts and volatile effects, and their mutual type effectiveness.
- **Play style:** switches, damage dealt, super-effective hits, moves by category, Spikes, Rapid
  Spin, healing, setup, phazing and sleep moves.
- **Damage-based matchups:**
  - a damage matrix between the alive Pokémon, using each one's revealed moves (the species' most
    used attacks when none is revealed, else a generic STAB move);
  - from the matrix: best-hit damage, OHKO/2HKO fractions, "checked" opponents, hits needed to KO
    the opposing team;
  - the duel between the actives, with boosts, paralysis and Substitute;
  - Spikes in chronological order;
  - Rest sleep versus induced sleep;
  - setup that cannot be phazed, alive clerics and status burden.
- **Bag-of-words:** counts of Pokémon (team, alive, seen) and of moves for both sides
  (`CountVectorizer`, `min_df=20`).

The damage engine uses Gen 2 stats at maximum DVs and stat experience, the type-based
physical/special split, STAB, the type chart and the mean damage roll. `tests/test_damage.py`
checks it within 3% of Showdown's exact integer formula (15 tests).

**Nothing is learned from the test set.** The pokédex, Hidden Power types, co-occurrence, move
statistics and bag-of-words vocabulary are learned from the training battles only
(`dataset.build_reference`, `dataset.add_bags`). Test battles are only transformed; a species absent
from the training set takes the stats described in its own battle, or average stats.

## 4. Simulation-based features

- **Monte Carlo** (`simulate.py`): a simplified Gen 2 simulator plays each battle to the end 100
  times from the last observed state.
  - **Randomness:** player 2's unseen Pokémon sampled from co-occurrence, damage rolls, critical
    hits, accuracy, sleep duration, full paralysis, thawing.
  - **Mechanics:** speed order, residual damage, Leftovers, Spikes and healing.
  - **Decisions:** switches out of bad matchups, matchup-based replacements.
  - **Features:** win rate and its spread, final HP difference with quantiles, KOs, remaining turns.
- **Recursive model** (`recursive.py`): four LightGBM models learn one turn of the battle (HP
  deltas and KO probability per side) from 274k transitions of the training battles. From the
  last turn they are applied to their own outputs, 30 roll-outs per battle; the short-horizon HP
  difference is the useful signal.

Both feature sets are also computed on the training snapshots used for augmentation.

## 5. Models and stacking

- **Snapshot augmentation:** inside each training fold, the fold's battles truncated at turns 24,
  26 and 28 are added with weight 0.5. Validation always uses the full 30 turns. Models trained this
  way have the `_aug` suffix.
- **Base model zoo** (`stacking.MODELS`, 22 models):

  | Family | Models |
  |---|---|
  | Gradient boosting | LightGBM, XGBoost and CatBoost, on all features and on the dense set without bag-of-words, with and without augmentation |
  | Bagged trees | ExtraTrees, Random Forest |
  | Linear and kernel | logistic regressions (standardized or quantile-normalized), RBF SVM, kNN on PCA |
  | Gaussian generative | LDA with shrinkage, QDA on PCA, full-covariance Gaussian mixtures on a supervised PLS projection |
  | Noise-regularized | LightGBM and XGBoost trained on copies of the data with Gaussian noise |

  Every scaler and projection is fitted inside the training fold.
- **Out-of-fold predictions:** each base model gets out-of-fold predictions from a stratified
  5-fold split (seed 42). Its test predictions are the mean of the 5 fold models.
- **Meta-learner:** a logistic regression (C = 1) on the logits of the out-of-fold predictions,
  threshold 0.5.
- **The 3 selected configurations** (`final_model.CONFIGS`):
  - **`stack10`:** 10 models kept by a leave-one-model-out cost/benefit analysis of the zoo: the
    boosting models with augmentation, ExtraTrees, Random Forest, LDA and the two logistic
    regressions. It keeps the models that make different errors; the out-of-fold residuals of the
    22 models correlate at 0.97 on average.
  - **`stack22`:** the whole zoo, a hedge against having pruned too closely to the folds.
  - **`stack10_mean`:** the 10 models averaged without learned weights, the most robust to
    overfitting of the stacking layer.

## 6. Validation

- **Local validation only.** All decisions use cross-validation on `train.jsonl`. The public
  leaderboard is used only as an occasional sanity check.
- **Nested CV** (`evaluate_nested.py`): for each of 5 outer folds, the whole stacking procedure is
  re-run on the outer training part, then scores the untouched outer fold:
  1. inner out-of-fold predictions;
  2. fitting the meta-learner;
  3. refitting the base models.

  Duplicates stay in the same fold. With the earlier version of the context, the nested estimate
  of `stack10` was 0.6957 against 0.6994 non-nested: the model selection costs about 0.4 points of
  optimism.
- **Strict CV** (`strict_cv.py`): the pokédex, context statistics, bag-of-words vocabulary, Monte
  Carlo context and transition model are re-learned from each training fold, so the validation
  fold is treated exactly like the test set.
- **Paired comparisons:** variants are compared on the same battles, with a bootstrap 95% CI of the
  accuracy difference and an exact McNemar test.
  - Two similar stacks disagree on about 6% of the battles, so the smallest detectable difference
    is about ±0.5 points.
  - A change is kept only if its gain is reproducible and does not depend on a single fold.

## 7. Results and what did not work

| Model | CV accuracy |
|---|---|
| Sign of the HP difference at turn 30 | 0.631 |
| Single LightGBM | 0.683 |
| Best single base model (`xgb_aug` / `lgbm_aug`) | 0.689 |
| 4-model average (logistic regression, LightGBM, XGBoost, CatBoost) | 0.692 |
| **`stack10`** | **0.6956** |

Tried, measured with paired comparisons and removed:
- hyperparameter tuning with Optuna (no gain);
- filtering the sticky moves (−0.3 to −0.5 points);
- more folds and repeated k-fold (the stack gets worse: averaged predictions lose diversity);
- regime-specific expert models, alone and with a gated meta-learner;
- a median decision threshold (the predictions are already balanced);
- team-wide probabilistic matchup features;
- hidden-team inference from a pool of recurring teams;
- per-Pokémon history features.

These additional features carry real signal (the matchup features alone reach 0.629), but it is
redundant with the battle-state features: they add at most +0.2 points, below the detectable
difference.

The ceiling is informational: the battle is mostly decided after turn 30, and the learning curve
gains only about 0.15 points per extra 1,000 training battles.

## 8. Data sources and domain knowledge

The only data used are the competition files and the supplied type chart. No external databases,
replay datasets, usage statistics or hand-collected tables are used.

| Source | What comes from it |
|---|---|
| Competition data (training battles) | Pokédex (base stats, types); move attributes (type, category, base power, accuracy, priority, all read from `move_details`); Hidden Power type per species, inferred from observed damage; team co-occurrence; move frequencies per species, which also identify phazers and clerics |
| Supplied Gen 2 type chart | All type multipliers |
| Gen 2 game rules, encoded as mechanics | Stat and damage formulas; physical/special split by type; STAB ×1.5; damage roll 217–255/255; critical hits 17/256; stat-stage multipliers; Spikes 1/8; Leftovers 1/16; poison and burn 1/8, Toxic n/16; paralysis (speed ×0.25, 25% full paralysis); sleep (Rest 2 turns, induced 1–7); thaw 10%; Thick Club and Light Ball; fixed-power moves (Return and Frustration 102, Hidden Power 70, Seismic Toss and Night Shade equal to the level); which moves heal, set up, phaze or put to sleep |
| Modelling choices | Simulator heuristics, such as switching when the matchup score is below −2 and healing below 40% HP; the generic 80-power STAB move for Pokémon with no revealed attack |

Pokémon Showdown's Gen 2 damage code is used only as a reference in the tests.

## 9. Hardware and portability

`compute.py` lets every script run on macOS, Linux and Windows.

- **Settings:** `--n-jobs` / `POKEMON_N_JOBS` (default: all cores), `--device` / `POKEMON_DEVICE`
  (`auto`, `cpu`, `cuda`) and `POKEMON_MP` (`fork` or `spawn`).
- **Multiprocessing:** workers receive their data through the pool initializer, so the results are
  identical with `fork`, `spawn` and a single process.
- **Fold-level parallelism:** with 10k rows, boosting scales poorly beyond ~3 threads, so several
  folds train at once with fewer threads each. On the M5 this is 1.5× faster for LightGBM and
  XGBoost, and the predictions are bit-identical.
- **GPU:** with CUDA, XGBoost and CatBoost train on the GPU. These libraries have no Apple GPU
  backend.

## License

The code is released under the [MIT License](LICENSE). The competition data is not covered by this
license and is subject to the Kaggle competition rules.
