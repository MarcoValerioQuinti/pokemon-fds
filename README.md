# Pokémon Gen 2 OU battle outcome prediction

Classical machine-learning pipeline (no neural networks) for the Kaggle competition
FDS Pokémon Battles Prediction 2026:
given the first 30 turns of a competitive Gen 2 OU battle, predict whether player 1 wins.

**Final model:** a stack of 10 diverse base models with a logistic-regression meta-learner, built on
hand-crafted battle-state features plus Monte Carlo and recursive roll-out features.

| Estimate | Accuracy |
|---|---|
| Out-of-fold, meta evaluated with 5×5 repeated CV | **0.6994 ± 0.010** |
| Nested CV of the whole stacking procedure (5 outer × 5 inner folds) | **0.6957** (folds 0.686–0.710) |
| 95% confidence interval of the OOF accuracy | about ±0.9 points |

All results are local cross-validation estimates; nothing has been tuned on the Kaggle leaderboard.

---

## Contents

1. [The problem and the data](#1-the-problem-and-the-data)
2. [Repository layout](#2-repository-layout)
3. [Quick start](#3-quick-start)
4. [Feature engineering](#4-feature-engineering)
5. [Simulation-based features](#5-simulation-based-features)
6. [Models and stacking](#6-models-and-stacking)
7. [Evaluation methodology](#7-evaluation-methodology)
8. [Results](#8-results)
9. [Statistical reports](#9-statistical-reports)
10. [What did not work](#10-what-did-not-work)
11. [Hardware and portability](#11-hardware-and-portability)

---

## 1. The problem and the data

| File | Content |
|---|---|
| `data/train.jsonl` | 10,000 battles with the label `player_won` (exactly 50% wins) |
| `data/test.jsonl` | 5,000 battles to predict |
| `data/type_chart_gen2.csv` | Gen 2 type-effectiveness chart |
| `data/sample_submission.csv` | Submission format (`battle_id`, `player_won`) |

For every battle we get:

- **Player 1:** the full team of 6 Pokémon (base stats, types, level, item). Moves are *not* given.
- **Player 2:** only the lead Pokémon. The rest of the team appears only when it enters the field.
- **The timeline:** always exactly **30 turns**. Each turn has each active Pokémon's state (HP %,
  status, boosts, volatile effects) and the move it used (name, type, category, base power,
  accuracy, priority).

Facts about the data that shape the whole design:

- **Most of the outcome is not observable.** Battles usually continue well past turn 30.
  - In turns 1–30, player 1 suffers about 3× more KOs than player 2 (1.69 vs 0.50 on average),
    yet wins 50% of the games.
  - Only 0.6% of the battles are already over at turn 30.
  - So the dynamics after turn 30 are different from those observed.
- **Reported moves are sticky.** After a Pokémon's first move, the move field keeps showing its
  *last* move even on turns where it did not act (switch-ins, sleep, freeze, replacement after a
  KO). About 25% of the reported moves were never executed. A helper detects real actions, but
  the raw moves are kept for most features because they carry useful signal (time on field,
  re-entries).
- **Teams come from a recurring pool.**
  - The 15,000 player-1 teams contain only 5,607 distinct teams.
  - 75% of the player-2 teams seen in full also appear as player-1 teams.
  - This lets us infer player 2's hidden Pokémon by matching whole teams.
- **Duplicates.** There are 27 groups of battles with identical timelines, and their labels are
  always consistent.
  - Cross-validation keeps them in the same fold.
  - The 7 test battles identical to a training battle receive its label.
- **No leakage, no shift.**
  - `battle_id` has AUC 0.505 for the label.
  - A train-vs-test adversarial classifier reaches AUC 0.501.
  - No feature has a significantly different train/test distribution.
  - Train and test `battle_id`s overlap, so rows are always keyed by (`battle_id`, `is_test`).

## 2. Repository layout

```
data/                      competition data, not versioned (see data/README.md)
src/
  features.py              battle-state features, Gen 2 damage engine, label-free context
  train.py                 feature cache + first 4-model prototype
  simulate.py              Monte Carlo simulator of the rest of the battle
  recursive.py             turn-by-turn transition model and recursive roll-outs
  strategic.py             team-wide matchup and probabilistic KO features (experimental)
  team_infer.py            hidden-team inference from the team pool + reconstruction test
  longitudinal.py          per-Pokémon history features (experimental)
  stacking.py              base model zoo, out-of-fold predictions, meta-learner
  final_model.py           final 10-model stack and submission
  evaluate_nested.py       nested CV of the whole stacking procedure + paired comparisons
  analyze_errors.py        OOF error analysis by game regime
  compute.py               cores / GPU / multiprocessing configuration (macOS, Linux, Windows)
tests/test_damage.py       damage engine checked against Pokémon Showdown's Gen 2 formula
reports/
  01_report_statistico_dati.ipynb     data statistics and significance
  02_report_costi_benefici.ipynb      cost/benefit of every method and model
  build_notebooks.py                  generates both notebooks
outputs/                   caches, OOF predictions, experiments, submissions (generated)
```

The notebooks are written in Italian, and their content is summarized in [section 9](#9-statistical-reports).
They are committed with their outputs. Re-running them needs the cached predictions of every
experiment in `outputs/`, which is not versioned: the OOF folders and the nested experiments
produced by `stacking.py` and `evaluate_nested.py`.

## 3. Quick start

First download the competition data into `data/` (see [`data/README.md`](data/README.md)).

```bash
conda create -n pokemon-fds python=3.13
conda activate pokemon-fds
pip install -r requirements.txt
# macOS only: LightGBM needs OpenMP
conda install -c conda-forge llvm-openmp

python src/train.py           # base features + snapshots (cached in outputs/), ~30 s
python src/simulate.py        # Monte Carlo features, ~3 min
python src/recursive.py       # recursive roll-out features, ~2 min
python src/final_model.py     # 10 base models + meta-learner -> outputs/submission_final.csv, ~6 min
```

Other entry points:

```bash
python src/stacking.py --models lgbm,xgb --extra mc,rec      # OOF of single base models (cached per model)
python src/stacking.py --meta --extra mc,rec                 # meta-learner over every cached OOF
python src/evaluate_nested.py --name X --models a,b --extra mc,rec
python src/evaluate_nested.py --compare X Y                  # paired accuracy difference with bootstrap CI
python src/analyze_errors.py --name X
python -m pytest tests -q
```

**Caches.** Features, simulations and OOF predictions are cached in `outputs/`.

- If `features.py` changes, delete `outputs/features*.pkl` and the OOF folders.
- If a simulator changes, delete its `.pkl` and the OOF folders that use it.

Timings are on an Apple M5 with 10 cores.

## 4. Feature engineering

Every battle becomes one row describing the state at the **last observed turn**. There are about
280 dense features, plus about 990 bag-of-words columns.

### Side state (`p1_`, `p2_`, `diff_` prefixes)

- Seen, fainted and alive Pokémon.
- **Total HP** (unseen Pokémon counted at 100%), mean HP of the alive Pokémon, count of low-HP
  Pokémon.
- Status counts; healthy Pokémon (neither asleep nor frozen).
- Status-weighted **effective HP** (sleep/freeze ×0.5, paralysis/toxic ×0.8).
- Player 2's unseen Pokémon enter with their *expected* stats (see the label-free context below),
  so both sides are compared on the same footing.

### Trajectory

- HP and KO differences at every turn.
- Checkpoints 5, 10, 15, 20 and 25 turns before the end.
- Slope, min, max and mean of the HP difference; change over the last 10 turns.
- First and last KO turn.

### Team and active Pokémon

- Base-stat aggregates of player 1's team and of player 2's lead; stat sums of the alive Pokémon;
  HP-weighted base-stat total.
- For the active Pokémon at the last turn: HP, status, boosts, volatile effects (substitute,
  leech seed, confusion…), mutual STAB type effectiveness.

### Play style

- Switches, turns without a move, damage dealt, super-effective hits, moves by category.
- Spikes, rapid spin, healing, setup and phazing moves, sleep moves, explosions, number of
  distinct moves.

### Matchup features from a Gen 2 damage engine

The damage engine:

- uses stats at maximum DVs and stat experience: `((base + 15) * 2 + 63) * L / 100 + 5`;
- decides the physical/special category by **type**, as in Gen 2;
- applies STAB, the type chart, Thick Club / Light Ball, and Explosion halving defense;
- uses the mean random roll (217–255/255) and scales damage by accuracy;
- gives Hidden Power the type that best explains the observed damage; Return and Frustration
  power 102; Seismic Toss and Night Shade damage equal to the level.

`tests/test_damage.py` re-implements Pokémon Showdown's integer Gen 2 formula and checks the
engine within 3% (15 tests).

On top of the engine, the matchup features are:

- **Damage matrix** between alive Pokémon, using each Pokémon's revealed moves. If none are
  revealed, the species' most common attacking moves are used; failing that, a generic STAB move.
  From the matrix: best-hit damage, OHKO/2HKO fractions, OHKO before being hit, "checked"
  opponents, hits needed to KO the whole opposing team.
- **Active duel:** with boosts, paralysis, substitute and speed order.
- **Spikes:** tracked chronologically (Rapid Spin fails against Ghost types).
- **Sleep:** Rest sleep (about to wake) is separated from induced sleep (turns already used).
- **Win conditions:** unphazable setup, alive clerics, status burden.

### Bag of words

Counts of Pokémon (team, alive, seen) and moves for both sides (`CountVectorizer`,
`min_df=20`), plus active-status dummies.

### Label-free context

Some statistics are computed once on **train + test together without ever touching the
label**, and are reused identically at inference:

- species pokédex;
- Hidden Power types;
- species co-occurrence;
- most common moves per species;
- the team pool;
- the bag-of-words vocabulary.

Scalers and projections (StandardScaler, QuantileTransformer, PCA, PLS) are always fit inside
the folds.

## 5. Simulation-based features

### Monte Carlo (`simulate.py`)

A simplified Gen 2 simulator plays each battle to the end from the last observed state, 100
times.

- **Hidden information:** player 2's missing Pokémon are sampled from team co-occurrence.
- **Randomness:** damage rolls, critical hits (17/256), accuracy.
- **Mechanics:** speed order (paralysis ×0.25); sleep duration (Rest = 2 turns, induced = random);
  thawing; poison, burn and toxic damage; Leftovers; Spikes on entry; healing below 40% HP.
- **Decisions:** voluntary switches out of bad matchups, matchup-based replacement choice.

Features: simulated win rate and its spread, final HP difference with p10/p90 quantiles, KOs,
remaining turns. `mc_win` alone reaches 0.63 accuracy.

### Recursive model (`recursive.py`)

An RNN-like recursion built with gradient boosting:

- **State:** an aggregate state per turn, with an exponential moving average of past HP and KO
  deltas as "memory".
- **Transitions:** four LightGBM models learn turn-to-turn transitions (HP deltas and KO
  probability per side) from about 411k unlabeled transitions.
- **Roll-out:** 30 roll-outs per battle, from turn 30 until a team is exhausted.

Only the short horizon is informative: the HP difference after 10 simulated turns has AUC 0.718.
The simulated winner is not (AUC 0.57), because the model only sees turns 1–30.

Both feature sets are also computed on training snapshots, for augmentation.

## 6. Models and stacking

### Snapshot augmentation

Every training battle is also truncated at turns 24, 26 and 28. The truncated copies have the
same label and weight 0.5.

- They are added **only inside the training folds**, and only for the battles of that fold.
- Validation always uses the turn-30 state.
- Models trained this way have the `_aug` suffix.

### Final stack (`final_model.py`)

| Family | Base models | Feature set |
|---|---|---|
| Gradient boosting with augmentation | `lgbm_aug`, `xgb_aug`, `catboost_aug` | all features |
| Gradient boosting with augmentation | `lgbm_dense_aug`, `catboost_dense_aug` | dense features (no bag of words) |
| Bagged deep trees | `extratrees`, `rf` | all features |
| Linear / generative | `lda` (shrinkage), `logreg` (standardized) | all / dense |
| Linear / generative | `logreg_dense` (quantile-normalized) | dense |

- **Base models:** each produces out-of-fold (OOF) probabilities with a stratified 5-fold split
  (seed 42). Test predictions are averaged over the folds.
- **Meta-learner:** a logistic regression (C = 1) on the logits of the OOF probabilities, with
  threshold 0.5.
- **Duplicates:** the 7 test battles that duplicate a training battle take its label.

The 10 models were chosen by a cost/benefit analysis (section 9). Starting from a zoo of 22 base
models, we kept the ones that bring **diversity** and dropped the redundant ones.

| Dropped | Why |
|---|---|
| SVM, kNN, QDA, Gaussian mixtures | No marginal value |
| LightGBM / XGBoost / CatBoost without augmentation | Redundant with the `_aug` versions |
| Gaussian-noise variants | No marginal value |
| Regime-specific experts | Slightly harmful |

## 7. Evaluation methodology

- **OOF evaluation:** stratified 5-fold for the base models. The meta-learner is evaluated with a
  5-fold CV repeated 5 times (seed 43).
- **Nested evaluation (`evaluate_nested.py`).** For each of 5 outer folds, the whole stacking
  procedure is re-run on the outer training part:
  1. inner 5-fold OOF;
  2. fitting the meta-learner;
  3. refitting the base models;
  4. predicting the untouched outer fold.

  Duplicates are grouped in the same fold, and snapshots only ever come from training battles.
- **Paired comparisons:** variants are compared on the *same* battles, with a bootstrap 95% CI of
  the accuracy difference (5,000 resamples) and an exact McNemar test.
  - Two similar stacks disagree on about 6% of the battles.
  - So the smallest detectable difference is about **±0.5 accuracy points**.
- **Acceptance rule:** a change is kept only if the gain is reproducible, is not explained by
  leakage, and does not depend on a single fold.

## 8. Results

### Progression

| Model | CV accuracy |
|---|---|
| Baseline: sign of the KO difference | 0.581 |
| Baseline: sign of the HP difference | 0.631 |
| Single LightGBM, first feature set | 0.683 |
| 4-model average (logreg, LightGBM, XGBoost, CatBoost) | 0.692 |
| 17-model stack | 0.693 |
| + snapshot augmentation + Monte Carlo + recursive features, 22 models | 0.6981 ± 0.011 |
| **Final 10-model stack** | **0.6994 ± 0.010** |

### Final vs previous best

Final 10-model stack against the 22-model stack, on the same battles:

- **Accuracy:** +0.15 points, 95% CI [−0.15, +0.45], McNemar p = 0.36. This is statistically the
  same accuracy.
- **Cost:** 43% less training time.
- **Predictions:** the two submissions agree on 98.7% of the test battles.

**Unbiased estimate.** The 10 models were chosen by looking at the same OOF predictions, so
0.6994 is slightly optimistic. The nested CV of the full final procedure gives **0.6957**.

Against the nested 4-model baseline (logreg, LightGBM, XGBoost, CatBoost: 0.6911):

- the difference is +0.46 points, 95% CI [−0.03, +0.96];
- the final stack is better in all 5 outer folds;
- P(diff > 0) = 0.96.

Removing the Monte Carlo and recursive features from the final stack would cost 0.44 points
(p = 0.11): with augmented models they are worth keeping, because they are also computed on the
snapshots.

### Single base models (OOF)

| Model | Accuracy | Train-OOF gap | Time (5 folds) |
|---|---|---|---|
| xgb_aug | 0.692 | 0.09 | 32 s |
| extratrees | 0.689 | 0.31 | 56 s |
| lgbm_aug | 0.689 | 0.11 | 23 s |
| catboost_aug | 0.689 | 0.27 | 93 s |
| lda | 0.687 | 0.02 | 1 s |
| logreg_dense | 0.682 | 0.03 | 2 s |

## 9. Statistical reports

### `01_report_statistico_dati.ipynb`: the data and how significant the results are

- **Classes:** exactly balanced.
- **State vs win:** the win probability rises monotonically with the HP difference, from 0.16 in
  the lowest decile to 0.83 in the highest (Cramér's V 0.37). The central deciles are close to a
  coin flip.
- **Features:**
  - 168 of 269 features are significant after Benjamini-Hochberg correction;
  - only 36 have |AUC − 0.5| ≥ 0.1;
  - the strongest ones are highly correlated with each other: they all measure "who is ahead in
    HP".
- **Learning curve:** it flattens.

  | Training battles | Accuracy |
  |---|---|
  | 400 | 0.640 |
  | 3,200 | 0.667 |
  | 8,000 | 0.681 |

  The remaining slope is about +0.15 points per extra 1,000 battles.
- **The 0.70 mark:** the 95% CI of the stack's accuracy is about [0.689, 0.707], so 0.70 cannot be
  told apart from 0.698 on this data.

### `02_report_costi_benefici.ipynb`: cost vs benefit of every component

- **The ensemble pays off.**
  - It beats the best single model by +0.6 points (p = 0.04).
  - Its learned weights add little over a plain average of logits (+0.3, p = 0.19).
- **Diversity matters more than individual accuracy.**
  - OOF residuals of the 22 models correlate at 0.97 on average.
  - In a leave-one-model-out ablation, the largest contributions come from ExtraTrees, Random
    Forest and the augmented CatBoost models, despite their weaker standalone accuracy.
- **Cheap stacks lose.**
  - A stack of only cheap models loses 0.6 points (p = 0.01).
  - The 4 classic models lose 0.9 points (p < 0.001).
  - A nested greedy selection loses 0.6 points (p = 0.02).
- **Components**, as accuracy difference against their own baseline, on the same battles:

  | Component | Δ accuracy (points) | 95% CI | Extra cost | Verdict |
  |---|---|---|---|---|
  | Snapshot augmentation | +0.30 | [−0.08, +0.67] | 3.5 min | best value |
  | Team-wide strategic matchups + team pool | +0.17 | [−0.26, +0.60] | 2 min | not proven, dropped |
  | Per-Pokémon history | +0.12 | [−0.34, +0.59] | 14 s | not proven, dropped |
  | Regime experts (with/without gating) | −0.06 / −0.13 | crosses 0 | | dropped |
  | 10 folds / 5-fold × 3 repeats | −0.29 / −0.41 | crosses 0 | ×2.2 / ×3 time | dropped |

## 10. What did not work

All of the following were measured with paired comparisons and dropped:

- **Hyperparameter tuning** of LightGBM with Optuna: 0.6877 vs 0.6879 with the defaults.
- **Filtering sticky moves** from the features: −0.3 to −0.5 points, because the repetitions
  carry information.
- **More folds or repeated k-fold:** the train/OOF gap stays the same. Bagging improves single
  models but makes OOF predictions more similar, so the stack gets worse.
- **Gaussian input noise:** it helps XGBoost alone, but adds nothing to the stack.
- **Regime experts** (attrition, balanced, setup games) and a regime-gated meta-learner.
- **A median decision threshold:** the predictions are already balanced.
- **Probabilistic KO features** from the exact damage distribution, team-wide matchup
  assignment, hidden-team inference from the team pool, and per-Pokémon histories.
  - These carry real signal: the strategic features alone reach 0.629.
  - Team-pool inference recovers 55% of the hidden Pokémon vs 32% for pairwise co-occurrence.
  - But the signal is redundant with the state features already in the stack.

The ceiling is informational: what happens after turn 30 is not in the data, and every new
direction tried lands within ±0.3 points of the current result.

## 11. Hardware and portability

`src/compute.py` makes every script run on macOS, Linux and Windows.

| Setting | Values | Default |
|---|---|---|
| `--n-jobs N` / `POKEMON_N_JOBS` | worker count | all cores |
| `--device` / `POKEMON_DEVICE` | `auto`, `cpu`, `cuda` | `auto`: CUDA when `nvidia-smi` finds a GPU, never on macOS |
| `POKEMON_MP` | `fork`, `spawn` | `fork` on macOS/Linux, `spawn` on Windows |

- **Multiprocessing:** workers receive data through an initializer, so the code works with both
  `fork` and `spawn`.
- **Fold-level parallelism** (`compute.fold_plan`). With 10k rows, gradient boosting does not
  scale past a few threads, especially on CPUs that mix performance and efficiency cores (the M5
  has 4 + 6). So several folds are trained at once, each with fewer threads.

  Measured on the M5, 6 fits:

  | Model | 1 worker × 10 threads | Chosen plan | Speed-up |
  |---|---|---|---|
  | LightGBM | 43.5 s | 3 × 3: 28.6 s | 1.5× |
  | XGBoost | 46.7 s | 3 × 3: 29.3 s | 1.6× |
  | CatBoost | 107 s | 2 × 5: 97 s | 1.1× |
  | ExtraTrees | 65.8 s | 1 × 10 (unchanged) | — |

  Single-thread models (linear, SVM, kNN, Gaussian) run one fold per core. The predictions are
  bit-identical to sequential training.
- **GPU:** with CUDA, XGBoost and CatBoost train on the GPU. LightGBM always uses the CPU.
- **No Apple GPU (Metal / MLX) path exists for these models.** We checked:
  - XGBoost is built without GPU support on macOS and rejects `device="mps"`;
  - the LightGBM wheel has no GPU learner;
  - CatBoost's GPU mode is CUDA-only;
  - MLX is an array and neural-network framework with no tree learners.

  With 10k rows, GPU histogram building would not pay off anyway. The remaining cost is
  CPU-bound tree training.
- **Determinism:** results are identical across `fork`, `spawn` and single-process runs. All sets
  are sorted, so they no longer depend on hash randomization.

Tested with Python 3.13, scikit-learn 1.9, LightGBM 4.7, XGBoost 3.4, CatBoost 1.2.

## License

The code is released under the [MIT License](LICENSE). The competition data is not covered by
this license and is subject to the Kaggle competition rules.
