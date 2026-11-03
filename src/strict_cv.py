"""Strict cross-validation: everything learned from the data is re-learned inside each fold.

For each fold k (StratifiedKFold(5, seed 42), the same folds as stacking.py):
  - pokédex, Hidden Power types, co-occurrence and move statistics from the training fold only;
  - bag-of-words vocabulary from the training-fold rows only (turn 30 + snapshots);
  - Monte Carlo context and recursive transition model from the training fold only;
  - features for all training battles with this context: the validation battles are only
    transformed, exactly like the test battles for the submissions;
  - every base model is fitted on the training fold (with the training-fold snapshots) and predicts
    the validation fold.
OOF predictions go to outputs/oof_strict/<model>.npz; `final_model.py --strict` scores the configurations.
It takes about 35 minutes on a 10-core machine.

Usage: python src/strict_cv.py [--models a,b,...] [--n-jobs N]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
import recursive
import simulate
import stacking
from compute import configure, describe
from dataset import OUT, SEED, SNAP_TURNS, add_bags, build_reference, load_battles
from evaluate_nested import fit_predict
from features import build_frame

STRICT_DIR = os.path.join(OUT, "oof_strict")
N_SIMS, N_ROLLOUTS = 100, 30


def fold_frame(train, tr_idx, chart):
    """Frame of one fold: turn 30 of every training battle + snapshots of the training fold, with
    bags, Monte Carlo and recursive features learned from the training fold only."""
    ref = [train[i] for i in tr_idx]
    t0 = time.time()
    dex, ctx = build_reference(ref, chart)
    base = build_frame(train, dex, chart, ctx)
    base["is_test"], base["player_won"], base["snap_turn"] = 0, [b["player_won"] for b in train], 30
    snaps = []
    for t in SNAP_TURNS:
        s = build_frame(ref, dex, chart, ctx, truncate=t)
        s["is_test"], s["player_won"], s["snap_turn"] = 0, [b["player_won"] for b in ref], t
        snaps.append(s)
    df = pd.concat([base] + snaps, ignore_index=True)
    in_ref = np.isin(df.battle_id.values, [b["battle_id"] for b in ref])
    df = add_bags(df, fit_mask=in_ref)  # validation rows are only transformed, like the test set
    t_feat = time.time() - t0
    # Monte Carlo and recursive features on the same states
    states = train + [{**b, "battle_timeline": b["battle_timeline"][:t]} for t in SNAP_TURNS for b in ref]
    key = pd.DataFrame({"battle_id": [b["battle_id"] for b in states],
                        "snap_turn": [30] * len(train) + [t for t in SNAP_TURNS for _ in ref]})
    t0 = time.time()
    mc = simulate.simulate_many(states, N_SIMS, simulate.build_globals(ref))
    t_mc = time.time() - t0
    t0 = time.time()
    X, Y, _ = recursive.build_transitions(ref)
    models, residuals = recursive.fit_transition(X, Y, 100)
    rec = recursive.rollout(models, residuals, np.array([recursive.last_state(b) for b in states]), N_ROLLOUTS)
    t_rec = time.time() - t0
    extra = pd.concat([key, mc.reset_index(drop=True), rec.reset_index(drop=True)], axis=1)
    df = df.merge(extra, on=["battle_id", "snap_turn"], how="left")
    print(f"  context+features {t_feat:.0f}s, Monte Carlo {t_mc:.0f}s, recursive {t_rec:.0f}s", flush=True)
    return df


def main():
    from final_model import STACK22
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(STACK22))
    ap.add_argument("--n-jobs", type=int, default=0)
    args = ap.parse_args()
    configure(args.n_jobs)
    print(describe(), flush=True)
    names = [m for m in args.models.split(",") if m]
    train, _, chart = load_battles()
    y = np.array([b["player_won"] for b in train])
    oof = {n: np.full(len(y), np.nan) for n in names}
    os.makedirs(STRICT_DIR, exist_ok=True)
    t_all = time.time()
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)):
        print(f"fold {k}: {len(tr)} training, {len(va)} validation", flush=True)
        df = fold_frame(train, tr, chart)
        sets, y_sets, _ = stacking.make_sets(df)
        assert (sets["train_ids"] == np.array([b["battle_id"] for b in train])).all() and (y_sets == y).all()
        for n in names:
            t0 = time.time()
            oof[n][va] = fit_predict(n, sets, y, tr, va)
            print(f"  {n:20s} fold acc {((oof[n][va] > 0.5) == y[va]).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    for n in names:
        np.savez(os.path.join(STRICT_DIR, f"{n}.npz"), oof=oof[n])
        print(f"{n:20s} strict OOF acc {((oof[n] > 0.5) == y).mean():.4f}")
    print(f"total {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
