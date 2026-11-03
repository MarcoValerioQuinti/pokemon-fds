"""Nested cross-validation of the whole stacking procedure.

For each outer fold (5, stratified, seed 42):
  1. inner 5-fold OOF predictions of the base models on the outer training part only;
  2. a logistic-regression meta-learner (chosen beforehand) fitted on those inner OOF predictions;
  3. the base models refitted on the whole outer training part predict the outer fold;
  4. the meta-learner combines them: one outer prediction per battle.
The outer-fold labels never touch the base models, the meta-learner or the threshold (0.5).
Battles with identical timelines always end up in the same fold, outer and inner.

Usage:
    python src/evaluate_nested.py --name final --models lgbm_aug,xgb_aug,...
    python src/evaluate_nested.py --compare A B   # paired accuracy difference with bootstrap CI
Outputs in outputs/experiments/<name>/: predictions.csv, metrics.json, config.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from compute import configure, describe, fold_map, fold_plan
from dataset import DATA, OUT, SEED
from features import load_jsonl
from stacking import PARALLEL_FOLDS, fit_model, get_X, load_data, MODELS

EXP = os.path.join(OUT, "experiments")
N_OUTER = N_INNER = 5


def duplicate_groups(battles):
    """Group id per battle: battles with an identical timeline share the group."""
    first = {}
    groups = []
    for i, b in enumerate(battles):
        h = hashlib.md5(json.dumps(b["battle_timeline"], sort_keys=True).encode()).hexdigest()
        groups.append(first.setdefault(h, i))
    return np.array(groups)


def make_folds(y, groups, n, seed):
    """Standard StratifiedKFold; then duplicates follow the fold of the first battle of their group."""
    fold = np.empty(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        fold[va] = k
    leader = {}
    for i, g in enumerate(groups):
        fold[i] = leader.setdefault(g, fold[i])
    return fold


def fit_predict(name, sets, y, tr, ev):
    """Fits base model `name` on the training rows tr and predicts the rows ev."""
    base = name[:-4] if name.endswith("_aug") else name
    X = get_X(sets, MODELS[base][0])[0]
    return fit_model(name, sets, y, tr).predict_proba(X.iloc[ev])[:, 1]


def run_model(name, sets, y, outer, inner):
    """Inner OOF predictions (for the meta-learner) and outer predictions of one base model."""
    jobs = []  # (outer fold, training rows, rows to predict, kind)
    for k in range(N_OUTER):
        tr_o = np.where(outer != k)[0]
        jobs.append((k, tr_o, np.where(outer == k)[0], "outer"))
        for j in range(N_INNER):
            jobs.append((k, tr_o[inner[k][tr_o] != j], tr_o[inner[k][tr_o] == j], "inner"))
    base = name[:-4] if name.endswith("_aug") else name
    preds = fold_map(fit_predict, [(name, sets, y, tr, ev) for _, tr, ev, _ in jobs],
                     *fold_plan(base, base in PARALLEL_FOLDS))
    inner_oof = np.full((N_OUTER, len(y)), np.nan)
    outer_pred = np.zeros(len(y))
    for (k, _, ev, kind), p in zip(jobs, preds):
        if kind == "outer":
            outer_pred[ev] = p
        else:
            inner_oof[k, ev] = p
    return inner_oof, outer_pred


def _z(p):
    return logit(np.clip(p, 1e-4, 1 - 1e-4))


def metrics(y, p, fold):
    acc = ((p > 0.5) == y)
    return {"acc": acc.mean(), "auc": roc_auc_score(y, p), "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
            "acc_per_fold": [acc[fold == k].mean() for k in range(N_OUTER)]}


def evaluate(name, models):
    out_dir = os.path.join(EXP, name)
    cache = os.path.join(out_dir, "models")
    os.makedirs(cache, exist_ok=True)
    sets, y, _ = load_data()
    train = load_jsonl(os.path.join(DATA, "train.jsonl"))
    assert [b["battle_id"] for b in train] == list(sets["train_ids"])
    groups = duplicate_groups(train)
    outer = make_folds(y, groups, N_OUTER, SEED)
    inner = {}
    for k in range(N_OUTER):
        tr_o = np.where(outer != k)[0]
        f = np.full(len(y), -1)
        f[tr_o] = make_folds(y[tr_o], groups[tr_o], N_INNER, SEED + 100 + k)
        inner[k] = f
    print(f"{len(set(groups))} groups over {len(y)} battles (duplicates kept in the same fold)", flush=True)

    inner_oof, outer_pred = {}, {}
    for m in models:
        path = os.path.join(cache, f"{m}.npz")
        t0 = time.time()
        if os.path.exists(path):
            d = np.load(path)
            inner_oof[m], outer_pred[m] = d["inner"], d["outer"]
        else:
            inner_oof[m], outer_pred[m] = run_model(m, sets, y, outer, inner)
            np.savez(path, inner=inner_oof[m], outer=outer_pred[m])
        print(f"{m:20s} outer acc {((outer_pred[m] > 0.5) == y).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)

    stack, mean = np.zeros(len(y)), np.zeros(len(y))
    for k in range(N_OUTER):
        tr_o, va_o = outer != k, outer == k
        Z = np.column_stack([_z(inner_oof[m][k, tr_o]) for m in models])
        Zv = np.column_stack([_z(outer_pred[m][va_o]) for m in models])
        stack[va_o] = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y[tr_o]).predict_proba(Zv)[:, 1]
        mean[va_o] = 1 / (1 + np.exp(-Zv.mean(axis=1)))

    pred = pd.DataFrame({"battle_id": sets["train_ids"], "y_true": y, "fold": outer})
    for m in models:
        pred[f"p_{m}"] = outer_pred[m]
    pred["p_mean"], pred["p_stack"], pred["pred"] = mean, stack, (stack > 0.5).astype(int)
    pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    res = {"stack": metrics(y, stack, outer), "mean": metrics(y, mean, outer),
           "base": {m: metrics(y, outer_pred[m], outer) for m in models}}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"models": models, "seed": SEED, "outer": N_OUTER, "inner": N_INNER,
                   "meta": "logistic regression C=1 on logits", "threshold": 0.5}, f, indent=1)
    s = res["stack"]
    print(f"\nNESTED stack (logreg) acc {s['acc']:.4f} | AUC {s['auc']:.4f} | logloss {s['logloss']:.4f}")
    print("  per fold:", " ".join(f"{a:.4f}" for a in s["acc_per_fold"]))
    print(f"NESTED mean of logits acc {res['mean']['acc']:.4f}")
    return res


def compare(a, b, n_boot=5000):
    """Paired accuracy difference (b - a) on the same battles, with a 95% bootstrap CI."""
    A = pd.read_csv(os.path.join(EXP, a, "predictions.csv"))
    B = pd.read_csv(os.path.join(EXP, b, "predictions.csv"))
    assert (A.battle_id.values == B.battle_id.values).all()
    ca = (A.pred.values == A.y_true.values).astype(float)
    cb = (B.pred.values == B.y_true.values).astype(float)
    d = cb - ca
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    per_fold = [d[A.fold.values == k].mean() for k in range(N_OUTER)]
    print(f"{a}: {ca.mean():.4f}  {b}: {cb.mean():.4f}  diff {d.mean():+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  "
          f"P(diff>0) {np.mean(boot > 0):.2f}")
    print("  diff per fold:", " ".join(f"{x:+.4f}" for x in per_fold))
    return d.mean(), (lo, hi), per_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--models", default="logreg,lgbm,xgb,catboost")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    configure(args.n_jobs, args.device)
    print(describe(), flush=True)
    evaluate(args.name, [m for m in args.models.split(",") if m])


if __name__ == "__main__":
    main()
