"""Final models: the stacking configurations behind the selected submissions.

Every configuration is a set of base models (battle-state + Monte Carlo + recursive features) and a
meta-learner on the logits of their out-of-fold predictions:

  stack10       the 10 base models kept by the cost/benefit analysis (models that make different
                errors: boosting with augmentation, ExtraTrees, Random Forest, linear), logistic meta
  stack22       all 22 base models of the zoo, logistic meta
  stack10_mean  the same 10 models as stack10, plain average of the logits (no learned weights)
  stack19       the 22 without svm, catboost and catboost_dense, logistic meta

Writes outputs/submission_<name>.csv for each configuration. With --strict it scores the
configurations on the OOF predictions of src/strict_cv.py instead, without writing submissions.

Usage: python src/final_model.py [--configs stack10,stack22,stack10_mean] [--strict] [--n-jobs N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
import stacking
from compute import configure, describe
from dataset import DATA, SEED
from features import load_jsonl

STACK10 = ["lgbm_aug", "xgb_aug", "lgbm_dense_aug", "catboost_aug", "catboost_dense_aug",
           "extratrees", "rf", "lda", "logreg", "logreg_dense"]
STACK22 = sorted(STACK10 + ["lgbm", "xgb", "catboost", "lgbm_dense", "catboost_dense", "lgbm_noise", "xgb_noise",
                            "svm", "knn", "qda", "gmm", "gmm_mix"])
CONFIGS = {
    "stack10": (STACK10, "logreg"),
    "stack22": (STACK22, "logreg"),
    "stack10_mean": (STACK10, "mean"),
    "stack19": ([m for m in STACK22 if m not in ("svm", "catboost", "catboost_dense")], "logreg"),
}
SELECTED = ["stack10", "stack22", "stack10_mean"]  # the 3 final submissions


def logits(d, names, key="oof"):
    return np.column_stack([logit(np.clip(np.load(os.path.join(d, f"{n}.npz"))[key], 1e-4, 1 - 1e-4))
                            for n in names])


def meta_cv(Z, y, meta, repeats=5):
    """Accuracy of the meta-learner with 5-fold CV repeated 5 times (seed 43): mean and sd over folds."""
    accs = []
    for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED + 1).split(Z, y):
        if meta == "mean":
            p = (Z[b].mean(1) > 0).astype(int)
        else:
            p = LogisticRegression(C=1.0, max_iter=2000).fit(Z[a], y[a]).predict(Z[b])
        accs.append((p == y[b]).mean())
    return np.mean(accs), np.std(accs)


def predict(Z, Zte, y, meta):
    if meta == "mean":
        return (Zte.mean(1) > 0).astype(int)
    return LogisticRegression(C=1.0, max_iter=2000).fit(Z, y).predict(Zte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(SELECTED))
    ap.add_argument("--strict", action="store_true", help="score on the OOF predictions of strict_cv.py")
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    configure(args.n_jobs, args.device)
    print(describe(), flush=True)
    configs = [c for c in args.configs.split(",") if c]
    if args.strict:
        from strict_cv import STRICT_DIR
        y = np.array([b["player_won"] for b in load_jsonl(os.path.join(DATA, "train.jsonl"))])
        for c in configs:
            names, meta = CONFIGS[c]
            acc, sd = meta_cv(logits(STRICT_DIR, names), y, meta)
            print(f"{c:14s} ({len(names)} models, meta {meta}): strict CV {acc:.4f} ± {sd:.4f}")
        return
    sets, y, test_ids = stacking.load_data()
    for name in sorted({m for c in configs for m in CONFIGS[c][0]}):
        stacking.run_oof(name, sets, y)
    for c in configs:
        names, meta = CONFIGS[c]
        Z, Zte = logits(stacking.OOF_DIR, names), logits(stacking.OOF_DIR, names, "test")
        acc, sd = meta_cv(Z, y, meta)
        pred = stacking.duplicate_override(test_ids, predict(Z, Zte, y, meta))
        out = os.path.join(stacking.OUT, f"submission_{c}.csv")
        pd.DataFrame({"battle_id": test_ids, "player_won": pred.astype(int)}).to_csv(out, index=False)
        print(f"{c:14s} ({len(names)} models, meta {meta}): CV {acc:.4f} ± {sd:.4f} -> {os.path.basename(out)}, "
              f"player 1 wins in {pred.mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
