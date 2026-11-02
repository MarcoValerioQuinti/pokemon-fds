"""Final model: the stack reduced to useful components (reports/02_report_costi_benefici).

Keep diverse base models (augmented boosting, ExtraTrees, RF and linear models) and remove
redundant or unhelpful ones (svm, knn, qda, GMM, noise variants, non-augmented boosting,
regime experts) and strategic/longitudinal features. Retain Monte Carlo and recursive
features (--extra mc,rec, default): also computed for augmented snapshots, they add
0.44 percentage points (final stack 0.6994 with mc,rec vs 0.6937 without; McNemar p = 0.11).

Compare the final stack with the reference (22 models + mc,rec) on the same battles
(paired difference, bootstrap CI and McNemar) and write outputs/submission_final.csv.

Usage: python src/final_model.py [--extra mc,rec] [--n-jobs N] [--device auto|cpu|cuda]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
import stacking
from compute import configure, describe
from train import OUT, SEED

FINAL_MODELS = ["lgbm_aug", "xgb_aug", "lgbm_dense_aug", "catboost_aug", "catboost_dense_aug",
                "extratrees", "rf", "lda", "logreg", "logreg_dense"]
REFERENCE = os.path.join(OUT, "oof_mc_rec")  # 22-model stack: previous best reference


def logits(d, names, key="oof"):
    return np.column_stack([logit(np.clip(np.load(os.path.join(d, f"{n}.npz"))[key], 1e-4, 1 - 1e-4))
                            for n in names])


def meta_oof(Z, y, repeats=5):
    """Evaluate the logistic meta-model on logits with repeated 5-fold CV (seed 43,
    as in stacking.meta_eval); average repeated OOF predictions for paired comparisons."""
    accs, pm = [], np.zeros(len(y))
    for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED + 1).split(Z, y):
        p = LogisticRegression(C=1.0, max_iter=2000).fit(Z[a], y[a]).predict_proba(Z[b])[:, 1]
        pm[b] += p / repeats
        accs.append(((p > 0.5) == y[b]).mean())
    return np.mean(accs), np.std(accs), pm


def paired(ok_ref, ok_new, n_boot=5000, seed=0):
    d = ok_new.astype(float) - ok_ref.astype(float)
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(1)
    b, c = int((ok_ref & ~ok_new).sum()), int((~ok_ref & ok_new).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="mc,rec", help="additional features; \"\" for none (-0.44 points; see docstring)")
    ap.add_argument("--out", default="submission_final.csv")
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    configure(args.n_jobs, args.device)
    print(describe(), flush=True)
    extra = [e for e in args.extra.split(",") if e]
    # Reuse the stacking.py cache: identical model/features produce identical OOF values.
    stacking.OOF_DIR = os.path.join(OUT, "oof_" + "_".join(extra) if extra else "oof_final")
    os.makedirs(stacking.OOF_DIR, exist_ok=True)
    sets, y, test_ids = stacking.load_data(extra)
    for name in FINAL_MODELS:
        stacking.run_oof(name, sets, y)

    Z, Zte = logits(stacking.OOF_DIR, FINAL_MODELS), logits(stacking.OOF_DIR, FINAL_MODELS, "test")
    acc, sd, pm = meta_oof(Z, y)
    print(f"\nfinal stack ({len(FINAL_MODELS)} models, extra={extra or 'none'}): meta logreg {acc:.4f} ± {sd:.4f}")
    # Compare with the 22-model stack only if cached, not on a fresh environment such as Kaggle.
    ref_names = sorted(f[:-4] for f in os.listdir(REFERENCE) if f.endswith(".npz")) if os.path.isdir(REFERENCE) else []
    if set(ref_names) > set(FINAL_MODELS):
        ref_acc, ref_sd, pr = meta_oof(logits(REFERENCE, ref_names), y)
        d, lo, hi, p = paired((pr > 0.5) == y, (pm > 0.5) == y)
        print(f"reference ({len(ref_names)} models, mc,rec): {ref_acc:.4f} ± {ref_sd:.4f}")
        print(f"final - reference difference: {d * 100:+.2f} points, 95% CI [{lo * 100:+.2f}, {hi * 100:+.2f}], McNemar p = {p:.3f}")

    meta = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
    print("meta-model weights:", dict(zip(FINAL_MODELS, np.round(meta.coef_[0], 3))))
    pred = stacking.duplicate_override(test_ids, meta.predict(Zte))
    pd.DataFrame({"battle_id": test_ids, "player_won": pred.astype(int)}).to_csv(os.path.join(OUT, args.out), index=False)
    print(f"{args.out} written: {len(pred)} rows, p1 win rate {pred.mean():.1%}")


if __name__ == "__main__":
    main()
