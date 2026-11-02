"""Stack heterogeneous base models (trees, linear, kernel and generative models).

Generate stratified k-fold OOF probabilities per base model and cache them in
outputs/oof/<name>.npz so new models can be added without recomputing everything.

Usage:
    python src/stacking.py --models lgbm,xgb,gmm   # compute or reuse OOF predictions
    python src/stacking.py --meta                  # evaluate meta-models and write a submission
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(__file__))
from compute import catboost_params, configure, describe, fold_map, fold_plan, n_jobs, xgb_params
from train import LGB_PARAMS, OUT, SEED, add_bags, get_features

N_FOLDS = 5    # Override with --folds
N_REPEATS = 1  # Override with --repeats: repeated k-fold with different splits, averaged OOF predictions
OOF_DIR = os.path.join(OUT, "oof")


class GMMClassifier(BaseEstimator, ClassifierMixin):
    """Generative classifier: one full-covariance GaussianMixture per class.

    Full covariance models the joint feature distribution rather than assuming
    independence as Naive Bayes does. Compute P(y|x) using Bayes' theorem.
    """

    def __init__(self, n_components=3, covariance_type="full", reg_covar=1e-3, random_state=SEED):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.gmms_ = [GaussianMixture(self.n_components, covariance_type=self.covariance_type,
                                      reg_covar=self.reg_covar, n_init=2, random_state=self.random_state)
                      .fit(X[y == c]) for c in self.classes_]
        self.log_prior_ = np.log(np.bincount(y) / len(y))
        return self

    def predict_proba(self, X):
        ll = np.column_stack([g.score_samples(X) for g in self.gmms_]) + self.log_prior_
        ll -= ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class PLSProjection(BaseEstimator, TransformerMixin):
    """Supervised PLS projection onto a few target-correlated directions.

    Unlike PCA, it retains discriminative directions; downstream GMMs model
    the joint distribution in this reduced space.
    """

    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        self.pls_ = PLSRegression(self.n_components).fit(X, y)
        return self

    def transform(self, X):
        return self.pls_.transform(X)


class NoisyFit(BaseEstimator, ClassifierMixin):
    """Augment training data with Gaussian-noise copies.

    Noise scales with each continuous feature's standard deviation; binary
    features remain unchanged. This regularizes models prone to overfitting.
    """

    def __init__(self, estimator, sigma=0.1, n_copies=1, random_state=SEED):
        self.estimator = estimator
        self.sigma = sigma
        self.n_copies = n_copies
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        std = np.nanstd(X, axis=0)
        cont = np.array([len(np.unique(c[~np.isnan(c)])) > 2 for c in X.T])
        scale = np.where(cont, std * self.sigma, 0.0)
        Xs = [X] + [X + rng.normal(size=X.shape) * scale for _ in range(self.n_copies)]
        self.estimator_ = clone(self.estimator).fit(np.vstack(Xs), np.tile(y, self.n_copies + 1))
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X, dtype=float))


# Regimes identified by OOF error analysis (src/analyze_errors.py): where the stack makes more errors
REGIMES = {
    "attrition": lambda X: ((X.p1_n_statused + X.p2_n_statused) >= 3) | ((X.p1_rest_sleep + X.p2_rest_sleep) >= 2),
    "balanced": lambda X: X.diff_hp_sum.abs() < 0.75,
    "setup": lambda X: (X.p1_setup_alive > 0) & (X.p2_setup_alive > 0),
}


class RegimeExpert(BaseEstimator, ClassifierMixin):
    """Regime expert: the same model, with out-of-regime rows weighted by `w_out`."""

    def __init__(self, estimator, regime, w_out=0.2):
        self.estimator = estimator
        self.regime = regime
        self.w_out = w_out

    def fit(self, X, y, sample_weight=None):
        w = np.where(REGIMES[self.regime](X).values, 1.0, self.w_out)
        if sample_weight is not None:
            w = w * sample_weight
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=w)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


EXTRA_FILES = {"mc": "mc_features.pkl", "rec": "rec_features.pkl", "strat": "strat_features.pkl",
               "lon": "lon_features.pkl"}


AUG_TURNS = (24, 26, 28)  # Augmentation snapshots; must be included in train.SNAP_TURNS.
AUG_WEIGHT = 0.5  # Phase 8: weight 0.25 wins in the three-model nested estimate, but not the full stack (0.6967 vs 0.6981).


def load_data(extra=()):
    """Base features plus optional Monte Carlo / recursive-model features.

    Also return train-only snapshots at AUG_TURNS with matching columns,
    used by *_aug models inside their training folds.
    """
    base, snap = get_features(snapshots=True)
    snap = snap[snap.snap_turn.isin(AUG_TURNS)]
    # Build bags jointly so vocabulary and dummy columns align.
    df = add_bags(pd.concat([base.assign(snap_turn=30), snap], ignore_index=True))
    for name in extra:
        ex = pd.read_pickle(os.path.join(OUT, EXTRA_FILES[name]))
        if "snap_turn" not in ex:
            ex = ex.assign(snap_turn=30)
        df = df.merge(ex, on=["battle_id", "is_test", "snap_turn"], how="left")
    df.columns = [c.replace(" ", "_") for c in df.columns]
    main = df.snap_turn.values == 30
    trm = main & (df.is_test.values == 0)
    tem = main & (df.is_test.values == 1)
    snm = ~main
    y = df.player_won.values[trm]
    feat = df.drop(columns=["battle_id", "is_test", "player_won", "snap_turn"])
    bag_cols = [c for c in feat.columns if "__" in c]
    full = feat.astype(float)
    dense = feat.drop(columns=bag_cols).astype(float).fillna(0)
    part = lambda X, m: X[m].reset_index(drop=True)
    return {
        "full": (part(full, trm), part(full, tem)),
        "dense": (part(dense, trm), part(dense, tem)),
        "snap_full": part(full, snm), "snap_dense": part(dense, snm),
        "train_ids": df.battle_id.values[trm], "snap_ids": df.battle_id.values[snm],
        "snap_y": df.player_won.values[snm],
    }, y, df.battle_id.values[tem]


XGB = dict(n_estimators=700, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.3,
           min_child_weight=5, reg_lambda=5.0, random_state=SEED)
CAT = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5, random_seed=SEED, verbose=0)


# Multithreaded model factories read device/thread settings from compute at creation time.
def _lgb():
    return lgb.LGBMClassifier(**LGB_PARAMS, n_jobs=n_jobs())


def _xgb():
    return xgb.XGBClassifier(**XGB, **xgb_params())


def _cat():
    return CatBoostClassifier(**CAT, **catboost_params())


# name -> (feature set, factory)
MODELS = {
    # Trees on all features
    "lgbm": ("full", _lgb),
    "xgb": ("full", _xgb),
    "catboost": ("full", _cat),
    "extratrees": ("full_fill", lambda: ExtraTreesClassifier(800, min_samples_leaf=3, max_features=0.2,
                                                             n_jobs=n_jobs(), random_state=SEED)),
    "rf": ("full_fill", lambda: RandomForestClassifier(800, min_samples_leaf=3, max_features=0.15,
                                                       n_jobs=n_jobs(), random_state=SEED)),
    # Trees without bags: state-feature selection
    "lgbm_dense": ("dense", _lgb),
    "catboost_dense": ("dense", _cat),
    # Linear / kernel / nearest-neighbor models with normalization pipelines
    "logreg": ("full_fill", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=3000))),
    "logreg_dense": ("dense", lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                                    LogisticRegression(C=0.05, max_iter=3000))),
    "svm": ("dense", lambda: make_pipeline(StandardScaler(), SVC(C=0.5, gamma="scale", probability=True,
                                                                 random_state=SEED))),
    "knn": ("dense", lambda: make_pipeline(StandardScaler(), PCA(30, random_state=SEED),
                                           KNeighborsClassifier(150, weights="distance"))),
    # Gaussian generative models of the joint distribution with full covariance
    "lda": ("dense", lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
    "qda": ("dense", lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                           PCA(20, random_state=SEED), QuadraticDiscriminantAnalysis(reg_param=0.1))),
    "gmm": ("dense", lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                           PLSProjection(4), GMMClassifier(n_components=1, reg_covar=1e-2))),
    "gmm_mix": ("dense", lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                               PLSProjection(2), GMMClassifier(n_components=6, reg_covar=1e-2))),
    # Gaussian-noise regularization variants
    "lgbm_noise": ("dense", lambda: NoisyFit(_lgb(), sigma=0.1, n_copies=2)),
    "xgb_noise": ("full_fill", lambda: NoisyFit(_xgb(), sigma=0.1, n_copies=1)),
    # Conditional experts (Phase 7): one LGBM per regime
    **{f"lgbm_exp_{r}": ("full", (lambda r=r: RegimeExpert(_lgb(), r))) for r in ("attrition", "balanced", "setup")},
}
# Single-threaded models run folds in parallel using portable joblib/loky workers.
PARALLEL_FOLDS = {"logreg", "logreg_dense", "svm", "knn", "lda", "qda", "gmm", "gmm_mix"}


def get_X(sets, kind):
    if kind == "full_fill":
        tr, te = sets["full"]
        return tr.fillna(0), te.fillna(0)
    return sets[kind]


def get_snap(sets, kind):
    if kind == "full_fill":
        return sets["snap_full"].fillna(0)
    return sets["snap_" + kind]


def run_oof(name, sets, y):
    """Generate base-model OOF probabilities. *_aug names add training snapshots
    from that fold's training battles with weight AUG_WEIGHT; validation uses
    only the turn-30 state."""
    path = os.path.join(OOF_DIR, f"{name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["oof"], d["test"]
    aug = name.endswith("_aug")
    base = name[:-4] if aug else name
    kind, make = MODELS[base]
    X, Xte = get_X(sets, kind)
    S = get_snap(sets, kind) if aug else None
    splits = [(r, tr, va) for r in range(N_REPEATS)
              for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(X, y)]

    def fit_fold(tr, va):
        if aug:
            sm = np.isin(sets["snap_ids"], sets["train_ids"][tr])
            Xa = pd.concat([X.iloc[tr], S[sm]], ignore_index=True)
            ya = np.concatenate([y[tr], sets["snap_y"][sm]])
            wa = np.concatenate([np.ones(len(tr)), np.full(sm.sum(), AUG_WEIGHT)])
            m = make().fit(Xa, ya, sample_weight=wa)
        else:
            m = make().fit(X.iloc[tr], y[tr])
        tr_acc = ((m.predict_proba(X.iloc[tr])[:, 1] > 0.5) == y[tr]).mean()
        return m.predict_proba(X.iloc[va])[:, 1], m.predict_proba(Xte)[:, 1], tr_acc

    t0 = time.time()
    res = fold_map(fit_fold, [(tr, va) for _, tr, va in splits], *fold_plan(base, base in PARALLEL_FOLDS))
    oof, pte = np.zeros(len(y)), np.zeros(len(Xte))
    for (_, _, va), (p_va, p_te, _) in zip(splits, res):
        oof[va] += p_va / N_REPEATS  # Repeated k-fold: average OOF predictions across splits.
        pte += p_te / len(splits)
    tr_acc = [r[2] for r in res]
    acc = ((oof > 0.5) == y).mean()
    print(f"{name:15s} OOF acc {acc:.4f} | train acc {np.mean(tr_acc):.4f} (gap {np.mean(tr_acc) - acc:+.3f}) "
          f"| {time.time() - t0:.0f}s", flush=True)
    np.savez(path, oof=oof, test=pte, train_acc=np.mean(tr_acc))
    return oof, pte


def meta_eval(names, y, n_repeats=5):
    """Evaluate meta-models on OOF values with repeated k-fold CV and a different seed."""
    P = {n: np.load(os.path.join(OOF_DIR, f"{n}.npz")) for n in names}
    Z = np.column_stack([logit(np.clip(P[n]["oof"], 1e-4, 1 - 1e-4)) for n in names])
    Zte = np.column_stack([logit(np.clip(P[n]["test"], 1e-4, 1 - 1e-4)) for n in names])
    # Fixed 5x5 meta-validation to compare runs with different --folds/--repeats
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED + 1)
    metas = {
        "mean": None,
        "logreg_meta": lambda: LogisticRegression(C=1.0, max_iter=2000),
        "logreg_meta_l1": lambda: LogisticRegression(C=0.05, penalty="l1", solver="liblinear"),
        "lgbm_meta": lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.02, num_leaves=4,
                                                min_child_samples=100, random_state=SEED, verbose=-1),
    }
    print("\nbase models (OOF acc):")
    for n in names:
        print(f"  {n:15s} {((P[n]['oof'] > 0.5) == y).mean():.4f}")
    results = {}
    for mname, make in metas.items():
        accs = []
        for tr, va in rskf.split(Z, y):
            if make is None:
                p = Z[va].mean(axis=1)
                accs.append(((p > 0) == y[va]).mean())
            else:
                m = make().fit(Z[tr], y[tr])
                accs.append((m.predict(Z[va]) == y[va]).mean())
        results[mname] = (np.mean(accs), np.std(accs))
        print(f"meta {mname:15s} acc {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
    print("logreg_meta weights:", dict(zip(names, np.round(lr.coef_[0], 3))))
    return results, Z, Zte


def duplicate_override(test_ids, pred):
    """Assign the known label to test battles identical to a training battle
    (team, lead and timeline). All 17 within-train duplicate pairs have matching labels."""
    import hashlib, json
    from features import load_jsonl
    from train import DATA
    key = lambda b: hashlib.md5(json.dumps([b["p1_team_details"], b["p2_lead_details"], b["battle_timeline"]],
                                           sort_keys=True).encode()).hexdigest()
    known = {key(b): b["player_won"] for b in load_jsonl(os.path.join(DATA, "train.jsonl"))}
    pos = {bid: i for i, bid in enumerate(test_ids)}
    pred = pred.copy()
    n = 0
    for b in load_jsonl(os.path.join(DATA, "test.jsonl")):
        k = key(b)
        if k in known:
            pred[pos[b["battle_id"]]] = known[k]
            n += 1
    print(f"{n} test battles duplicate training battles: known label copied")
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--no-submit", action="store_true", help="evaluate the meta-model without writing a submission")
    ap.add_argument("--use", default="", help="base models for the meta-model (default: all cached models)")
    ap.add_argument("--extra", default="", help="additional features: mc,rec")
    ap.add_argument("--folds", type=int, default=5, help="number of base-model folds")
    ap.add_argument("--repeats", type=int, default=1, help="k-fold repetitions with different splits")
    ap.add_argument("--n-jobs", type=int, default=0, help="processes/threads (default: all cores)")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                    help="cuda: XGBoost and CatBoost on NVIDIA GPU (Linux/Windows)")
    args = ap.parse_args()
    configure(args.n_jobs, args.device)
    print(describe(), flush=True)
    extra = [e for e in args.extra.split(",") if e]
    global OOF_DIR, N_FOLDS, N_REPEATS
    N_FOLDS, N_REPEATS = args.folds, args.repeats
    tag = "_".join(extra)
    if tag:
        OOF_DIR = os.path.join(OUT, f"oof_{tag}")
    if (N_FOLDS, N_REPEATS) != (5, 1):
        OOF_DIR += f"_k{N_FOLDS}r{N_REPEATS}"
    os.makedirs(OOF_DIR, exist_ok=True)
    sets, y, test_ids = load_data(extra)

    for name in [m for m in args.models.split(",") if m]:
        run_oof(name, sets, y)

    if args.meta:
        names = [m for m in args.use.split(",") if m] or \
                sorted(f[:-4] for f in os.listdir(OOF_DIR) if f.endswith(".npz"))
        results, Z, Zte = meta_eval(names, y)
        best = max((k for k in results if k != "mean"), key=lambda k: results[k][0])
        meta = {"logreg_meta": LogisticRegression(C=1.0, max_iter=2000),
                "logreg_meta_l1": LogisticRegression(C=0.05, penalty="l1", solver="liblinear"),
                "lgbm_meta": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.02, num_leaves=4,
                                                min_child_samples=100, random_state=SEED, verbose=-1)}[best]
        pred = duplicate_override(test_ids, meta.fit(Z, y).predict(Zte))
        if args.no_submit:
            return
        fname = f"submission_stacking{'_' + tag if tag else ''}.csv"
        pd.DataFrame({"battle_id": test_ids, "player_won": pred.astype(int)}).to_csv(
            os.path.join(OUT, fname), index=False)
        print(f"{fname} written with {best} (CV {results[best][0]:.4f})")


if __name__ == "__main__":
    main()
