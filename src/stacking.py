"""Base model zoo and out-of-fold (OOF) predictions for the stacking.

Each base model gets out-of-fold predictions on the training set and averaged test predictions from
a stratified 5-fold split (seed 42), cached in outputs/oof_mc_rec/<name>.npz. Models named <base>_aug
are trained with snapshot augmentation: the training battles of the fold, truncated at turns 24/26/28,
are added with weight 0.5 (validation always uses the full 30-turn state).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(__file__))
from compute import catboost_params, fold_map, fold_plan, n_jobs, xgb_params
from dataset import DATA, OUT, SEED, add_bags, get_features
from features import load_jsonl

N_FOLDS = 5
OOF_DIR = os.path.join(OUT, "oof_mc_rec")
EXTRA_FILES = {"mc": "mc_features.pkl", "rec": "rec_features.pkl"}
AUG_WEIGHT = 0.5


class GMMClassifier(BaseEstimator, ClassifierMixin):
    """Generative classifier: one full-covariance Gaussian mixture per class, P(y|x) by Bayes' rule.
    The full covariance models the joint distribution of the features (unlike Naive Bayes)."""

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
    """Supervised projection (PLS) on a few directions maximally correlated with the target. Unlike
    PCA it keeps the discriminative directions; the Gaussian mixtures model this reduced space."""

    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        self.pls_ = PLSRegression(self.n_components).fit(X, y)
        return self

    def transform(self, X):
        return self.pls_.transform(X)


class NoisyFit(BaseEstimator, ClassifierMixin):
    """Regularization by augmentation: adds copies of the training set with Gaussian noise
    proportional to the standard deviation of each continuous feature (binary ones stay intact)."""

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


def load_data(extra=("mc", "rec")):
    """Battle-state features + bag-of-words + the extra feature files, as model feature sets."""
    # bag-of-words vocabulary learned from the training rows (turn 30 and snapshots) only
    df = add_bags(get_features())
    for name in extra:
        ex = pd.read_pickle(os.path.join(OUT, EXTRA_FILES[name]))
        df = df.merge(ex, on=["battle_id", "is_test", "snap_turn"], how="left")
    return make_sets(df)


def make_sets(df):
    """From one frame (turn 30 of train and test + training snapshots, with bags and extras) to the
    model feature sets: full (with bags), dense (no bags, NaN -> 0) and their snapshot versions."""
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


LGB_PARAMS = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=30, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0, random_state=SEED, verbose=-1)
XGB = dict(n_estimators=700, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.3,
           min_child_weight=5, reg_lambda=5.0, random_state=SEED)
CAT = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5, random_seed=SEED, verbose=0)


# factories of the multi-threaded models: device and threads are read from compute when created
def _lgb():
    return lgb.LGBMClassifier(**LGB_PARAMS, n_jobs=n_jobs())


def _xgb():
    return xgb.XGBClassifier(**XGB, **xgb_params())


def _cat():
    return CatBoostClassifier(**CAT, **catboost_params())


def _quantile():
    return QuantileTransformer(output_distribution="normal", random_state=SEED)


# name -> (feature set, factory); every scaler/projection is fitted inside the training fold
MODELS = {
    # trees on all the features
    "lgbm": ("full", _lgb),
    "xgb": ("full", _xgb),
    "catboost": ("full", _cat),
    "extratrees": ("full_fill", lambda: ExtraTreesClassifier(800, min_samples_leaf=3, max_features=0.2,
                                                             n_jobs=n_jobs(), random_state=SEED)),
    "rf": ("full_fill", lambda: RandomForestClassifier(800, min_samples_leaf=3, max_features=0.15,
                                                       n_jobs=n_jobs(), random_state=SEED)),
    # trees on the dense battle-state features only (no bag-of-words)
    "lgbm_dense": ("dense", _lgb),
    "catboost_dense": ("dense", _cat),
    # linear, kernel and neighbour models, with normalization pipelines
    "logreg": ("full_fill", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=3000))),
    "logreg_dense": ("dense", lambda: make_pipeline(_quantile(), LogisticRegression(C=0.05, max_iter=3000))),
    "svm": ("dense", lambda: make_pipeline(StandardScaler(), SVC(C=0.5, gamma="scale", probability=True,
                                                                 random_state=SEED))),
    "knn": ("dense", lambda: make_pipeline(StandardScaler(), PCA(30, random_state=SEED),
                                           KNeighborsClassifier(150, weights="distance"))),
    # Gaussian generative models (joint distribution, full covariance)
    "lda": ("dense", lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
    "qda": ("dense", lambda: make_pipeline(_quantile(), PCA(20, random_state=SEED),
                                           QuadraticDiscriminantAnalysis(reg_param=0.1))),
    "gmm": ("dense", lambda: make_pipeline(_quantile(), PLSProjection(4), GMMClassifier(n_components=1, reg_covar=1e-2))),
    "gmm_mix": ("dense", lambda: make_pipeline(_quantile(), PLSProjection(2), GMMClassifier(n_components=6, reg_covar=1e-2))),
    # Gaussian-noise variants (regularization)
    "lgbm_noise": ("dense", lambda: NoisyFit(_lgb(), sigma=0.1, n_copies=2)),
    "xgb_noise": ("full_fill", lambda: NoisyFit(_xgb(), sigma=0.1, n_copies=1)),
}
# single-threaded models: their folds run in parallel (joblib/loky, portable)
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


def fit_model(name, sets, y, tr):
    """Fits base model `name` on the training rows tr (plus their snapshots for *_aug models)."""
    aug = name.endswith("_aug")
    base = name[:-4] if aug else name
    kind, make = MODELS[base]
    X = get_X(sets, kind)[0]
    if not aug:
        return make().fit(X.iloc[tr], y[tr])
    S = get_snap(sets, kind)
    sm = np.isin(sets["snap_ids"], sets["train_ids"][tr])  # snapshots of the training battles only
    Xa = pd.concat([X.iloc[tr], S[sm]], ignore_index=True)
    ya = np.concatenate([y[tr], sets["snap_y"][sm]])
    wa = np.concatenate([np.ones(len(tr)), np.full(sm.sum(), AUG_WEIGHT)])
    return make().fit(Xa, ya, sample_weight=wa)


def run_oof(name, sets, y):
    """OOF and averaged test predictions of a base model, cached in OOF_DIR."""
    path = os.path.join(OOF_DIR, f"{name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["oof"], d["test"]
    base = name[:-4] if name.endswith("_aug") else name
    X, Xte = get_X(sets, MODELS[base][0])
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))

    def fit_fold(tr, va):
        m = fit_model(name, sets, y, tr)
        tr_acc = ((m.predict_proba(X.iloc[tr])[:, 1] > 0.5) == y[tr]).mean()
        return m.predict_proba(X.iloc[va])[:, 1], m.predict_proba(Xte)[:, 1], tr_acc

    t0 = time.time()
    res = fold_map(fit_fold, splits, *fold_plan(base, base in PARALLEL_FOLDS))
    oof, pte = np.zeros(len(y)), np.zeros(len(Xte))
    for (_, va), (p_va, p_te, _) in zip(splits, res):
        oof[va] = p_va
        pte += p_te / len(splits)
    tr_acc = np.mean([r[2] for r in res])
    acc = ((oof > 0.5) == y).mean()
    print(f"{name:20s} OOF acc {acc:.4f} | train acc {tr_acc:.4f} (gap {tr_acc - acc:+.3f}) | {time.time() - t0:.0f}s",
          flush=True)
    os.makedirs(OOF_DIR, exist_ok=True)
    np.savez(path, oof=oof, test=pte, train_acc=tr_acc)
    return oof, pte


def duplicate_override(test_ids, pred):
    """Test battles identical to a training battle (same team, lead and timeline) get its label:
    the 17 duplicate pairs inside the training set always share the label."""
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
    print(f"{n} test battles identical to a training battle: label copied")
    return pred
