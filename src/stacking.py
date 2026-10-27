"""Stacking: base models eterogenei (alberi, lineari, kernel, generativi) + meta-modello.

Le predizioni out-of-fold (OOF) di ogni base model sono calcolate con k-fold stratificato
e salvate in outputs/oof/<nome>.npz, così si possono aggiungere modelli senza ricalcolare tutto.

Uso:
    python src/stacking.py --models lgbm,xgb,gmm   # calcola (o riusa) le OOF
    python src/stacking.py --meta                  # valuta i meta-modelli e scrive la submission
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

N_FOLDS = 5    # sovrascrivibile con --folds
N_REPEATS = 1  # sovrascrivibile con --repeats (k-fold ripetuto con split diversi, OOF mediate)
OOF_DIR = os.path.join(OUT, "oof")


class GMMClassifier(BaseEstimator, ClassifierMixin):
    """Classificatore generativo: una GaussianMixture a covarianza piena per classe.

    La covarianza piena modella la distribuzione congiunta delle feature (non le
    tratta come indipendenti come farebbe Naive Bayes); P(y|x) via teorema di Bayes.
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
    """Proiezione supervisionata (PLS): poche direzioni massimamente correlate col target.

    A differenza della PCA non scarta le direzioni discriminative; i GMM a valle
    modellano la distribuzione congiunta in questo spazio ridotto.
    """

    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        self.pls_ = PLSRegression(self.n_components).fit(X, y)
        return self

    def transform(self, X):
        return self.pls_.transform(X)


class NoisyFit(BaseEstimator, ClassifierMixin):
    """Data augmentation: aggiunge copie del training set con rumore gaussiano.

    Il rumore è proporzionale alla deviazione standard di ogni feature continua
    (le binarie restano intatte); serve a regolarizzare i modelli che overfittano.
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


# regimi individuati dall'analisi degli errori OOF (src/analyze_errors.py): dove lo stack sbaglia di più
REGIMES = {
    "attrition": lambda X: ((X.p1_n_statused + X.p2_n_statused) >= 3) | ((X.p1_rest_sleep + X.p2_rest_sleep) >= 2),
    "balanced": lambda X: X.diff_hp_sum.abs() < 0.75,
    "setup": lambda X: (X.p1_setup_alive > 0) & (X.p2_setup_alive > 0),
}


class RegimeExpert(BaseEstimator, ClassifierMixin):
    """Esperto di un regime: stesso modello, ma gli esempi fuori dal regime pesano `w_out`."""

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


AUG_TURNS = (24, 26, 28)  # snapshot usati per l'augmentation (devono essere in train.SNAP_TURNS)
AUG_WEIGHT = 0.5  # Fase 8: 0.25 vince nella stima annidata a 3 modelli ma non nello stack completo (0.6967 vs 0.6981)


def load_data(extra=()):
    """Feature base + (opzionale) feature da simulazione Monte Carlo / modello ricorsivo.

    Restituisce anche gli snapshot dei turni AUG_TURNS (solo train) con le stesse colonne,
    usati dai modelli *_aug dentro i fold di training.
    """
    base, snap = get_features(snapshots=True)
    snap = snap[snap.snap_turn.isin(AUG_TURNS)]
    # bag calcolati insieme, così vocabolario e dummies coincidono
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


# factory dei modelli multi-thread: device e thread letti da compute al momento della creazione
def _lgb():
    return lgb.LGBMClassifier(**LGB_PARAMS, n_jobs=n_jobs())


def _xgb():
    return xgb.XGBClassifier(**XGB, **xgb_params())


def _cat():
    return CatBoostClassifier(**CAT, **catboost_params())


# nome -> (feature set, factory)
MODELS = {
    # alberi su tutte le feature
    "lgbm": ("full", _lgb),
    "xgb": ("full", _xgb),
    "catboost": ("full", _cat),
    "extratrees": ("full_fill", lambda: ExtraTreesClassifier(800, min_samples_leaf=3, max_features=0.2,
                                                             n_jobs=n_jobs(), random_state=SEED)),
    "rf": ("full_fill", lambda: RandomForestClassifier(800, min_samples_leaf=3, max_features=0.15,
                                                       n_jobs=n_jobs(), random_state=SEED)),
    # alberi senza bag (feature selection: solo le feature di stato)
    "lgbm_dense": ("dense", _lgb),
    "catboost_dense": ("dense", _cat),
    # lineari / kernel / vicinato, con pipeline di normalizzazione
    "logreg": ("full_fill", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=3000))),
    "logreg_dense": ("dense", lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                                    LogisticRegression(C=0.05, max_iter=3000))),
    "svm": ("dense", lambda: make_pipeline(StandardScaler(), SVC(C=0.5, gamma="scale", probability=True,
                                                                 random_state=SEED))),
    "knn": ("dense", lambda: make_pipeline(StandardScaler(), PCA(30, random_state=SEED),
                                           KNeighborsClassifier(150, weights="distance"))),
    # generativi gaussiani (distribuzione congiunta, covarianza piena)
    "lda": ("dense", lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
    "qda": ("dense", lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                           PCA(20, random_state=SEED), QuadraticDiscriminantAnalysis(reg_param=0.1))),
    "gmm": ("dense", lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                           PLSProjection(4), GMMClassifier(n_components=1, reg_covar=1e-2))),
    "gmm_mix": ("dense", lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED),
                                               PLSProjection(2), GMMClassifier(n_components=6, reg_covar=1e-2))),
    # versioni con rumore gaussiano (regolarizzazione)
    "lgbm_noise": ("dense", lambda: NoisyFit(_lgb(), sigma=0.1, n_copies=2)),
    "xgb_noise": ("full_fill", lambda: NoisyFit(_xgb(), sigma=0.1, n_copies=1)),
    # esperti condizionali (Fase 7): un LGBM per regime
    **{f"lgbm_exp_{r}": ("full", (lambda r=r: RegimeExpert(_lgb(), r))) for r in ("attrition", "balanced", "setup")},
}
# modelli single-thread: i loro fold girano in parallelo (joblib/loky, portabile su ogni OS)
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
    """OOF di un base model. I nomi *_aug usano il modello base con l'augmentation:
    agli esempi del fold di train si aggiungono i loro snapshot (peso AUG_WEIGHT);
    la validazione resta sul solo turno 30."""
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
        oof[va] += p_va / N_REPEATS  # con k-fold ripetuto, media delle OOF dei diversi split
        pte += p_te / len(splits)
    tr_acc = [r[2] for r in res]
    acc = ((oof > 0.5) == y).mean()
    print(f"{name:15s} OOF acc {acc:.4f} | train acc {np.mean(tr_acc):.4f} (gap {np.mean(tr_acc) - acc:+.3f}) "
          f"| {time.time() - t0:.0f}s", flush=True)
    np.savez(path, oof=oof, test=pte, train_acc=np.mean(tr_acc))
    return oof, pte


def meta_eval(names, y, n_repeats=5):
    """Valuta i meta-modelli sulle OOF con k-fold ripetuto (seed diverso dai base model)."""
    P = {n: np.load(os.path.join(OOF_DIR, f"{n}.npz")) for n in names}
    Z = np.column_stack([logit(np.clip(P[n]["oof"], 1e-4, 1 - 1e-4)) for n in names])
    Zte = np.column_stack([logit(np.clip(P[n]["test"], 1e-4, 1 - 1e-4)) for n in names])
    # validazione del meta fissa (5x5) per confrontare run con --folds/--repeats diversi
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
    print("pesi logreg_meta:", dict(zip(names, np.round(lr.coef_[0], 3))))
    return results, Z, Zte


def duplicate_override(test_ids, pred):
    """Battaglie di test identiche (team, lead e timeline) a una di train ricevono l'etichetta
    di quella: nelle 17 coppie duplicate interne al train l'etichetta coincide sempre."""
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
    print(f"{n} battaglie di test duplicate di battaglie di train: etichetta copiata")
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--no-submit", action="store_true", help="valuta il meta senza scrivere la submission")
    ap.add_argument("--use", default="", help="base model da usare nel meta (default: tutti quelli in cache)")
    ap.add_argument("--extra", default="", help="feature aggiuntive: mc,rec")
    ap.add_argument("--folds", type=int, default=5, help="numero di fold dei base model")
    ap.add_argument("--repeats", type=int, default=1, help="ripetizioni del k-fold con split diversi")
    ap.add_argument("--n-jobs", type=int, default=0, help="processi/thread (default: tutti i core)")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                    help="cuda: XGBoost e CatBoost su GPU NVIDIA (Linux/Windows)")
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
        print(f"{fname} scritta con {best} (CV {results[best][0]:.4f})")


if __name__ == "__main__":
    main()
