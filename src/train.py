"""Training + CV dei modelli e generazione della submission.

Uso: python src/train.py [--tune N_TRIALS]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(__file__))
from compute import catboost_params, configure, describe, xgb_params
from features import build_context, build_frame, build_pokedex, clean_battles, load_jsonl, load_type_chart

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
SEED = 42
N_FOLDS = 5


SNAP_TURNS = (20, 22, 24, 26, 28)
# Pulizia delle mosse persistenti (features.clean_battles): corretta come meccanica di gioco ma
# peggiora la CV di ~0.4 punti (le mosse "ripetute" portano segnale), quindi disattivata.
CLEAN_MOVES = False


def _load_clean():
    prep = clean_battles if CLEAN_MOVES else (lambda x: x)
    train = prep(load_jsonl(os.path.join(DATA, "train.jsonl")))
    test = prep(load_jsonl(os.path.join(DATA, "test.jsonl")))
    chart = load_type_chart(os.path.join(DATA, "type_chart_gen2.csv"))
    dex = build_pokedex(train + test)
    return train, test, dex, chart, build_context(train + test, dex, chart)


def get_features(snapshots=False):
    """Feature al turno 30 (train+test). Con snapshots=True anche le feature dei soli
    battle di train troncati ai turni SNAP_TURNS, per l'augmentation (colonna snap_turn)."""
    os.makedirs(OUT, exist_ok=True)  # outputs/ non è versionata: su un clone pulito non esiste
    cache = os.path.join(OUT, "features.pkl")
    snap_cache = os.path.join(OUT, "features_snap.pkl")
    if os.path.exists(cache) and (not snapshots or os.path.exists(snap_cache)):
        df = pd.read_pickle(cache)
        return (df, pd.read_pickle(snap_cache)) if snapshots else df
    train, test, dex, chart, ctx = _load_clean()
    df = build_frame(train + test, dex, chart, ctx)
    df["is_test"] = [0] * len(train) + [1] * len(test)
    df["player_won"] = [b["player_won"] for b in train] + [-1] * len(test)
    df.to_pickle(cache)
    if not snapshots:
        return df
    snaps = []
    for t in SNAP_TURNS:
        s = build_frame(train, dex, chart, ctx, truncate=t)
        s["is_test"], s["player_won"], s["snap_turn"] = 0, [b["player_won"] for b in train], t
        snaps.append(s)
    snap = pd.concat(snaps, ignore_index=True)
    snap.to_pickle(snap_cache)
    return df, snap


def add_bags(df, min_df=20):
    """Bag-of-pokemon / bag-of-moves come colonne di conteggio."""
    parts = [df]
    for col in [c for c in df.columns if c.startswith("_")]:
        cv = CountVectorizer(token_pattern=r"\S+", min_df=min_df)
        m = cv.fit_transform(df[col]).toarray()
        parts.append(pd.DataFrame(m, columns=[f"{col[1:]}__{v}" for v in cv.get_feature_names_out()], index=df.index))
    out = pd.concat(parts, axis=1).drop(columns=[c for c in df.columns if c.startswith("_")])
    for c in ["p1_active_status", "p2_active_status"]:
        out = pd.concat([out.drop(columns=c), pd.get_dummies(out[c], prefix=c, dtype=int)], axis=1)
    return out


def cv_model(name, make, X, y, Xte):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    pte = np.zeros(len(Xte))
    for tr, va in skf.split(X, y):
        m = make()
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pte += m.predict_proba(Xte)[:, 1] / N_FOLDS
    acc = ((oof > 0.5) == y).mean()
    print(f"{name:10s} CV acc = {acc:.4f}", flush=True)
    return oof, pte, acc


LGB_PARAMS = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=5.0,
                  random_state=SEED, verbose=-1)


def tune_lgb(X, y, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 200, 1500),
                 learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 4, 64),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.05, 0.8),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                 random_state=SEED, verbose=-1)
        accs = []
        for tr, va in skf.split(X, y):
            m = lgb.LGBMClassifier(**p).fit(X.iloc[tr], y[tr])
            accs.append(((m.predict_proba(X.iloc[va])[:, 1] > 0.5) == y[va]).mean())
        return np.mean(accs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, callbacks=[
        lambda s, t: print(f"trial {t.number}: {t.value:.4f} (best {s.best_value:.4f})", flush=True)])
    print("best tuned lgb", study.best_value, study.best_params)
    return {**study.best_params, "subsample_freq": 1, "random_state": SEED, "verbose": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=0, help="processi/thread (default: tutti i core)")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    configure(args.n_jobs, args.device)
    print(describe())
    os.makedirs(OUT, exist_ok=True)

    df = add_bags(get_features())
    tr, te = df[df.is_test == 0], df[df.is_test == 1]
    y = tr.player_won.values
    drop = ["battle_id", "is_test", "player_won"]
    X, Xte = tr.drop(columns=drop), te.drop(columns=drop)
    X.columns = Xte.columns = [c.replace(" ", "_") for c in X.columns]
    print("shape", X.shape, flush=True)

    lgb_params = tune_lgb(X, y, args.tune) if args.tune else LGB_PARAMS
    models = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=3000)),
        "lgbm": lambda: lgb.LGBMClassifier(**lgb_params),
        "xgb": lambda: xgb.XGBClassifier(n_estimators=700, learning_rate=0.02, max_depth=4, subsample=0.8,
                                         colsample_bytree=0.3, min_child_weight=5, reg_lambda=5.0,
                                         random_state=SEED, **xgb_params()),
        "catboost": lambda: CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                                               random_seed=SEED, verbose=0, **catboost_params()),
    }
    Xf, Xtef = X.fillna(0), Xte.fillna(0)
    res = {}
    for name, make in models.items():
        a, b = (Xf, Xtef) if name == "logreg" else (X, Xte)
        res[name] = cv_model(name, make, a, y, b)

    # ensemble: media delle probabilità, e media dei migliori
    names = sorted(res, key=lambda n: -res[n][2])
    best_combo, best_acc = None, 0
    for k in range(1, len(names) + 1):
        oof = np.mean([res[n][0] for n in names[:k]], axis=0)
        acc = ((oof > 0.5) == y).mean()
        print(f"ensemble top{k} {names[:k]} CV acc = {acc:.4f}")
        if acc > best_acc:
            best_combo, best_acc = names[:k], acc
    pte = np.mean([res[n][1] for n in best_combo], axis=0)
    sub = pd.DataFrame({"battle_id": te.battle_id.values, "player_won": (pte > 0.5).astype(int)})
    sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
    print(f"submission salvata ({best_combo}, CV {best_acc:.4f}) -> outputs/submission.csv")

    # importanza feature (lgbm full fit)
    m = lgb.LGBMClassifier(**lgb_params).fit(X, y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(OUT, "feature_importance.csv"))
    print(imp.head(25))


if __name__ == "__main__":
    main()
