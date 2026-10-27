"""Valutazione annidata (nested CV) dell'intera pipeline di stacking.

Per ogni fold esterno (5, stratificato, seed 42):
  1. i fold interni (5) sul solo training esterno danno le OOF interne dei base model;
  2. il meta-modello (scelto a priori: logreg sui logit) si allena su quelle OOF;
  3. i base model, riaddestrati su tutto il training esterno, predicono il fold esterno;
  4. il meta combina queste predizioni: una sola predizione esterna per battaglia.
Le etichette del fold esterno non toccano né i base model, né il meta, né la soglia (0.5).

I duplicati esatti (stessa timeline) finiscono sempre nello stesso fold, sia esterno sia interno.

Uso:
    python src/evaluate_nested.py --name base4 --models logreg,lgbm,xgb,catboost --extra mc,rec
    python src/evaluate_nested.py --compare base4 altro   # differenza appaiata + bootstrap
Output in outputs/experiments/<name>/: predictions.csv, metrics.json, config.json.
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
from features import load_jsonl
import stacking
from stacking import MODELS, PARALLEL_FOLDS, REGIMES, get_snap, get_X, load_data
from train import DATA, OUT, SEED

EXP = os.path.join(OUT, "experiments")
N_OUTER = N_INNER = 5


def duplicate_groups(battles):
    """Gruppo per battaglia: battaglie con timeline identica condividono il gruppo."""
    first = {}
    groups = []
    for i, b in enumerate(battles):
        h = hashlib.md5(json.dumps(b["battle_timeline"], sort_keys=True).encode()).hexdigest()
        groups.append(first.setdefault(h, i))
    return np.array(groups)


def make_folds(y, groups, n, seed):
    """StratifiedKFold standard; poi i duplicati seguono il fold del primo del gruppo."""
    fold = np.empty(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        fold[va] = k
    leader = {}
    for i, g in enumerate(groups):
        fold[i] = leader.setdefault(g, fold[i])
    return fold


def fit_predict(name, sets, y, tr, ev):
    """Allena il base model `name` sulle righe tr del train (con snapshot se *_aug) e predice ev."""
    aug = name.endswith("_aug")
    base = name[:-4] if aug else name
    kind, make = MODELS[base]
    X, _ = get_X(sets, kind)
    if aug:
        S = get_snap(sets, kind)
        sm = np.isin(sets["snap_ids"], sets["train_ids"][tr])  # snapshot dei soli battle di training
        Xa = pd.concat([X.iloc[tr], S[sm]], ignore_index=True)
        ya = np.concatenate([y[tr], sets["snap_y"][sm]])
        wa = np.concatenate([np.ones(len(tr)), np.full(sm.sum(), stacking.AUG_WEIGHT)])
        m = make().fit(Xa, ya, sample_weight=wa)
    else:
        m = make().fit(X.iloc[tr], y[tr])
    return m.predict_proba(X.iloc[ev])[:, 1]


def run_model(name, sets, y, outer, inner):
    """Per un base model: OOF interne (per il meta) e predizioni esterne."""
    jobs = []  # (fold esterno, indici train, indici da predire, tipo)
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


def select_columns(sets, keep=(), drop=()):
    """Restringe le feature: solo le colonne con prefisso in keep (se dato), senza quelle in drop."""
    def f(X):
        cols = [c for c in X.columns if (not keep or c.startswith(tuple(keep))) and not c.startswith(tuple(drop))]
        return X[cols]
    for k in ("full", "dense"):
        sets[k] = tuple(f(X) for X in sets[k])
    for k in ("snap_full", "snap_dense"):
        sets[k] = f(sets[k])
    return sets


def meta_inputs(models, P, regimes, R):
    """Logit dei base model; con il gating, anche gli indicatori di regime e le interazioni
    regime x logit dell'esperto corrispondente (il meta impara quanto fidarsi dell'esperto)."""
    Z = [_z(P[m]) for m in models]
    for r in regimes:
        Z.append(R[r])
        for m in models:
            if m.endswith(f"_exp_{r}"):
                Z.append(R[r] * _z(P[m]))
    return np.column_stack(Z)


def evaluate(name, models, extra, keep=(), drop=(), gate=()):
    out_dir = os.path.join(EXP, name)
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "models")
    os.makedirs(cache, exist_ok=True)
    sets, y, _ = load_data(extra)
    if keep or drop:
        sets = select_columns(sets, keep, drop)
        print(f"feature selezionate: {sets['full'][0].shape[1]} (keep={keep}, drop={drop})")
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
    print(f"{len(set(groups))} gruppi su {len(y)} battaglie (duplicati nello stesso fold)", flush=True)

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

    R = {r: REGIMES[r](sets["full"][0]).values.astype(float) for r in gate}
    stack = np.zeros(len(y))
    mean = np.zeros(len(y))
    for k in range(N_OUTER):
        tr_o, va_o = outer != k, outer == k
        Z = meta_inputs(models, {m: inner_oof[m][k, tr_o] for m in models}, gate, {r: v[tr_o] for r, v in R.items()})
        Zv = meta_inputs(models, {m: outer_pred[m][va_o] for m in models}, gate, {r: v[va_o] for r, v in R.items()})
        meta = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y[tr_o])
        stack[va_o] = meta.predict_proba(Zv)[:, 1]
        mean[va_o] = 1 / (1 + np.exp(-np.mean([_z(outer_pred[m][va_o]) for m in models], axis=0)))

    pred = pd.DataFrame({"battle_id": sets["train_ids"], "y_true": y, "fold": outer})
    for m in models:
        pred[f"p_{m}"] = outer_pred[m]
    pred["p_mean"] = mean
    pred["p_stack"] = stack
    pred["pred"] = (stack > 0.5).astype(int)
    pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    res = {"stack": metrics(y, stack, outer), "mean": metrics(y, mean, outer),
           "base": {m: metrics(y, outer_pred[m], outer) for m in models}}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"models": models, "extra": extra, "keep": keep, "drop": drop, "seed": SEED, "outer": N_OUTER, "inner": N_INNER,
                   "aug_weight": stacking.AUG_WEIGHT, "meta": "logreg C=1 sui logit", "gate": gate, "threshold": 0.5}, f, indent=1)
    s = res["stack"]
    print(f"\nNESTED stack (logreg) acc {s['acc']:.4f} | AUC {s['auc']:.4f} | logloss {s['logloss']:.4f}")
    print("  per fold:", " ".join(f"{a:.4f}" for a in s["acc_per_fold"]))
    print(f"NESTED media dei logit acc {res['mean']['acc']:.4f}")
    return res


def compare(a, b, n_boot=5000):
    """Differenza appaiata di accuracy (b - a) sulle stesse battaglie, con IC bootstrap al 95%."""
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
    print(f"{a}: {ca.mean():.4f}  {b}: {cb.mean():.4f}  diff {d.mean():+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
          f"P(diff>0) {np.mean(boot > 0):.2f}")
    print("  diff per fold:", " ".join(f"{x:+.4f}" for x in per_fold))
    print(f"  battaglie: {int(((cb == 1) & (ca == 0)).sum())} corrette solo da {b}, "
          f"{int(((ca == 1) & (cb == 0)).sum())} solo da {a}")
    return d.mean(), (lo, hi), per_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--models", default="logreg,lgbm,xgb,catboost")
    ap.add_argument("--extra", default="mc,rec")
    ap.add_argument("--keep", default="", help="prefissi delle sole feature da usare (es. strat_,prob_)")
    ap.add_argument("--drop", default="", help="prefissi delle feature da escludere")
    ap.add_argument("--gate", default="", help="regimi per il gating del meta (attrition,balanced,setup)")
    ap.add_argument("--aug-weight", type=float, default=None, help="peso degli snapshot nei modelli *_aug")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    configure(args.n_jobs, args.device)
    print(describe(), flush=True)
    if args.aug_weight is not None:
        stacking.AUG_WEIGHT = args.aug_weight
    split = lambda v: [x for x in v.split(",") if x]
    evaluate(args.name, split(args.models), split(args.extra), split(args.keep), split(args.drop), split(args.gate))


if __name__ == "__main__":
    main()
