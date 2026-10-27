"""Modello ricorsivo "stile RNN" con ML classico.

1. Un modello di transizione impara l'esito di un singolo turno:
   (stato_t, memoria_t) -> (delta HP p1, delta HP p2, P(KO p1), P(KO p2)).
   La memoria è una media esponenziale dei delta passati (l'analogo dell'hidden state
   di una RNN). Non usa le etichette, quindi si allena anche sulle timeline di test.
2. Dal turno 30 il modello viene applicato ricorsivamente alle proprie uscite (con
   residui campionati) fino a che un team è esausto: si ricostruisce il resto della battaglia.
3. Le statistiche dei rollout diventano feature per lo stacking.

Uso: python src/recursive.py [--n-rollouts 30] [--min-turn 18]  ->  outputs/rec_features.pkl
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from compute import configure, describe, n_jobs
from features import load_jsonl

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
EMA = 0.8
MAX_TURNS = 200
SNAP_TURNS = (24, 26, 28)
STATE = ["hp1", "hp2", "alive1", "alive2", "slp1", "slp2", "par1", "par2", "psn1", "psn2", "spk1", "spk2",
         "ema_dhp1", "ema_dhp2", "ema_f1", "ema_f2"]
STATIC = ["p1_bst", "p1_spe", "p1_def", "p1_spd"]


def battle_sequence(b):
    """Sequenza degli stati aggregati turno per turno + contesto statico."""
    team = b["p1_team_details"]
    static = [np.mean([sum(p["base_" + s] for s in ("hp", "atk", "def", "spa", "spd", "spe")) for p in team]),
              np.mean([p["base_spe"] for p in team]), np.mean([p["base_def"] for p in team]),
              np.mean([p["base_spd"] for p in team])]
    hp = {"p1": {p["name"]: 1.0 for p in team}, "p2": {}}
    st = {"p1": {}, "p2": {}}
    spikes = {"p1": 0, "p2": 0}
    rows = []
    for t in b["battle_timeline"]:
        for s, o in (("p1", "p2"), ("p2", "p1")):
            ps = t[s + "_pokemon_state"]
            hp[s][ps["name"]], st[s][ps["name"]] = ps["hp_pct"], ps["status"]
            m = t[s + "_move_details"]
            if m and m["name"] == "spikes":
                spikes[o] = 1
            elif m and m["name"] == "rapidspin":
                spikes[s] = 0
        r = []
        for s in ("p1", "p2"):
            unseen = max(6 - len(hp[s]), 0)
            r.append(sum(hp[s].values()) + unseen)
        for s in ("p1", "p2"):
            r.append(6 - sum(1 for h in hp[s].values() if h == 0))
        for code in (("slp",), ("par",), ("psn", "tox")):
            for s in ("p1", "p2"):
                r.append(sum(1 for n, x in st[s].items() if x in code and hp[s][n] > 0))
        r += [spikes["p1"], spikes["p2"]]
        rows.append(r)
    return np.array(rows, dtype=float), np.array(static)


def add_memory(seq):
    """Aggiunge la memoria (EMA dei delta di HP e dei KO) a ogni turno."""
    mem = np.zeros((len(seq), 4))
    h = np.zeros(4)
    for t in range(1, len(seq)):
        d = np.array([seq[t, 0] - seq[t - 1, 0], seq[t, 1] - seq[t - 1, 1],
                      seq[t - 1, 2] - seq[t, 2], seq[t - 1, 3] - seq[t, 3]])
        h = EMA * h + (1 - EMA) * d
        mem[t] = h
    return np.hstack([seq, mem])


def last_state(b):
    """Stato (con memoria) all'ultimo turno della timeline, punto di partenza dei rollout."""
    seq, static = battle_sequence(b)
    return np.concatenate([add_memory(seq)[-1], static])


def build_transitions(battles, min_turn=1):
    """Transizioni turno -> turno. Esclude i turni di rimpiazzo dopo un KO (nessuno agisce
    davvero) e, con min_turn, i primi turni della battaglia."""
    X, Y, last = [], [], []
    for b in battles:
        seq, static = battle_sequence(b)
        S = add_memory(seq)
        last.append(np.concatenate([S[-1], static]))
        for t in range(len(S) - 1):
            if t + 1 < min_turn:
                continue
            if t > 0 and (S[t, 2] < S[t - 1, 2] or S[t, 3] < S[t - 1, 3]):
                continue  # KO nel turno t: t -> t+1 è il turno di rimpiazzo
            X.append(np.concatenate([S[t], static]))
            Y.append([S[t + 1, 0] - S[t, 0], S[t + 1, 1] - S[t, 1],
                      float(S[t + 1, 2] < S[t, 2]), float(S[t + 1, 3] < S[t, 3])])
    return np.array(X), np.array(Y), np.array(last)


def fit_transition(X, Y, n_trees=100):
    """Quattro modelli: due regressori sui delta HP, due classificatori sui KO."""
    # learning rate scalato col numero di alberi: meno alberi = rollout più veloci
    params = dict(n_estimators=n_trees, learning_rate=15 / n_trees, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
                  n_jobs=n_jobs())
    n = len(X)
    rng = np.random.default_rng(0)
    va = rng.random(n) < 0.1
    models = []
    for k in range(4):
        cls = lgb.LGBMClassifier if k >= 2 else lgb.LGBMRegressor
        m = cls(**params).fit(X[~va], Y[~va, k])
        if k >= 2:
            print(f"  P(KO {'p1' if k == 2 else 'p2'}) AUC validation: {roc_auc_score(Y[va, k], m.predict_proba(X[va])[:, 1]):.3f}")
        else:
            print(f"  delta HP {'p1' if k == 0 else 'p2'} R2 validation: {r2_score(Y[va, k], m.predict(X[va])):.3f}")
        models.append(m)
    residuals = [Y[:, k] - models[k].predict(X) for k in (0, 1)]
    return models, residuals


def rollout(models, residuals, start, n_rollouts, seed=0):
    """Applica ricorsivamente il modello di transizione da turno 30 fino alla fine."""
    rng = np.random.default_rng(seed)
    n_b = len(start)
    S = np.repeat(start, n_rollouts, axis=0)
    idx = np.repeat(np.arange(n_b), n_rollouts)
    done = np.zeros(len(S), bool)
    win = np.zeros(len(S))
    turns = np.zeros(len(S))
    snap = {}
    for step in range(1, MAX_TURNS + 1):
        act = np.where(~done)[0]
        if len(act) == 0:
            break
        X = S[act]
        dhp1 = models[0].predict(X) + rng.choice(residuals[0], len(act))
        dhp2 = models[1].predict(X) + rng.choice(residuals[1], len(act))
        f1 = rng.random(len(act)) < models[2].predict_proba(X)[:, 1]
        f2 = rng.random(len(act)) < models[3].predict_proba(X)[:, 1]
        new = X.copy()
        new[:, 2] -= f1
        new[:, 3] -= f2
        new[:, 0] = np.clip(X[:, 0] + dhp1, 0, new[:, 2])
        new[:, 1] = np.clip(X[:, 1] + dhp2, 0, new[:, 3])
        # i dormienti si svegliano con probabilità 1/3 a turno; status persi coi KO
        for c in (4, 5):
            new[:, c] = np.maximum(0, new[:, c] - (rng.random(len(act)) < new[:, c] / 3))
        for c, a in ((4, 2), (5, 3), (6, 2), (7, 3), (8, 2), (9, 3)):
            new[:, c] = np.minimum(new[:, c], new[:, a])
        d = np.column_stack([new[:, 0] - X[:, 0], new[:, 1] - X[:, 1], f1, f2])
        new[:, 12:16] = EMA * X[:, 12:16] + (1 - EMA) * d
        S[act] = new
        end = (new[:, 2] <= 0) | (new[:, 3] <= 0)
        ended = act[end]
        win[ended] = (new[end, 3] <= 0) & (new[end, 2] > 0)
        turns[ended] = step
        done[ended] = True
        if step in (10, 30):
            snap[step] = S[:, [0, 1, 2, 3]].copy()
    open_ = ~done
    win[open_] = S[open_, 0] > S[open_, 1]
    turns[open_] = MAX_TURNS
    agg = lambda v: pd.Series(v).groupby(idx).mean().values
    out = {"rec_win": agg(win), "rec_turns": agg(turns), "rec_capped": agg(open_.astype(float)),
           "rec_alive_diff_end": agg(S[:, 2] - S[:, 3]), "rec_hp_diff_end": agg(S[:, 0] - S[:, 1])}
    for step, v in snap.items():
        out[f"rec_hp_diff_{step}"] = agg(v[:, 0] - v[:, 1])
        out[f"rec_alive_diff_{step}"] = agg(v[:, 2] - v[:, 3])
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rollouts", type=int, default=30)
    ap.add_argument("--min-turn", type=int, default=1, help="allena solo sulle transizioni dal turno N in poi")
    ap.add_argument("--trees", type=int, default=100, help="alberi per modello di transizione")
    ap.add_argument("--n-jobs", type=int, default=0, help="thread (default: tutti i core)")
    ap.add_argument("--out", default="rec_features.pkl")
    args = ap.parse_args()
    configure(args.n_jobs)
    print(describe())
    t0 = time.time()
    train = load_jsonl(os.path.join(DATA, "train.jsonl"))
    test = load_jsonl(os.path.join(DATA, "test.jsonl"))
    battles = train + test
    X, Y, last = build_transitions(battles, args.min_turn)
    print(f"{len(X)} transizioni da {len(battles)} battaglie ({time.time() - t0:.0f}s)")
    models, residuals = fit_transition(X, Y, args.trees)
    # anche gli snapshot dei battle di train, per l'augmentation dello stacking
    snap_turns = [t for t in SNAP_TURNS for _ in train]
    snap_last = np.array([last_state({**b, "battle_timeline": b["battle_timeline"][:t]})
                          for t in SNAP_TURNS for b in train])
    df = rollout(models, residuals, np.vstack([last, snap_last]), args.n_rollouts)
    df.insert(0, "battle_id", [b["battle_id"] for b in battles] + [b["battle_id"] for _ in SNAP_TURNS for b in train])
    df.insert(1, "is_test", [0] * len(train) + [1] * len(test) + [0] * len(snap_last))
    df.insert(2, "snap_turn", [30] * len(battles) + snap_turns)
    y = np.array([b["player_won"] for b in train])
    trm = (df.is_test.values == 0) & (df.snap_turn.values == 30)
    print(f"rollout completati ({time.time() - t0:.0f}s)")
    print(f"accuracy di rec_win > 0.5 sul train: {((df.rec_win.values[trm] > 0.5) == y).mean():.4f}")
    print(df[trm].describe().T[["mean", "std"]])
    df.to_pickle(os.path.join(OUT, args.out))


if __name__ == "__main__":
    main()
