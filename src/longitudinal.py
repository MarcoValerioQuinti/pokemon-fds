"""Storia per singolo pokemon fino all'ultimo turno osservato (Fase 6).

Per ogni pokemon visto si traccia, turno per turno:
  - ingressi e turni in campo; HP all'ingresso;
  - danno subito, recupero (cure e leftovers), danno inflitto attribuito alle sue mosse reali;
  - turni con status (sonno separato), boost massimi, ingressi con le spikes;
  - mosse riportate ma non eseguite (persistenti, features.get_actual_move) tenute separate da quelle reali.
Le mosse reali servono solo ad attribuire danni ed effetti; il conteggio delle persistenti resta
come feature di comportamento. Le storie vengono aggregate per lato con max/media/quantili e conteggi
sopra soglia, così una singola win condition non sparisce nella media.

Uso: python src/longitudinal.py  ->  outputs/lon_features.pkl (turno 30 + snapshot 24/26/28 di train)
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from compute import configure, describe, pool_map
from features import build_pokedex, get_actual_move, load_jsonl
from train import DATA, OUT

SNAP_TURNS = (24, 26, 28)
FIELDS = ["entries", "turns", "hp_entry_min", "dmg_taken", "healed", "dmg_dealt", "kos", "status_turns",
          "sleep_turns", "max_boost", "spikes_entries", "persistent", "actual_moves"]
_W = {}


def histories(b):
    """Dizionario lato -> specie -> statistiche cumulative."""
    dex = _W["dex"]
    H = {s: defaultdict(lambda: dict(entries=0, turns=0, hp_entry_min=1.0, dmg_taken=0.0, healed=0.0,
                                     dmg_dealt=0.0, kos=0, status_turns=0, sleep_turns=0, max_boost=0,
                                     spikes_entries=0, persistent=0, actual_moves=0, hp=1.0, recent=[]))
         for s in ("p1", "p2")}
    spikes = {"p1": False, "p2": False}
    last_move = {"p1": {}, "p2": {}}
    prev = None
    for t in b["battle_timeline"]:
        for s, o in (("p1", "p2"), ("p2", "p1")):
            ps = t[s + "_pokemon_state"]
            n = ps["name"]
            h = H[s][n]
            if prev is None or prev[s + "_pokemon_state"]["name"] != n:
                h["entries"] += 1
                h["hp_entry_min"] = min(h["hp_entry_min"], ps["hp_pct"])
                if spikes[s] and "flying" not in dex.get(n, {}).get("types", []):
                    h["spikes_entries"] += 1
            h["turns"] += 1
            delta = ps["hp_pct"] - h["hp"]
            if delta < 0:
                h["dmg_taken"] -= delta
            elif delta > 0:
                h["healed"] += delta
            h["hp"] = ps["hp_pct"]
            if ps["status"] == "slp":
                h["sleep_turns"] += 1
            elif ps["status"] in ("par", "psn", "tox", "brn", "frz"):
                h["status_turns"] += 1
            h["max_boost"] = max(h["max_boost"], sum(ps["boosts"].values()))
            reported = t[s + "_move_details"]
            m = get_actual_move(prev, t, s, last_move[s].get(n))
            if reported is not None and m is None:
                h["persistent"] += 1
            if m is not None:
                h["actual_moves"] += 1
                last_move[s][n] = m["name"]
                if m["name"] == "spikes":
                    spikes[o] = True
                elif m["name"] == "rapidspin":
                    spikes[s] = False
            # danno inflitto: calo di HP dell'avversario rimasto in campo, se questo pokemon ha agito
            if prev is not None and m is not None and m["category"] != "status":
                po, co = prev[o + "_pokemon_state"], t[o + "_pokemon_state"]
                if po["name"] == co["name"]:
                    d = po["hp_pct"] - co["hp_pct"]
                    if d > 0:
                        h["dmg_dealt"] += d
                        h["recent"].append((t["turn"], d))
                        if co["hp_pct"] == 0:
                            h["kos"] += 1
        prev = t
    return H


def side_features(H, side, team, alive_names, active, now):
    f = {}
    mons = list(H[side].values())
    for k in FIELDS:
        v = np.array([m[k] for m in mons], dtype=float) if mons else np.zeros(1)
        f[f"lon_{side}_{k}_max"] = v.max()
        f[f"lon_{side}_{k}_mean"] = v.mean()
    alive = [H[side][n] for n in alive_names if n in H[side]]
    f[f"lon_{side}_alive_dmg_dealt_max"] = max((m["dmg_dealt"] for m in alive), default=0.0)
    f[f"lon_{side}_alive_dmg_dealt_q75"] = np.percentile([m["dmg_dealt"] for m in alive], 75) if alive else 0.0
    f[f"lon_{side}_n_strong_attackers"] = sum(m["dmg_dealt"] >= 0.8 for m in alive)  # vivi che hanno fatto molto danno
    f[f"lon_{side}_n_healthy_unused"] = sum(1 for n in alive_names if n not in H[side]) + \
        sum(m["hp"] > 0.7 and m["turns"] <= 3 for m in alive)  # vivi poco usati e in salute
    f[f"lon_{side}_n_worn"] = sum(m["dmg_taken"] >= 1.0 and m["hp"] > 0 for m in alive)  # logorati ma vivi
    a = H[side].get(active)
    f[f"lon_{side}_active_turns"] = a["turns"] if a else 0
    f[f"lon_{side}_active_dmg_last5"] = sum(d for tt, d in a["recent"] if tt > now - 5) if a else 0.0
    f[f"lon_{side}_active_dmg_dealt"] = a["dmg_dealt"] if a else 0.0
    f[f"lon_{side}_active_persistent"] = a["persistent"] if a else 0
    return f


def battle_row(b):
    H = histories(b)
    last = b["battle_timeline"][-1]
    now = last["turn"]
    p1_team = [p["name"] for p in b["p1_team_details"]]
    alive = {"p1": [n for n in p1_team if H["p1"][n]["hp"] > 0],
             "p2": [n for n, m in H["p2"].items() if m["hp"] > 0]}
    f = {}
    for s in ("p1", "p2"):
        f.update(side_features(H, s, p1_team if s == "p1" else None, alive[s], last[s + "_pokemon_state"]["name"], now))
    for k in list(f):
        if k.startswith("lon_p1_"):
            f["lon_diff_" + k[7:]] = f[k] - f["lon_p2_" + k[7:]]
    return f


def _init(dex):
    _W["dex"] = dex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=0)
    args = ap.parse_args()
    configure(args.n_jobs)
    print(describe())
    t0 = time.time()
    train = load_jsonl(os.path.join(DATA, "train.jsonl"))
    test = load_jsonl(os.path.join(DATA, "test.jsonl"))
    dex = build_pokedex(train + test)
    snaps = [{**b, "battle_timeline": b["battle_timeline"][:t]} for t in SNAP_TURNS for b in train]
    battles = train + test + snaps
    df = pd.DataFrame(pool_map(battle_row, battles, _init, (dex,), chunksize=200))
    df.insert(0, "battle_id", [b["battle_id"] for b in battles])
    df.insert(1, "is_test", [0] * len(train) + [1] * len(test) + [0] * len(snaps))
    df.insert(2, "snap_turn", [30] * (len(train) + len(test)) + [t for t in SNAP_TURNS for _ in train])
    print(f"{len(df)} righe, {df.shape[1] - 3} feature, {time.time() - t0:.0f}s")
    from sklearn.metrics import roc_auc_score
    y = np.array([b["player_won"] for b in train])
    m = (df.is_test.values == 0) & (df.snap_turn.values == 30)
    aucs = {c: roc_auc_score(y, df.loc[m, c]) for c in df.columns[3:]}
    for c, a in sorted(aucs.items(), key=lambda kv: -abs(kv[1] - 0.5))[:15]:
        print(f"  AUC {c:40s} {a:.3f}")
    df.to_pickle(os.path.join(OUT, "lon_features.pkl"))


if __name__ == "__main__":
    main()
