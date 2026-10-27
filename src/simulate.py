"""Pseudo Monte Carlo: simula il resto della battaglia a partire dal turno 30.

Riusa il motore di danno di features.py (stats Gen 2, mosse rivelate, tipo di hiddenpower
inferito) e campiona le incertezze: pokemon di p2 non visti (dalle co-occorrenze),
random roll, crit, miss, durata del sonno, paralisi, scongelamento. Ogni battaglia viene
giocata fino alla fine N volte; le statistiche delle simulazioni diventano feature per lo stacking.

Uso: python src/simulate.py [--n-sims 40]  ->  outputs/mc_features.pkl
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from compute import configure, describe, pool_map
from features import (HEAL_MOVES, FIXED_DAMAGE, build_cooccurrence, build_pokedex, clean_battles,
                      damage_frac, infer_hp_types, load_jsonl, load_type_chart, stab_moves)

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.environ.get("POKEMON_DATA", os.path.join(ROOT, "data"))  # su Kaggle: /kaggle/input/...
OUT = os.environ.get("POKEMON_OUT", os.path.join(ROOT, "outputs"))
MAX_TURNS = 250
SNAP_TURNS = (24, 26, 28)
CRIT = 0.066

_G = {}  # stato globale dei worker (dex, chart, ...)


def _prep(battles):
    return clean_battles(battles) if _G.get("clean") else battles


def build_globals(clean=False):
    """Dati condivisi dalle simulazioni, calcolati una volta nel processo padre.
    clean=True: timeline con le sole mosse realmente eseguite (spikes, rest, mosse note)."""
    _G["clean"] = clean
    battles = _prep(load_jsonl(os.path.join(DATA, "train.jsonl")) + load_jsonl(os.path.join(DATA, "test.jsonl")))
    chart = load_type_chart(os.path.join(DATA, "type_chart_gen2.csv"))
    dex = build_pokedex(battles)
    return dict(clean=clean, dex=dex, chart=chart, hp_types=infer_hp_types(battles, dex, chart),
                cooc=build_cooccurrence(battles)[0])


def _set_globals(g):
    """Initializer dei worker: con spawn (Windows) i dati arrivano da qui, con fork sono già ereditati."""
    _G.update(g)


def boost_mult(stage):
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def parse_state(b):
    """Stato noto al turno 30: per ogni pokemon hp, status, mosse note, info sul sonno."""
    tl = b["battle_timeline"]
    side = {"p1": {}, "p2": {}}
    for p in b["p1_team_details"]:
        side["p1"][p["name"]] = dict(hp=1.0, status="nostatus", moves={}, level=p["level"], item=p["item"],
                                     rest_turn=None, slp_since=None, tox_turns=0)
    spikes = {"p1": False, "p2": False}
    for t in tl:
        for s, o in (("p1", "p2"), ("p2", "p1")):
            ps = t[s + "_pokemon_state"]
            mon = side[s].setdefault(ps["name"], dict(hp=1.0, status="nostatus", moves={}, level=100,
                                                      item="leftovers", rest_turn=None, slp_since=None, tox_turns=0))
            mon["hp"], prev_status, mon["status"] = ps["hp_pct"], mon["status"], ps["status"]
            if ps["status"] == "slp" and prev_status != "slp":
                mon["slp_since"] = t["turn"]
            mon["tox_turns"] = mon["tox_turns"] + 1 if ps["status"] == "tox" else 0
            m = t[s + "_move_details"]
            if m:
                mon["moves"][m["name"]] = m
                if m["name"] == "rest":
                    mon["rest_turn"] = t["turn"]
                elif m["name"] == "spikes":
                    spikes[o] = True
                elif m["name"] == "rapidspin":
                    if "ghost" not in _G["dex"].get(t[o + "_pokemon_state"]["name"], {}).get("types", []):
                        spikes[s] = False
    last = tl[-1]
    active = {s: last[s + "_pokemon_state"]["name"] for s in ("p1", "p2")}
    boosts = {s: last[s + "_pokemon_state"]["boosts"] for s in ("p1", "p2")}
    return side, active, boosts, spikes


class Battle:
    """Una singola simulazione. I pokemon sono indicizzati 0..n-1 per lato."""

    def __init__(self, b, side, active, boosts, spikes, rng):
        dex, chart, hp_types = _G["dex"], _G["chart"], _G["hp_types"]
        self.rng = rng
        mons = {s: {n: dict(v) for n, v in side[s].items() if n in dex} for s in ("p1", "p2")}
        # completa il team di p2 campionando i non visti dalle co-occorrenze
        n_missing = 6 - len(side["p2"])
        if n_missing > 0:
            probs = {}
            for a in mons["p2"]:
                for c, p in _G["cooc"].get(a, {}).items():
                    if c not in side["p2"] and c in dex:
                        probs[c] = probs.get(c, 0) + p
            cands = sorted(probs)  # ordine indipendente dall'hash randomizzato
            w = np.array([probs[c] for c in cands])
            if len(cands):
                for c in rng_np_choice(rng, cands, w, min(n_missing, len(cands))):
                    mons["p2"][c] = dict(hp=1.0, status="nostatus", moves={}, level=100, item="leftovers",
                                         rest_turn=None, slp_since=None, tox_turns=0)
        self.names = {s: list(mons[s]) for s in ("p1", "p2")}
        self.hp = {s: [mons[s][n]["hp"] for n in self.names[s]] for s in ("p1", "p2")}
        self.status = {s: [mons[s][n]["status"] for n in self.names[s]] for s in ("p1", "p2")}
        self.tox = {s: [mons[s][n]["tox_turns"] for n in self.names[s]] for s in ("p1", "p2")}
        self.leftovers = {s: [mons[s][n]["item"] == "leftovers" for n in self.names[s]] for s in ("p1", "p2")}
        self.spe = {s: [dex[n]["spe"] for n in self.names[s]] for s in ("p1", "p2")}
        self.flying = {s: ["flying" in dex[n]["types"] for n in self.names[s]] for s in ("p1", "p2")}
        self.heal = {s: [next((m for m in mons[s][n]["moves"] if m in HEAL_MOVES), None) for n in self.names[s]]
                     for s in ("p1", "p2")}
        self.spikes = dict(spikes)
        # turni di sonno rimanenti (now = ultimo turno osservato, 30 o lo snapshot)
        now = b["battle_timeline"][-1]["turn"]
        self.sleep = {}
        for s in ("p1", "p2"):
            rem = []
            for n in self.names[s]:
                m = mons[s][n]
                if m["status"] != "slp":
                    rem.append(0)
                elif m["rest_turn"] is not None and m["rest_turn"] >= (m["slp_since"] or 0):
                    rem.append(max(1, 2 - (now - m["rest_turn"])))
                else:
                    slept = now - (m["slp_since"] or now)
                    rem.append(rng.randint(1, max(1, 7 - slept)))
            self.sleep[s] = rem
        # matrici di danno: (danno grezzo frazionale, accuratezza) della mossa migliore
        self.dmg = {}
        for s, o in (("p1", "p2"), ("p2", "p1")):
            M = []
            for a in self.names[s]:
                ma = mons[s][a]
                mv = [m for m in ma["moves"].values() if m["category"] != "status" or m["name"] in FIXED_DAMAGE]
                if not mv:
                    # STAB generico: le mosse più usate dalla specie (build_species_moves) peggiorano
                    # l'AUC di mc_hp_diff da 0.693 a 0.684
                    mv = stab_moves(dex, a)
                row = []
                for d in self.names[o]:
                    best = (0.0, 1.0)
                    for m in mv:
                        acc = m["accuracy"] or 1.0
                        raw = damage_frac(dex, chart, a, d, dict(m, accuracy=1.0), ma["level"],
                                          mons[o][d]["level"], ma["item"], hp_types.get(a)) / 0.925
                        if raw * acc > best[0] * best[1]:
                            best = (raw, acc)
                    row.append(best)
                M.append(row)
            self.dmg[s] = M
        self.active = {s: self.names[s].index(active[s]) if active[s] in self.names[s] else 0 for s in ("p1", "p2")}
        self.boost = {s: (max(boosts[s]["atk"], boosts[s]["spa"]), max(boosts[s]["def"], boosts[s]["spd"]),
                          boosts[s]["spe"]) for s in ("p1", "p2")}
        self.turns = 0

    def alive(self, s):
        return [i for i, h in enumerate(self.hp[s]) if h > 0]

    def eff_speed(self, s):
        i = self.active[s]
        sp = self.spe[s][i] * boost_mult(self.boost[s][2])
        return sp * 0.25 if self.status[s][i] == "par" else sp

    def matchup(self, s, o, i):
        """Punteggio del pokemon i di s contro l'attivo avversario."""
        j = self.active[o]
        mine = self.dmg[s][i][j][0] * self.dmg[s][i][j][1]
        theirs = self.dmg[o][j][i][0] * self.dmg[o][j][i][1]
        return mine / max(self.hp[o][j], 0.05) - theirs / max(self.hp[s][i], 0.05)

    def switch_in(self, s, i):
        self.active[s] = i
        self.boost[s] = (0, 0, 0)
        if self.status[s][i] == "tox":
            self.tox[s][i] = 0
        if self.spikes[s] and not self.flying[s][i]:
            self.hp[s][i] = max(0.0, self.hp[s][i] - 0.125)

    def choose_replacement(self, s, o):
        alive = self.alive(s)
        return max(alive, key=lambda i: self.matchup(s, o, i) + self.rng.gauss(0, 0.3))

    def can_act(self, s):
        i = self.active[s]
        st = self.status[s][i]
        if st == "slp":
            self.sleep[s][i] -= 1
            if self.sleep[s][i] <= 0:
                self.status[s][i] = "nostatus"
            return False
        if st == "frz":
            if self.rng.random() < 0.1:
                self.status[s][i] = "nostatus"
            return False
        if st == "par" and self.rng.random() < 0.25:
            return False
        return True

    def act(self, s, o):
        """Azione del lato s: switch volontario, cura o attacco."""
        i, j = self.active[s], self.active[o]
        # switch volontario se il matchup è pessimo ed esiste un'alternativa nettamente migliore
        if self.matchup(s, o, i) < -2.0:
            alts = [k for k in self.alive(s) if k != i]
            if alts:
                k = max(alts, key=lambda k: self.matchup(s, o, k))
                if self.matchup(s, o, k) > self.matchup(s, o, i) + 1.5 and self.rng.random() < 0.5:
                    self.switch_in(s, k)
                    return
        if not self.can_act(s):
            return
        heal = self.heal[s][i]
        if heal and self.hp[s][i] < 0.4:
            if heal == "rest":
                self.hp[s][i], self.status[s][i], self.sleep[s][i] = 1.0, "slp", 2
            else:
                self.hp[s][i] = min(1.0, self.hp[s][i] + 0.5)
            return
        raw, acc = self.dmg[s][i][j]
        if raw <= 0 or self.rng.random() > acc:
            return
        dmg = raw * self.rng.uniform(0.85, 1.0) * boost_mult(self.boost[s][0]) / boost_mult(self.boost[o][1])
        if self.rng.random() < CRIT:
            dmg *= 2
        self.hp[o][j] = max(0.0, self.hp[o][j] - dmg)

    def residual(self, s):
        i = self.active[s]
        if self.hp[s][i] <= 0:
            return
        st = self.status[s][i]
        if st in ("psn", "brn"):
            self.hp[s][i] -= 0.125
        elif st == "tox":
            self.tox[s][i] += 1
            self.hp[s][i] -= self.tox[s][i] / 16
        if self.leftovers[s][i]:
            self.hp[s][i] += 1 / 16
        self.hp[s][i] = min(1.0, max(0.0, self.hp[s][i]))

    def run(self):
        while self.turns < MAX_TURNS:
            self.turns += 1
            for s, o in (("p1", "p2"), ("p2", "p1")):
                if self.hp[s][self.active[s]] <= 0:
                    if not self.alive(s):
                        return self.result()
                    self.switch_in(s, self.choose_replacement(s, o))
            s1, s2 = self.eff_speed("p1"), self.eff_speed("p2")
            first = "p1" if s1 > s2 or (s1 == s2 and self.rng.random() < 0.5) else "p2"
            second = "p2" if first == "p1" else "p1"
            self.act(first, second)
            if self.hp[second][self.active[second]] > 0:
                self.act(second, first)
            self.residual(first)
            self.residual(second)
            if not self.alive("p1") or not self.alive("p2"):
                return self.result()
        return self.result()

    def result(self):
        a1, a2 = len(self.alive("p1")), len(self.alive("p2"))
        h1, h2 = sum(self.hp["p1"]), sum(self.hp["p2"])
        win = 1.0 if a2 == 0 and a1 > 0 else 0.0 if a1 == 0 else float(h1 > h2)
        return win, a1, a2, h1, h2, self.turns, float(self.turns >= MAX_TURNS)


def rng_np_choice(rng, items, w, k):
    """Campionamento senza reinserimento pesato, usando random.Random (riproducibile)."""
    items, w = list(items), list(w)
    out = []
    for _ in range(k):
        r, acc = rng.random() * sum(w), 0.0
        for idx, wi in enumerate(w):
            acc += wi
            if acc >= r:
                break
        out.append(items.pop(idx))
        w.pop(idx)
    return out


def simulate_battle(args):
    b, n_sims = args
    side, active, boosts, spikes = parse_state(_prep([b])[0])
    rng = random.Random(b["battle_id"] * 7919 + int(b.get("player_won", -1) == -1) + 100 * len(b["battle_timeline"]))
    res = np.array([Battle(b, side, active, boosts, spikes, rng).run() for _ in range(n_sims)])
    win, a1, a2, h1, h2, turns, capped = res.T
    return {"mc_win": win.mean(), "mc_alive_diff": (a1 - a2).mean(), "mc_hp_diff": (h1 - h2).mean(),
            "mc_turns": turns.mean(), "mc_turns_std": turns.std(), "mc_capped": capped.mean(),
            "mc_p1_alive": a1.mean(), "mc_p2_alive": a2.mean(), "mc_win_margin": ((a1 - a2) * (2 * win - 1)).mean(),
            "mc_win_std": win.std(), "mc_hp_diff_p10": np.percentile(h1 - h2, 10),
            "mc_hp_diff_p90": np.percentile(h1 - h2, 90), "mc_turns_left_median": np.median(turns)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=100)  # valore usato per le feature dello stack finale
    ap.add_argument("--limit", type=int, default=0, help="solo le prime N battaglie di train (debug)")
    ap.add_argument("--clean", action="store_true", help="usa le timeline pulite dalle mosse persistenti")
    ap.add_argument("--n-jobs", type=int, default=0, help="processi paralleli (default: tutti i core)")
    args = ap.parse_args()
    configure(args.n_jobs)
    print(describe())
    train = load_jsonl(os.path.join(DATA, "train.jsonl"))
    test = load_jsonl(os.path.join(DATA, "test.jsonl"))
    for b in test:
        b["player_won"] = -1
    battles = train[:args.limit] if args.limit else train + test
    # snapshot dei battle di train per l'augmentation dello stacking (stessi turni di stacking.AUG_TURNS)
    snaps = [] if args.limit else [{**b, "battle_timeline": b["battle_timeline"][:t]} for t in SNAP_TURNS for b in train]
    snap_turn = [30] * len(battles) + [len(b["battle_timeline"]) for b in snaps]
    battles = battles + snaps
    t0 = time.time()
    g = build_globals(args.clean)
    rows = pool_map(simulate_battle, [(b, args.n_sims) for b in battles], _set_globals, (g,), chunksize=50)
    df = pd.DataFrame(rows)
    df.insert(0, "battle_id", [b["battle_id"] for b in battles])
    df.insert(1, "is_test", [int(b["player_won"] == -1) for b in battles])
    df.insert(2, "snap_turn", snap_turn)
    y = np.array([b["player_won"] for b in battles])
    trm = (df.is_test.values == 0) & (df.snap_turn.values == 30)
    print(f"{len(df)} battaglie, {args.n_sims} sim ciascuna, {time.time() - t0:.0f}s")
    print(f"accuracy di mc_win > 0.5 sul train: {((df.mc_win.values[trm] > 0.5) == y[trm]).mean():.4f}")
    print(df[trm].describe().T[["mean", "std"]])
    if not args.limit:
        os.makedirs(OUT, exist_ok=True)
        df.to_pickle(os.path.join(OUT, "mc_features.pkl"))


if __name__ == "__main__":
    main()
