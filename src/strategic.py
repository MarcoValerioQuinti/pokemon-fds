"""Team-wide strategic matchup features (Phase 3) and probabilistic features (Phase 4).

For each living attacker-defender pair on opposing sides, use the features.py damage
engine (compared with Showdown in tests/test_damage.py) to estimate:
  - best expected move damage (revealed moves, common species moves, then STAB);
  - damage distribution: 39 rolls (217..255)/255, critical probability 17/256 (x2), accuracy;
  - one-hit and two-hit KO probabilities, hits to KO and speed order;
  - the one-on-one duel outcome (which Pokemon scores a KO first).
Aggregate matrices into interpretable strat_ features and probabilistic active-exchange
prob_ features, kept separate for A/B comparisons. Complete unseen p2 Pokemon using
K label-free co-occurrence scenarios; report team-score means and standard deviations.

Usage: python src/strategic.py -> outputs/strat_features.pkl (turn 30 and train snapshots 24/26/28)
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from compute import configure, describe, pool_map
from features import (FIXED_DAMAGE, SPECIAL_TYPES, boost_mult, build_context, build_pokedex, damage_frac,
                      load_jsonl, load_type_chart, stab_moves)
from team_infer import TeamPool, cooc_probs, seen_order
from train import DATA, OUT

SNAP_TURNS = (24, 26, 28)
K_SCENARIOS = 8
POOL_LAMBDA = 0.85  # Pool weight versus co-occurrence, chosen by label-free reconstruction evaluation
ROLLS = np.arange(217, 256) / 255
P_CRIT = 17 / 256
_W = {}


def hit_dist(raw, acc):
    """Discrete single-hit damage distribution, as fractions of maximum HP: values and probabilities."""
    vals = np.concatenate([[0.0], raw * ROLLS, 2 * raw * ROLLS])
    p = np.concatenate([[1 - acc], np.full(39, acc * (1 - P_CRIT) / 39), np.full(39, acc * P_CRIT / 39)])
    return vals, p


def p_ko(raw, acc, hp, hits):
    """P(cumulative damage >= hp) within `hits` attacks (one or two)."""
    if raw <= 0 or hp <= 0:
        return float(hp <= 0)
    v, p = hit_dist(raw, acc)
    if hits == 1:
        return float(p[v >= hp].sum())
    tot = v[:, None] + v[None, :]
    return float((p[:, None] * p[None, :])[tot >= hp].sum())


class State:
    """Final timeline state: HP, status, known moves, levels and p1 items."""

    def __init__(self, b):
        self.p1_items = {p["name"]: p["item"] for p in b["p1_team_details"]}
        self.p1_level = {p["name"]: p["level"] for p in b["p1_team_details"]}
        self.hp = {"p1": {n: 1.0 for n in self.p1_items}, "p2": {}}
        self.status = {"p1": {}, "p2": {}}
        self.known = {"p1": {}, "p2": {}}
        for t in b["battle_timeline"]:
            for s in ("p1", "p2"):
                ps = t[s + "_pokemon_state"]
                self.hp[s][ps["name"]] = ps["hp_pct"]
                self.status[s][ps["name"]] = ps["status"]
                m = t[s + "_move_details"]
                if m:
                    self.known[s].setdefault(ps["name"], {})[m["name"]] = m
        last = b["battle_timeline"][-1]
        self.active = {s: last[s + "_pokemon_state"]["name"] for s in ("p1", "p2")}
        self.boosts = {s: last[s + "_pokemon_state"]["boosts"] for s in ("p1", "p2")}
        self.sub = {s: "substitute" in last[s + "_pokemon_state"]["effects"] for s in ("p1", "p2")}
        self.virtual = {}  # Moves of unseen p2 Pokemon in the current scenario


def best_hit(st, side, att, dfn, cache):
    """Return (raw fractional damage without roll/accuracy, accuracy, is_special)
    for att's best expected move against dfn. Moves come from st.known or
    st.virtual (unseen scenario Pokemon with moves from a pool team)."""
    moves = st.known[side].get(att) or st.virtual.get(att) or {}
    key = (side, att, dfn, tuple(moves))
    if key in cache:
        return cache[key]
    dex, chart, ctx = _W["dex"], _W["chart"], _W["ctx"]
    mv = [m for m in moves.values() if m["category"] != "status" or m["name"] in FIXED_DAMAGE]
    if not mv:
        mv = ctx["species_moves"].get(att) or stab_moves(dex, att)
    lv_a = st.p1_level.get(att, 100) if side == "p1" else 100
    lv_d = 100 if side == "p1" else st.p1_level.get(dfn, 100)
    item = st.p1_items.get(att) if side == "p1" else None
    best = (0.0, 1.0, False)
    for m in mv:
        acc = m["accuracy"] or 1.0
        raw = damage_frac(dex, chart, att, dfn, dict(m, accuracy=1.0), lv_a, lv_d, item,
                          ctx["hp_types"].get(att)) / 0.925
        if raw * acc > best[0] * best[1]:
            mtype = ctx["hp_types"].get(att, m["type"]) if m["name"] == "hiddenpower" else m["type"]
            best = (raw, acc, mtype in SPECIAL_TYPES)
    if st.status[side].get(att) == "brn" and not best[2]:
        best = (best[0] * 0.5, best[1], best[2])
    cache[key] = best
    return best


def speed(st, side, n, active_boost=True):
    s = _W["dex"][n]["spe"]
    if active_boost and n == st.active[side]:
        s *= boost_mult(st.boosts[side]["spe"])
    return s * (0.25 if st.status[side].get(n) == "par" else 1.0)


def pair_matrix(st, A, D, side, other, hp, cache):
    """Living-Pokemon matrices: expected damage, hits to KO, P(KO1), P(KO2), speed advantage."""
    n, m = len(A), len(D)
    exp_dmg, htk, ko1, ko2, faster = (np.zeros((n, m)) for _ in range(5))
    for i, a in enumerate(A):
        for j, d in enumerate(D):
            raw, acc, _ = best_hit(st, side, a, d, cache)
            h = hp[other][d]
            exp_dmg[i, j] = raw * 0.925 * acc
            htk[i, j] = np.ceil(h / max(exp_dmg[i, j], 0.02))
            ko1[i, j] = p_ko(raw, acc, h, 1)
            ko2[i, j] = p_ko(raw, acc, h, 2)
            faster[i, j] = speed(st, side, a, False) > speed(st, other, d, False)
    return exp_dmg, htk, ko1, ko2, faster


def team_scores(st, A1, A2, hp, cache):
    """Team scores from p1 against p2 and vice versa."""
    _, htk12, ko1_12, ko2_12, fast12 = pair_matrix(st, A1, A2, "p1", "p2", hp, cache)
    _, htk21, ko1_21, ko2_21, _ = pair_matrix(st, A2, A1, "p2", "p1", hp, cache)
    # Attacker wins with fewer hits to KO, or equal hits and higher speed.
    t_a, t_d = htk12, htk21.T
    win1 = (t_a < t_d) | ((t_a == t_d) & (fast12 > 0))
    margin = np.log(t_d) - np.log(t_a)  # Positive values favor p1.
    f = {}
    f["p1_answered_frac"] = win1.any(axis=0).mean()   # Opponents with at least one p1 counter
    f["p2_answered_frac"] = (~win1).any(axis=1).mean()  # p1 Pokemon with at least one p2 counter
    f["p1_unanswered_threats"] = (~(~win1).any(axis=1)).sum()  # p1 Pokemon that no p2 Pokemon beats
    f["p2_unanswered_threats"] = (~win1.any(axis=0)).sum()
    f["fav_pair_frac"] = win1.mean()
    score1 = margin.mean(axis=1)  # Mean score of each p1 Pokemon against living p2 opponents
    score2 = -margin.mean(axis=0)
    for side, sc in (("p1", score1), ("p2", score2)):
        f[f"{side}_mon_score_min"], f[f"{side}_mon_score_med"], f[f"{side}_mon_score_max"] = (
            sc.min(), np.median(sc), sc.max())
    # Optimal duel assignment: a strategic proxy, not the actual battle order
    r, c = linear_sum_assignment(-margin)
    f["matching_margin"] = margin[r, c].sum() / max(len(A1), len(A2))
    f["p1_ohko_frac"] = (ko1_12.max(axis=0) > 0.5).mean()
    f["p2_ohko_frac"] = (ko1_21.max(axis=0) > 0.5).mean()
    f["p1_2hko_frac"] = (ko2_12.max(axis=0) > 0.5).mean()
    f["p2_2hko_frac"] = (ko2_21.max(axis=0) > 0.5).mean()
    return f


def active_features(st, A1, A2, hp, cache):
    """Active exchange and safe switches, including active boosts and substitute."""
    f = {}
    a1, a2 = st.active["p1"], st.active["p2"]
    if a1 not in A1 or a2 not in A2:
        return f
    for side, other, a, d in (("p1", "p2", a1, a2), ("p2", "p1", a2, a1)):
        raw, acc, special = best_hit(st, side, a, d, cache)
        off, dfs = ("spa", "spd") if special else ("atk", "def")
        raw *= boost_mult(st.boosts[side][off]) / boost_mult(st.boosts[other][dfs])
        h = hp[other][d] + (0.25 if st.sub[other] else 0.0)
        f[f"prob_{side}_ko1"] = p_ko(raw, acc, h, 1)
        f[f"prob_{side}_ko2"] = p_ko(raw, acc, h, 2)
        f[f"prob_{side}_exp_dmg"] = raw * 0.925 * acc
        v, p = hit_dist(raw, acc)
        f[f"prob_{side}_dmg_p10"] = v[np.searchsorted(np.cumsum(p), 0.1)]
        f[f"prob_{side}_dmg_p90"] = v[np.searchsorted(np.cumsum(p), 0.9)]
    sp1, sp2 = speed(st, "p1", a1), speed(st, "p2", a2)
    p1_first = 1.0 if sp1 > sp2 else 0.0 if sp1 < sp2 else 0.5
    # P(p1 KOs the p2 active before being KOed), within a two-turn horizon
    k1, k2 = f["prob_p1_ko1"], f["prob_p2_ko1"]
    f["prob_p1_survives_response"] = 1 - k2
    f["prob_p2_survives_response"] = 1 - k1
    f["prob_p1_wins_exchange"] = p1_first * (k1 + (1 - k1) * (1 - k2) * f["prob_p1_ko2"]) + \
        (1 - p1_first) * (1 - k2) * (k1 + (1 - k1) * (1 - f["prob_p2_ko2"]) * f["prob_p1_ko2"])
    f["prob_p2_wins_exchange"] = (1 - p1_first) * (k2 + (1 - k2) * (1 - k1) * f["prob_p2_ko2"]) + \
        p1_first * (1 - k1) * (k2 + (1 - k2) * (1 - f["prob_p1_ko2"]) * f["prob_p2_ko2"])
    f["prob_exchange_diff"] = f["prob_p1_wins_exchange"] - f["prob_p2_wins_exchange"]
    # Safe switches: entrants survive two hits with probability >0.5.
    for side, other, A, d in (("p1", "p2", A1, a2), ("p2", "p1", A2, a1)):
        bench = [a for a in A if a != st.active[side]]
        safe = []
        for a in bench:
            raw, acc, _ = best_hit(st, other, d, a, cache)
            safe.append(p_ko(raw, acc, hp[side][a], 2) < 0.5)
        f[f"strat_{side}_safe_switches"] = float(sum(safe))
        # Best available response to the opposing active Pokemon, measured by duel margin
        best = -9.0
        for a in A:
            r1, c1, _ = best_hit(st, side, a, d, cache)
            r2, c2, _ = best_hit(st, other, d, a, cache)
            t_a = np.ceil(hp[other][d] / max(r1 * 0.925 * c1, 0.02))
            t_d = np.ceil(hp[side][a] / max(r2 * 0.925 * c2, 0.02))
            best = max(best, np.log(t_d) - np.log(t_a))
        f[f"strat_{side}_best_answer"] = best
    return f


def battle_row(b):
    dex, cooc = _W["dex"], _W["ctx"]["cooc"]
    st = State(b)
    hp = {s: dict(st.hp[s]) for s in ("p1", "p2")}
    A1 = [n for n, h in hp["p1"].items() if h > 0 and n in dex]
    A2 = [n for n, h in hp["p2"].items() if h > 0 and n in dex]
    cache = {}
    f = {}
    if A1 and A2:
        f.update(active_features(st, A1, A2, hp, cache))
        seen = {k: v for k, v in team_scores(st, A1, A2, hp, cache).items()}
        f.update({f"strat_seen_{k}": v for k, v in seen.items()})
        # Unseen-p2 scenarios: co-occurrence (strat_full_) and team pool (strat_pool_)
        n_unseen = max(6 - len(st.hp["p2"]), 0)
        seen2 = sorted(st.hp["p2"])
        probs = {}
        for a in seen2:
            for c, p in cooc.get(a, {}).items():
                if c not in st.hp["p2"] and c in dex:
                    probs[c] = probs.get(c, 0) + p
        cands = sorted(probs)
        rng = random.Random(b["battle_id"] * 31 + len(b["battle_timeline"]))
        pool = _W["ctx"]["pool"]
        teams, level = pool.candidates(seen2)
        teams = sorted(((sorted(t), w) for t, w in teams.items() if t - set(seen2)))
        tot = sum(w for _, w in teams)
        f["pool_level"], f["pool_n_teams"] = level, len(teams)
        f["pool_top_share"] = max((w for _, w in teams), default=0) / tot if tot else 0.0
        f["pool_entropy"] = -sum(w / tot * np.log(w / tot) for _, w in teams) if tot else 0.0
        pp, _, _ = pool.member_probs(seen2)
        mp = {c: POOL_LAMBDA * pp.get(c, 0) + (1 - POOL_LAMBDA) * q
              for c, q in cooc_probs(cooc, seen2, n_unseen).items()} if n_unseen else {}
        for c, p in pp.items():
            mp.setdefault(c, POOL_LAMBDA * p)
        mp = {c: p for c, p in mp.items() if c in dex}
        tot_p = sum(mp.values())
        for s_ in ("hp", "atk", "def", "spa", "spd", "spe"):
            f[f"pool_unseen_exp_{s_}"] = sum(p * dex[c][s_] for c, p in mp.items()) / tot_p if tot_p else 0.0
        f["pool_unseen_p_max"] = max(mp.values(), default=0.0)

        def sample_cooc():
            pool_c, w, extra = list(cands), [probs[c] for c in cands], []
            for _ in range(min(n_unseen, len(pool_c))):
                i = rng.choices(range(len(pool_c)), weights=w)[0]
                extra.append(pool_c.pop(i))
                w.pop(i)
            return extra, {}

        def sample_pool():
            if not teams or rng.random() > POOL_LAMBDA:
                return sample_cooc()
            t, _ = rng.choices(teams, weights=[w for _, w in teams])[0]
            missing = [c for c in t if c not in st.hp["p2"] and c in dex]
            extra = rng.sample(missing, min(n_unseen, len(missing)))
            tm = _W["ctx"]["team_moves"].get(frozenset(t), {})
            return extra, {c: tm[c] for c in extra if c in tm}

        for tag, sampler in (("full", sample_cooc), ("pool", sample_pool)):
            if n_unseen and cands:
                scen = []
                for _ in range(K_SCENARIOS):
                    extra, virt = sampler()
                    st.virtual = virt
                    hp2 = {"p1": hp["p1"], "p2": {**hp["p2"], **{c: 1.0 for c in extra}}}
                    scen.append(team_scores(st, A1, A2 + extra, hp2, cache))
                st.virtual = {}
                keys = scen[0].keys()
            else:
                scen, keys = [seen], seen.keys()
            for k in keys:
                v = np.array([s[k] for s in scen], dtype=float)
                f[f"strat_{tag}_{k}_mean"] = v.mean()
                f[f"strat_{tag}_{k}_std"] = v.std()
    return f


def build_team_moves(battles):
    """Revealed species moves grouped by complete team: (team, species) -> moves.
    This is the moveset part of the joint distribution: recurring pool teams
    tend to reuse the same sets."""
    out = defaultdict(lambda: defaultdict(dict))
    for b in battles:
        teams = {"p1": frozenset(p["name"] for p in b["p1_team_details"])}
        s2 = seen_order(b)
        if len(s2) == 6:
            teams["p2"] = frozenset(s2)
        for t in b["battle_timeline"]:
            for side, team in teams.items():
                m = t[side + "_move_details"]
                if m:
                    out[team][t[side + "_pokemon_state"]["name"]][m["name"]] = m
    return {t: dict(v) for t, v in out.items()}


def _init(dex, chart, ctx):
    _W.update(dex=dex, chart=chart, ctx=ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    configure(args.n_jobs)
    print(describe())
    t0 = time.time()
    train = load_jsonl(os.path.join(DATA, "train.jsonl"))
    test = load_jsonl(os.path.join(DATA, "test.jsonl"))
    if args.limit:
        train, test = train[:args.limit], []
    chart = load_type_chart(os.path.join(DATA, "type_chart_gen2.csv"))
    full = load_jsonl(os.path.join(DATA, "train.jsonl")) + load_jsonl(os.path.join(DATA, "test.jsonl"))
    dex = build_pokedex(full)
    ctx = build_context(full, dex, chart)  # Same label-free context as the base features
    ctx["pool"] = TeamPool(full)
    ctx["team_moves"] = build_team_moves(full)
    snaps = [{**b, "battle_timeline": b["battle_timeline"][:t]} for t in SNAP_TURNS for b in train]
    battles = train + test + snaps
    rows = pool_map(battle_row, battles, _init, (dex, chart, ctx), chunksize=100)
    df = pd.DataFrame(rows)
    df.insert(0, "battle_id", [b["battle_id"] for b in battles])
    df.insert(1, "is_test", [0] * len(train) + [1] * len(test) + [0] * len(snaps))
    df.insert(2, "snap_turn", [30] * (len(train) + len(test)) + [t for t in SNAP_TURNS for _ in train])
    print(f"{len(df)} righe, {df.shape[1] - 3} feature, {time.time() - t0:.0f}s")
    y = np.array([b["player_won"] for b in train])
    from sklearn.metrics import roc_auc_score
    m = (df.is_test.values == 0) & (df.snap_turn.values == 30)
    aucs = {c: roc_auc_score(y, df.loc[m, c].fillna(0)) for c in df.columns[3:]}
    for c, a in sorted(aucs.items(), key=lambda kv: -abs(kv[1] - 0.5))[:20]:
        print(f"  AUC {c:45s} {a:.3f}")
    if not args.limit:
        df.to_pickle(os.path.join(OUT, "strat_features.pkl"))


if __name__ == "__main__":
    main()
