"""Feature engineering per la predizione delle battaglie Pokemon Gen 2 OU.

Ogni battaglia contiene il team completo di p1, il lead di p2 e i primi 30 turni.
Le feature descrivono lo stato della battaglia al turno 30 (e in alcuni checkpoint
intermedi), la composizione dei team e lo "stile" di gioco osservato.
"""
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from compute import pool_map

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
BAD_STATUS = ["slp", "par", "tox", "psn", "frz", "brn"]
EFFECTS = ["trapped", "confusion", "substitute", "leechseed", "perish", "protect", "attract", "encore"]
BOOSTS = ["atk", "def", "spa", "spd", "spe"]
CHECKPOINTS = [5, 10, 15, 20, 25]
HAZARD_SETUP = {"spikes"}
HEAL_MOVES = {"rest", "recover", "softboiled", "milkdrink", "moonlight", "morningsun", "synthesis"}
SETUP_MOVES = {"curse", "swordsdance", "growth", "agility", "amnesia", "bellydrum", "meditate", "sharpen",
               "doubleteam", "minimize", "barrier", "acidarmor", "harden", "withdraw", "defensecurl"}
PHAZE_MOVES = {"roar", "whirlwind"}
SLEEP_MOVES = {"lovelykiss", "sleeppowder", "hypnosis", "sing", "spore"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_actual_move(prev_t, t, side, last_move):
    """Mossa realmente eseguita da `side` nel turno t, oppure None.

    Il campo *_move_details riporta l'ultima mossa usata da quel pokemon anche nei turni
    in cui non agisce (circa il 25% delle mosse nel dataset). Va scartato quando:
    - il pokemon è appena entrato (switch, phazing, rimpiazzo dopo un KO);
    - il turno è di rimpiazzo dopo un KO e la mossa ripete quella del turno prima;
    - il pokemon era e resta addormentato (tranne sleeptalk/snore, rest con recupero
      di HP, o mossa diversa dall'ultima, cioè chiamata da sleeptalk);
    - il pokemon era e resta congelato.
    `last_move` è l'ultima mossa reale nota di quel pokemon.
    """
    m = t[side + "_move_details"]
    if m is None or prev_t is None:
        return m
    other = "p2" if side == "p1" else "p1"
    ps, pp = t[side + "_pokemon_state"], prev_t[side + "_pokemon_state"]
    if ps["name"] != pp["name"]:
        return None
    po, co = prev_t[other + "_pokemon_state"], t[other + "_pokemon_state"]
    pm = prev_t[side + "_move_details"]
    if po["hp_pct"] == 0 and co["name"] != po["name"] and pm is not None and pm["name"] == m["name"]:
        return None
    if pp["status"] == "slp" and ps["status"] == "slp":
        if m["name"] in ("sleeptalk", "snore"):
            return m
        if m["name"] == "rest" and ps["hp_pct"] > pp["hp_pct"]:
            return m
        if m["name"] == last_move:
            return None
    if pp["status"] == "frz" and ps["status"] == "frz":
        return None
    return m


def clean_battles(battles):
    """Copia delle battaglie con le mosse persistenti (non eseguite) messe a None."""
    out = []
    for b in battles:
        tl = []
        last = {"p1": {}, "p2": {}}
        prev = None
        for t in b["battle_timeline"]:
            t2 = dict(t)
            for side in ("p1", "p2"):
                name = t[side + "_pokemon_state"]["name"]
                m = get_actual_move(prev, t, side, last[side].get(name))
                t2[side + "_move_details"] = m
                if m is not None:
                    last[side][name] = m["name"]
            tl.append(t2)
            prev = t  # confronto con il turno grezzo (serve la mossa riportata, non quella pulita)
        out.append({**b, "battle_timeline": tl})
    return out


def load_type_chart(path):
    df = pd.read_csv(path)
    chart = {}  # dict semplice (picklabile per il multiprocessing spawn); assente = 1.0
    for a, d, m in df.itertuples(index=False):
        chart[(a, d)] = m
    return chart


def build_pokedex(battles):
    """Base stats e tipi per ogni specie, ricavati dai team p1 e dai lead p2."""
    dex = {}
    for b in battles:
        for p in b["p1_team_details"] + [b["p2_lead_details"]]:
            if p["name"] not in dex:
                dex[p["name"]] = {"types": [t for t in p["types"] if t != "notype"],
                                  **{s: p["base_" + s] for s in STATS}}
    return dex


def type_mult(chart, atk_type, def_types):
    m = 1.0
    for t in def_types:
        m *= chart.get((atk_type, t), 1.0)
    return m


class SideState:
    """Stato noto di un lato del campo: hp e status per ogni pokemon visto."""

    def __init__(self):
        self.hp = {}
        self.status = {}
        self.order = []

    def update(self, s):
        name = s["name"]
        if name not in self.hp:
            self.order.append(name)
        self.hp[name] = s["hp_pct"]
        self.status[name] = s["status"]

    def summary(self, team_size=6):
        hps = list(self.hp.values())
        n_seen = len(hps)
        fainted = sum(1 for n in self.hp if self.status[n] == "fnt" or self.hp[n] == 0)
        unseen = max(team_size - n_seen, 0)
        out = {
            "n_seen": n_seen,
            "fainted": fainted,
            "alive": team_size - fainted,
            "hp_sum": sum(hps) + unseen,
            "hp_seen_sum": sum(hps),
            "hp_alive_mean": np.mean([h for h in hps if h > 0]) if any(h > 0 for h in hps) else 0.0,
            "n_low_hp": sum(1 for h in hps if 0 < h < 0.3),
        }
        st = Counter(self.status[n] for n in self.hp if self.hp[n] > 0)
        for s in BAD_STATUS:
            out["n_" + s] = st.get(s, 0)
        out["n_statused"] = sum(st.get(s, 0) for s in BAD_STATUS)
        # pokemon "utili": vivi e non addormentati/congelati
        out["n_healthy"] = out["alive"] - st.get("slp", 0) - st.get("frz", 0)
        out["eff_hp"] = sum(h * (0.5 if self.status[n] in ("slp", "frz") else 0.8 if self.status[n] in ("par", "tox") else 1.0)
                            for n, h in self.hp.items()) + unseen
        return out


def battle_features(b, dex, chart):
    f = {"battle_id": b["battle_id"]}
    tl = b["battle_timeline"]
    p1_team = b["p1_team_details"]
    p1_names = [p["name"] for p in p1_team]

    # ---- composizione team p1 (statica) ----
    for s in STATS:
        vals = [p["base_" + s] for p in p1_team]
        f[f"p1_team_{s}_mean"] = np.mean(vals)
        f[f"p1_team_{s}_max"] = np.max(vals)
    f["p1_team_bst_mean"] = np.mean([sum(p["base_" + s] for s in STATS) for p in p1_team])
    f["p1_team_level_mean"] = np.mean([p["level"] for p in p1_team])
    f["p1_team_level_min"] = np.min([p["level"] for p in p1_team])
    f["p1_n_leftovers"] = sum(p["item"] == "leftovers" for p in p1_team)
    f["p1_n_unique_types"] = len({t for p in p1_team for t in p["types"] if t != "notype"})

    lead2 = b["p2_lead_details"]
    for s in STATS:
        f[f"p2_lead_{s}"] = lead2["base_" + s]
    f["lead_spe_diff"] = p1_team[0]["base_spe"] - lead2["base_spe"]

    # ---- simulazione dello stato lungo la timeline ----
    s1, s2 = SideState(), SideState()
    moves = {"p1": Counter(), "p2": Counter()}
    cats = {"p1": Counter(), "p2": Counter()}
    switches = {"p1": 0, "p2": 0}
    no_move = {"p1": 0, "p2": 0}
    dmg_dealt = {"p1": 0.0, "p2": 0.0}
    se_moves = {"p1": 0, "p2": 0}
    kos_turn = {"p1": [], "p2": []}
    hp_diff_traj, faint_diff_traj = [], []
    prev = {"p1": None, "p2": None}
    for t in tl:
        for side, st in (("p1", s1), ("p2", s2)):
            ps = t[side + "_pokemon_state"]
            name = ps["name"]
            if prev[side] is not None and prev[side] != name:
                switches[side] += 1
            prev[side] = name
            # danno subito da questo lato -> danno inflitto dall'avversario
            other = "p2" if side == "p1" else "p1"
            if name in st.hp:
                delta = st.hp[name] - ps["hp_pct"]
                if delta > 0:
                    dmg_dealt[other] += delta
            was_alive = st.hp.get(name, 1.0) > 0
            st.update(ps)
            if was_alive and ps["hp_pct"] == 0:
                kos_turn[other].append(t["turn"])
            m = t[side + "_move_details"]
            if m is None:
                no_move[side] += 1
            else:
                moves[side][m["name"]] += 1
                cats[side][m["category"]] += 1
                opp = t[other + "_pokemon_state"]["name"]
                if m["category"] != "status" and opp in dex:
                    if type_mult(chart, m["type"], dex[opp]["types"]) > 1:
                        se_moves[side] += 1
        a, c = s1.summary(), s2.summary()
        hp_diff_traj.append(a["hp_sum"] - c["hp_sum"])
        faint_diff_traj.append(a["fainted"] - c["fainted"])
    # checkpoint relativi all'ultimo turno (al turno 30: t25, t20, ... t5)
    for k in CHECKPOINTS:
        i = max(-1 - k, -len(hp_diff_traj))
        f[f"hp_diff_t{30 - k}"] = hp_diff_traj[i]
        f[f"faint_diff_t{30 - k}"] = faint_diff_traj[i]

    a, c = s1.summary(), s2.summary()
    for k, v in a.items():
        f["p1_" + k] = v
    for k, v in c.items():
        f["p2_" + k] = v
        f["diff_" + k] = a[k] - v
    f["hp_diff_slope"] = np.polyfit(np.arange(len(hp_diff_traj)), hp_diff_traj, 1)[0]
    f["hp_diff_last10_change"] = hp_diff_traj[-1] - hp_diff_traj[-11]
    f["hp_diff_min"] = min(hp_diff_traj)
    f["hp_diff_max"] = max(hp_diff_traj)
    f["hp_diff_mean"] = np.mean(hp_diff_traj)

    # p1 ha HP/status anche per i pokemon non ancora entrati (pieni)
    p1_unseen = [n for n in p1_names if n not in s1.hp]
    f["p1_n_unseen"] = len(p1_unseen)
    # stats dei pokemon vivi
    for side, st, extra in (("p1", s1, p1_unseen), ("p2", s2, [])):
        alive = [n for n, h in st.hp.items() if h > 0] + extra
        alive = [n for n in alive if n in dex]
        for s in ["atk", "spa", "spe", "def", "spd", "hp"]:
            f[f"{side}_alive_{s}_sum"] = sum(dex[n][s] for n in alive)
        f[f"{side}_alive_hpw_bst"] = sum(st.hp.get(n, 1.0) * sum(dex[n][s] for s in STATS) for n in alive)
    f["diff_alive_hpw_bst"] = f["p1_alive_hpw_bst"] - f["p2_alive_hpw_bst"]
    for s in ["atk", "spa", "spe"]:
        f[f"diff_alive_{s}_sum"] = f[f"p1_alive_{s}_sum"] - f[f"p2_alive_{s}_sum"]

    # ---- stato attivo al turno 30 ----
    last = tl[-1]
    act = {}
    for side in ("p1", "p2"):
        ps = last[side + "_pokemon_state"]
        act[side] = ps["name"]
        f[f"{side}_active_hp"] = ps["hp_pct"]
        f[f"{side}_active_status"] = ps["status"]
        for bst in BOOSTS:
            f[f"{side}_active_boost_{bst}"] = ps["boosts"][bst]
        f[f"{side}_active_boost_sum"] = sum(ps["boosts"].values())
        for e in EFFECTS:
            f[f"{side}_active_eff_{e}"] = int(e in ps["effects"])
        d = dex.get(ps["name"])
        for s in STATS:
            f[f"{side}_active_{s}"] = d[s] if d else np.nan
    for bst in BOOSTS:
        f[f"diff_active_boost_{bst}"] = f[f"p1_active_boost_{bst}"] - f[f"p2_active_boost_{bst}"]
    f["active_spe_diff"] = f["p1_active_spe"] - f["p2_active_spe"]
    f["p1_active_status"] = f["p1_active_status"]
    # efficacia di tipo tra gli attivi (STAB)
    d1, d2 = dex.get(act["p1"]), dex.get(act["p2"])
    if d1 and d2:
        f["p1_vs_p2_type"] = max(type_mult(chart, t, d2["types"]) for t in d1["types"])
        f["p2_vs_p1_type"] = max(type_mult(chart, t, d1["types"]) for t in d2["types"])
    else:
        f["p1_vs_p2_type"] = f["p2_vs_p1_type"] = np.nan

    # ---- matchup team: tipi STAB di p1 vivi contro p2 vivi (e viceversa) ----
    p1_alive = [n for n in p1_names if s1.hp.get(n, 1.0) > 0 and n in dex]
    p2_alive = [n for n, h in s2.hp.items() if h > 0 and n in dex]
    def best_cover(att, dfn):
        if not att or not dfn:
            return 0.0
        return np.mean([max(type_mult(chart, t, dex[d]["types"]) for a in att for t in dex[a]["types"]) for d in dfn])
    f["p1_cover_p2"] = best_cover(p1_alive, p2_alive)
    f["p2_cover_p1"] = best_cover(p2_alive, p1_alive)

    # ---- stile di gioco / mosse ----
    for side in ("p1", "p2"):
        mv = moves[side]
        f[f"{side}_switches"] = switches[side]
        f[f"{side}_no_move"] = no_move[side]
        f[f"{side}_dmg_dealt"] = dmg_dealt[side]
        f[f"{side}_se_moves"] = se_moves[side]
        f[f"{side}_n_kos"] = len(kos_turn[side])
        f[f"{side}_first_ko_turn"] = kos_turn[side][0] if kos_turn[side] else 31
        f[f"{side}_last_ko_turn"] = kos_turn[side][-1] if kos_turn[side] else 0
        for cat in ("physical", "special", "status"):
            f[f"{side}_cat_{cat}"] = cats[side][cat]
        f[f"{side}_spikes"] = int(mv["spikes"] > 0)
        f[f"{side}_rapidspin"] = mv["rapidspin"]
        f[f"{side}_heal"] = sum(mv[m] for m in HEAL_MOVES)
        f[f"{side}_setup"] = sum(mv[m] for m in SETUP_MOVES)
        f[f"{side}_phaze"] = sum(mv[m] for m in PHAZE_MOVES)
        f[f"{side}_sleepmove"] = sum(mv[m] for m in SLEEP_MOVES)
        f[f"{side}_explosion"] = mv["explosion"] + mv["selfdestruct"]
        f[f"{side}_n_unique_moves"] = len(mv)
    f["diff_dmg_dealt"] = f["p1_dmg_dealt"] - f["p2_dmg_dealt"]
    f["diff_switches"] = f["p1_switches"] - f["p2_switches"]
    # spikes attive: piazzate e non rimosse dopo l'ultimo rapid spin (approssimazione)
    f["spikes_on_p2"] = int(moves["p1"]["spikes"] > 0 and moves["p2"]["rapidspin"] == 0)
    f["spikes_on_p1"] = int(moves["p2"]["spikes"] > 0 and moves["p1"]["rapidspin"] == 0)
    f["diff_spikes"] = f["spikes_on_p2"] - f["spikes_on_p1"]

    # bag-of-pokemon (testo, vettorizzato dopo)
    f["_p1_team"] = " ".join(p1_names)
    f["_p1_alive"] = " ".join(p1_alive)
    f["_p2_seen"] = " ".join(s2.hp.keys())
    f["_p2_alive"] = " ".join(p2_alive)
    f["_p1_moves"] = " ".join(m for m, c in moves["p1"].items() for _ in range(c))
    f["_p2_moves"] = " ".join(m for m, c in moves["p2"].items() for _ in range(c))
    return f


SPECIAL_TYPES = {"fire", "water", "electric", "grass", "ice", "psychic", "dragon", "dark"}
FIXED_POWER = {"hiddenpower": 70, "return": 102, "frustration": 102, "present": 40, "magnitude": 70, "reversal": 60}
FIXED_DAMAGE = {"seismictoss", "nightshade"}  # danno = livello (100 HP)


def calc_stat(base, level, hp=False):
    """Stat Gen 2 con DV e stat exp massimi."""
    core = ((base + 15) * 2 + 63) * level // 100
    return core + level + 10 if hp else core + 5


HP_TYPES = ["fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel",
            "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark"]


def infer_hp_types(battles, dex, chart, min_obs=5):
    """Tipo di hiddenpower per specie, stimato dai danni osservati.

    Per ogni uso di hiddenpower in cui il difensore resta in campo, il rapporto tra
    danno osservato e danno neutro previsto stima il moltiplicatore di tipo; si sceglie
    il tipo che spiega meglio tutte le osservazioni della specie.
    """
    obs = defaultdict(list)  # specie -> [(danno osservato, difensore)]
    hp_move = {"name": "hiddenpower", "type": "normal", "category": "special", "base_power": 70, "accuracy": 1.0}
    for b in battles:
        tl = b["battle_timeline"]
        for prev, t in zip(tl, tl[1:]):
            for side, other in (("p1", "p2"), ("p2", "p1")):
                m = t[side + "_move_details"]
                if not m or m["name"] != "hiddenpower":
                    continue
                att, dfn = t[side + "_pokemon_state"]["name"], t[other + "_pokemon_state"]["name"]
                if (dfn != prev[other + "_pokemon_state"]["name"] or att != prev[side + "_pokemon_state"]["name"]
                        or att not in dex or dfn not in dex):
                    continue
                delta = prev[other + "_pokemon_state"]["hp_pct"] - t[other + "_pokemon_state"]["hp_pct"]
                if not 0.03 < delta < 1:
                    continue
                obs[att].append((delta, dfn))
    out = {}
    for sp, o in obs.items():
        if len(o) < min_obs:
            continue
        def loss(ty):
            return sum((np.log(delta) - np.log(max(damage_frac(dex, chart, sp, dfn, hp_move, hp_type=ty) / 0.925, 0.01))) ** 2
                       for delta, dfn in o)
        out[sp] = min(HP_TYPES, key=loss)
    return out


def build_cooccurrence(battles):
    """P(b nel team | a nel team) stimata dai team completi di p1."""
    pair, single = Counter(), Counter()
    for b in battles:
        names = sorted({p["name"] for p in b["p1_team_details"]})  # ordine deterministico
        for a in names:
            single[a] += 1
            for c in names:
                if a != c:
                    pair[(a, c)] += 1
    cond = defaultdict(dict)
    for (a, c), n in pair.items():
        cond[a][c] = n / single[a]
    return cond, single


def damage_frac(dex, chart, att, dfn, move, att_level=100, dfn_level=100, item=None, hp_type=None):
    """Danno atteso di `move` di att su dfn, come frazione degli HP massimi di dfn."""
    a, d = dex[att], dex[dfn]
    max_hp = calc_stat(d["hp"], dfn_level, hp=True)
    name, mtype = move["name"], move["type"]
    hp_unknown = name == "hiddenpower" and hp_type is None
    if name == "hiddenpower" and hp_type:
        mtype = hp_type
    if name in FIXED_DAMAGE:
        return att_level / max_hp * (0.0 if type_mult(chart, mtype, d["types"]) == 0 else 1.0)
    bp = move["base_power"] or FIXED_POWER.get(name, 0)
    if bp <= 1 or move["category"] == "status":
        return 0.0
    special = mtype in SPECIAL_TYPES
    A = calc_stat(a["spa" if special else "atk"], att_level)
    D = calc_stat(d["spd" if special else "def"], dfn_level)
    if item == "thickclub" and not special:
        A *= 2
    if item == "lightball" and special:
        A *= 2
    if name in ("explosion", "selfdestruct"):
        D = D / 2
    dmg = ((2 * att_level / 5 + 2) * bp * A / D / 50 + 2)
    stab = 1.5 if mtype in a["types"] else 1.0
    mult = 1.0 if hp_unknown else type_mult(chart, mtype, d["types"])
    return dmg * stab * mult * move["accuracy"] * 0.925 / max_hp  # 0.925 = media del random roll


def stab_moves(dex, name):
    """Mosse STAB generiche per pokemon di cui non conosciamo il set."""
    d = dex[name]
    return [{"name": "stab", "type": t, "category": "special" if t in SPECIAL_TYPES else "physical",
             "base_power": 80, "accuracy": 1.0} for t in d["types"]]


def build_species_moves(battles):
    """Statistiche delle mosse (reali) per specie, su tutte le battaglie.

    Restituisce (top_damaging, freq): le 2 mosse offensive più usate dalla specie, usate
    come fallback quando un pokemon non ha rivelato mosse d'attacco, e la frazione di
    battaglie in cui la specie ha usato ciascuna mossa.
    """
    used = defaultdict(Counter)
    appear = Counter()
    details = {}
    for b in battles:
        seen = defaultdict(set)
        for t in b["battle_timeline"]:
            for side in ("p1", "p2"):
                name = t[side + "_pokemon_state"]["name"]
                seen[name]  # registra la presenza
                m = t[side + "_move_details"]
                if m:
                    seen[name].add(m["name"])
                    details[m["name"]] = m
        for name, mv in seen.items():
            appear[name] += 1
            used[name].update(mv)
    freq = {sp: {m: c / appear[sp] for m, c in cnt.items()} for sp, cnt in used.items()}
    top = {}
    for sp, cnt in used.items():
        # ordinamento deterministico (conteggio, poi nome): most_common rompe i pareggi
        # nell'ordine di inserimento, che dipende dall'hash randomizzato dei set
        dmg = [m for m, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
               if details[m]["category"] != "status" or m in FIXED_DAMAGE]
        top[sp] = [details[m] for m in dmg[:2]]
    return top, freq


def boost_mult(stage):
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


CLERICS = {"blissey", "miltank", "chansey"}
WINCON_SETUP = {"curse", "bellydrum", "swordsdance", "growth"}


def matchup_features(b, dex, chart, hp_types, cooc, species_moves, species_freq):
    """Matrice di danno tra pokemon vivi dei due lati usando le mosse rivelate."""
    tl = b["battle_timeline"]
    p1_items = {p["name"]: p["item"] for p in b["p1_team_details"]}
    p1_level = {p["name"]: p["level"] for p in b["p1_team_details"]}
    known = {"p1": defaultdict(dict), "p2": defaultdict(dict)}
    hp = {"p1": {n: 1.0 for n in p1_items}, "p2": {}}
    status = {"p1": {}, "p2": {}}
    for t in tl:
        for side in ("p1", "p2"):
            ps = t[side + "_pokemon_state"]
            hp[side][ps["name"]] = ps["hp_pct"]
            status[side][ps["name"]] = ps["status"]
            m = t[side + "_move_details"]
            if m:
                known[side][ps["name"]][m["name"]] = m

    def best_move(side, att, dfn):
        """(danno frazionale atteso, mossa speciale?) della mossa migliore di att su dfn."""
        mv = list(known[side][att].values())
        dmg_moves = [m for m in mv if m["category"] != "status" or m["name"] in FIXED_DAMAGE]
        if not dmg_moves:
            # nessuna mossa d'attacco rivelata: le più usate dalla specie, altrimenti STAB generico
            dmg_moves = species_moves.get(att) or stab_moves(dex, att)
        lv_a = p1_level.get(att, 100) if side == "p1" else 100
        lv_d = 100 if side == "p1" else p1_level.get(dfn, 100)
        item = p1_items.get(att) if side == "p1" else None
        best = (0.0, False)
        for m in dmg_moves:
            d = damage_frac(dex, chart, att, dfn, m, lv_a, lv_d, item, hp_types.get(att))
            if d > best[0]:
                mtype = hp_types.get(att, m["type"]) if m["name"] == "hiddenpower" else m["type"]
                best = (d, mtype in SPECIAL_TYPES)
        if status[side].get(att) == "brn" and not best[1]:
            best = (best[0] * 0.5, best[1])
        return best

    def best_dmg(side, att, dfn):
        return best_move(side, att, dfn)[0]

    # distribuzione dei pokemon di p2 non ancora visti (co-occorrenze nei team p1)
    seen2 = [n for n in hp["p2"] if n in dex]
    n_unseen = max(6 - len(hp["p2"]), 0)
    probs = Counter()
    for a in seen2:
        for c, p in cooc.get(a, {}).items():
            if c not in hp["p2"] and c in dex:
                probs[c] += p
    tot = sum(probs.values())
    probs = {c: p / tot for c, p in probs.items()} if tot > 0 else {}

    f = {}
    alive = {s: [n for n, h in hp[s].items() if h > 0 and n in dex] for s in ("p1", "p2")}
    for side, other in (("p1", "p2"), ("p2", "p1")):
        A, D = alive[side], alive[other]
        if not A or not D:
            for k in ["best_dmg_mean", "ohko_frac", "2hko_frac", "fast_ohko_frac", "checked_frac", "threat_max"]:
                f[f"{side}_{k}"] = 0.0
            continue
        M = np.array([[best_dmg(side, a, d) for d in D] for a in A])  # danno frazionale su HP max
        rem = np.array([hp[other][d] for d in D])
        spe_a = np.array([dex[a]["spe"] for a in A])[:, None]
        spe_d = np.array([dex[d]["spe"] for d in D])[None, :]
        para_a = np.array([status[side].get(a) == "par" for a in A])[:, None]
        faster = (spe_a * np.where(para_a, 0.25, 1.0)) > spe_d
        best = M.max(axis=0)
        f[f"{side}_best_dmg_mean"] = best.mean()
        f[f"{side}_ohko_frac"] = (best >= rem).mean()
        f[f"{side}_2hko_frac"] = (best * 2 >= rem).mean()
        f[f"{side}_fast_ohko_frac"] = ((M >= rem[None, :]) & faster).any(axis=0).mean()
        # quanti avversari vengono "controllati": qualcuno li colpisce per >=40%
        f[f"{side}_checked_frac"] = (best >= 0.4).mean()
        f[f"{side}_threat_max"] = M.max()
        htk = np.ceil(rem / np.maximum(best, 0.05))
        n_targets = len(D)
        if side == "p1" and n_unseen and probs:
            # slot virtuali per i pokemon di p2 non visti (pieni), pesati con le co-occorrenze
            exp_htk = sum(p * np.ceil(1 / max(max(best_dmg("p1", a, c) for a in A), 0.05)) for c, p in probs.items())
            htk = np.append(htk, [exp_htk] * n_unseen)
            n_targets += n_unseen
        f[f"{side}_hits_to_kill"] = htk.sum()
        f[f"{side}_hits_to_kill_per_mon"] = htk.sum() / n_targets
    for k in ["best_dmg_mean", "ohko_frac", "2hko_frac", "fast_ohko_frac", "checked_frac", "threat_max",
              "hits_to_kill", "hits_to_kill_per_mon"]:
        f[f"diff_{k}"] = f.get(f"p1_{k}", 0) - f.get(f"p2_{k}", 0)

    # duello attuale degli attivi al turno 30, con boost e paralisi
    last = tl[-1]
    a1, a2 = last["p1_pokemon_state"]["name"], last["p2_pokemon_state"]["name"]
    if a1 in dex and a2 in dex and hp["p1"][a1] > 0 and hp["p2"][a2] > 0:
        b1, b2 = last["p1_pokemon_state"]["boosts"], last["p2_pokemon_state"]["boosts"]
        def boosted(side, att, dfn, ba, bd):
            d, special = best_move(side, att, dfn)
            off, dfs = ("spa", "spd") if special else ("atk", "def")
            return d * boost_mult(ba[off]) / boost_mult(bd[dfs])
        d12, d21 = boosted("p1", a1, a2, b1, b2), boosted("p2", a2, a1, b2, b1)
        if "substitute" in last["p2_pokemon_state"]["effects"]:
            d12 *= 0.75  # il primo colpo va sul substitute
        if "substitute" in last["p1_pokemon_state"]["effects"]:
            d21 *= 0.75
        f["active_p1_dmg"] = d12
        f["active_p2_dmg"] = d21
        f["active_p1_turns_to_ko"] = hp["p2"][a2] / max(d12, 0.01)
        f["active_p2_turns_to_ko"] = hp["p1"][a1] / max(d21, 0.01)
        f["active_duel"] = np.log(f["active_p2_turns_to_ko"]) - np.log(f["active_p1_turns_to_ko"])
        sp1 = dex[a1]["spe"] * boost_mult(b1["spe"]) * (0.25 if status["p1"].get(a1) == "par" else 1.0)
        sp2 = dex[a2]["spe"] * boost_mult(b2["spe"]) * (0.25 if status["p2"].get(a2) == "par" else 1.0)
        f["active_p1_faster"] = float(np.sign(sp1 - sp2))
    else:
        for k in ["active_p1_dmg", "active_p2_dmg", "active_p1_turns_to_ko", "active_p2_turns_to_ko", "active_duel",
                  "active_p1_faster"]:
            f[k] = np.nan

    # ---- spikes cronologici (rapid spin fallisce contro ghost) ----
    spikes = {"p1": False, "p2": False}  # spikes sul lato di ...
    for t in tl:
        for side, other in (("p1", "p2"), ("p2", "p1")):
            m = t[side + "_move_details"]
            if not m:
                continue
            if m["name"] == "spikes":
                spikes[other] = True
            elif m["name"] == "rapidspin":
                opp = t[other + "_pokemon_state"]["name"]
                if "ghost" not in dex.get(opp, {}).get("types", []):
                    spikes[side] = False
    f["spikes_now_p1"] = int(spikes["p1"])
    f["spikes_now_p2"] = int(spikes["p2"])
    f["spikes_now_diff"] = f["spikes_now_p2"] - f["spikes_now_p1"]
    f["p1_has_spinner_alive"] = int(any("rapidspin" in known["p1"][n] for n in alive["p1"]))
    f["p2_has_spinner_alive"] = int(any("rapidspin" in known["p2"][n] for n in alive["p2"]))

    # ---- sonno: da rest (temporaneo) o indotto; turni di sonno "consumati" al turno 30 ----
    # Il contatore del sonno in Gen 2 scende solo quando il pokemon è in campo e prova ad agire.
    for side in ("p1", "p2"):
        rest_sleep, induced_sleep, induced_burned, rest_waking = 0, 0, 0, 0
        last_rest, burned = {}, {}
        berry = 0
        prev = None
        for t in tl:
            ps = t[side + "_pokemon_state"]
            name = ps["name"]
            m = t[side + "_move_details"]
            if m and m["name"] == "rest":
                if ps["status"] == "slp":
                    last_rest[name] = t["turn"]
                    burned[name] = 0
                else:
                    berry += 1  # rest seguito da risveglio immediato: mintberry / miracleberry
            elif ps["status"] == "slp":
                if prev is not None and prev["name"] == name and prev["status"] == "slp":
                    burned[name] = burned.get(name, 0) + 1
                else:
                    burned.setdefault(name, 0)
            else:
                # sveglio: dimentica rest e contatore
                last_rest.pop(name, None)
                burned.pop(name, None)
            if (prev is not None and prev["name"] == name and prev["status"] in ("par", "psn", "tox", "brn", "frz")
                    and ps["status"] == "nostatus" and ps["hp_pct"] > 0 and not (m and m["name"] == "rest")):
                berry += 1  # status curato senza rest: miracleberry / przcureberry ecc.
            prev = ps
        for n, st in status[side].items():
            if st == "slp" and hp[side].get(n, 0) > 0:
                if n in last_rest:
                    rest_sleep += 1
                    rest_waking += int(burned.get(n, 0) >= 2)
                else:
                    induced_sleep += 1
                    induced_burned += burned.get(n, 0)
        f[f"{side}_rest_sleep"] = rest_sleep
        f[f"{side}_rest_waking"] = rest_waking
        f[f"{side}_induced_sleep"] = induced_sleep
        f[f"{side}_induced_sleep_turns"] = induced_burned
        f[f"{side}_berry_used"] = berry
        # mosse note per pokemon vivi (quanto del set è stato rivelato)
        f[f"{side}_known_moves_alive"] = sum(len(known[side][n]) for n in alive[side])
    f["diff_induced_sleep"] = f["p1_induced_sleep"] - f["p2_induced_sleep"]

    # ---- win condition di fine partita: setup non phazabile, cleric ----
    def has_move(side, n, moves, min_freq=0.3):
        if any(m in known[side][n] for m in moves):
            return True
        return sum(species_freq.get(n, {}).get(m, 0) for m in moves) >= min_freq
    for side, other in (("p1", "p2"), ("p2", "p1")):
        setup = any(any(m in known[side][n] for m in WINCON_SETUP) for n in alive[side])
        phazer = any(has_move(other, n, PHAZE_MOVES) for n in alive[other])
        f[f"{side}_setup_alive"] = int(setup)
        f[f"{side}_phazer_alive"] = int(phazer)
        f[f"{side}_unphazed_setup"] = int(setup and not phazer)
        cleric = any(n in CLERICS or "healbell" in known[side][n] for n in alive[side])
        f[f"{side}_cleric_alive"] = int(cleric)
        n_st = sum(1 for n in alive[side] if status[side].get(n) in ("par", "psn", "tox", "brn", "slp", "frz"))
        f[f"{side}_status_burden"] = n_st * (0.3 if cleric else 1.0)
    f["unphazed_setup_adv"] = f["p1_unphazed_setup"] - f["p2_unphazed_setup"]
    f["diff_status_burden"] = f["p1_status_burden"] - f["p2_status_burden"]

    # ---- pokemon di p2 non ancora visti, stimati dalle co-occorrenze nei team p1 ----
    f["p2_n_unseen"] = n_unseen
    if n_unseen and probs:
        for s in STATS:
            f[f"p2_unseen_exp_{s}"] = sum(p * dex[c][s] for c, p in probs.items())
        f["p2_unseen_exp_bst"] = sum(f[f"p2_unseen_exp_{s}"] for s in STATS)
        # quanto i vivi di p1 colpiscono (STAB) e sono colpiti dai probabili non visti
        f["p1_cover_p2_unseen"] = sum(
            p * max((type_mult(chart, t, dex[c]["types"]) for a in alive["p1"] for t in dex[a]["types"]), default=1.0)
            for c, p in probs.items())
        f["p2_unseen_threat"] = sum(
            p * np.mean([max(type_mult(chart, t, dex[a]["types"]) for t in dex[c]["types"]) for a in alive["p1"]] or [1.0])
            for c, p in probs.items())
        f["p2_unseen_p_snorlax"] = probs.get("snorlax", 0.0)
    else:
        for s in STATS + ["bst"]:
            f[f"p2_unseen_exp_{s}"] = 0.0
        f["p1_cover_p2_unseen"] = f["p2_unseen_threat"] = f["p2_unseen_p_snorlax"] = 0.0
    f["p2_unseen_exp_bst_tot"] = f["p2_unseen_exp_bst"] * n_unseen
    return f


def build_context(battles, dex, chart):
    """Statistiche globali (senza etichette) calcolate sulle battaglie già pulite."""
    species_moves, species_freq = build_species_moves(battles)
    return dict(hp_types=infer_hp_types(battles, dex, chart), cooc=build_cooccurrence(battles)[0],
                species_moves=species_moves, species_freq=species_freq)


def symmetrize_unseen(f):
    """Aggiunge a p2 il valore atteso dei pokemon non visti, come già avviene per p1."""
    n = f["p2_n_unseen"]
    for s in ["atk", "spa", "spe", "def", "spd", "hp"]:
        f[f"p2_alive_{s}_sum"] += n * f[f"p2_unseen_exp_{s}"]
    f["p2_alive_hpw_bst"] += n * f["p2_unseen_exp_bst"]
    f["diff_alive_hpw_bst"] = f["p1_alive_hpw_bst"] - f["p2_alive_hpw_bst"]
    for s in ["atk", "spa", "spe"]:
        f[f"diff_alive_{s}_sum"] = f[f"p1_alive_{s}_sum"] - f[f"p2_alive_{s}_sum"]
    return f


def build_frame(battles, dex, chart, ctx=None, truncate=None):
    """Feature per battaglia. Le battaglie devono essere già passate da `clean_battles`.

    `truncate=t` calcola le feature come se la timeline finisse al turno t (augmentation).
    """
    ctx = ctx or build_context(battles, dex, chart)
    if truncate:
        battles = [{**b, "battle_timeline": b["battle_timeline"][:truncate]} for b in battles]
    # le battaglie sono indipendenti: calcolo parallelo (portabile, vedi compute.py)
    return pd.DataFrame(pool_map(_frame_row, battles, _init_frame_worker, (dex, chart, ctx), chunksize=200))


_W = {}


def _init_frame_worker(dex, chart, ctx):
    _W.update(dex=dex, chart=chart, ctx=ctx)


def _frame_row(b):
    dex, chart = _W["dex"], _W["chart"]
    return symmetrize_unseen({**battle_features(b, dex, chart), **matchup_features(b, dex, chart, **_W["ctx"])})
