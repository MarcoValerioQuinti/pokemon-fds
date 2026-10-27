"""Test del motore di danno (features.damage_frac) contro la formula Gen 2 di Pokémon Showdown.

Riferimento: smogon/pokemon-showdown, data/mods/gen2/scripts.ts (getDamage/getStat). La formula
esatta con arrotondamenti interi è reimplementata qui (`showdown_damage`); damage_frac usa la
versione continua con roll medio, quindi i confronti hanno una tolleranza.

Esegui: conda run -n pokemon-fds python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import (SPECIAL_TYPES, boost_mult, calc_stat, damage_frac, load_type_chart,  # noqa: E402
                      type_mult)

CHART = load_type_chart(os.path.join(ROOT, "data", "type_chart_gen2.csv"))
DEX = {  # base stats e tipi come nel dataset
    "snorlax": {"types": ["normal"], "hp": 160, "atk": 110, "def": 65, "spa": 65, "spd": 110, "spe": 30},
    "zapdos": {"types": ["electric", "flying"], "hp": 90, "atk": 90, "def": 85, "spa": 125, "spd": 90, "spe": 100},
    "skarmory": {"types": ["steel", "flying"], "hp": 65, "atk": 80, "def": 140, "spa": 40, "spd": 70, "spe": 70},
    "gengar": {"types": ["ghost", "poison"], "hp": 60, "atk": 65, "def": 60, "spa": 130, "spd": 75, "spe": 110},
    "starmie": {"types": ["water", "psychic"], "hp": 60, "atk": 75, "def": 85, "spa": 100, "spd": 85, "spe": 115},
    "tyranitar": {"types": ["dark", "rock"], "hp": 100, "atk": 134, "def": 110, "spa": 95, "spd": 100, "spe": 61},
    "cloyster": {"types": ["water", "ice"], "hp": 50, "atk": 95, "def": 180, "spa": 85, "spd": 45, "spe": 70},
    "marowak": {"types": ["ground"], "hp": 60, "atk": 80, "def": 110, "spa": 50, "spd": 80, "spe": 45},
}


def mv(name, type_, bp, cat="physical", acc=1.0):
    return {"name": name, "type": type_, "category": cat, "base_power": bp, "accuracy": acc}


def showdown_damage(att, dfn, move, level=100, crit=False, a_boost=0, d_boost=0, item=None, burned=False,
                    screen=False):
    """Formula intera di Showdown Gen 2: lista dei 39 danni possibili (roll 217..255)."""
    a, d = DEX[att], DEX[dfn]
    special = move["type"] in SPECIAL_TYPES
    A = calc_stat(a["spa" if special else "atk"], level)
    D = calc_stat(d["spd" if special else "def"], level)
    if item == "thickclub" and not special:
        A *= 2
    ignore = crit and a_boost <= d_boost  # crit: ignora boost, screen e burn
    if not ignore:
        A = int(A * boost_mult(a_boost))
        D = int(D * boost_mult(d_boost))
        if burned and not special:
            A //= 2
        if screen:
            D *= 2
    A, D = min(max(A, 1), 999), min(max(D, 1), 999)
    if A >= 256 or D >= 256:
        A, D = max((A // 4) % 256, 1), max((D // 4) % 256, 1)
    if move["name"] in ("explosion", "selfdestruct"):
        D = max(D // 2, 1)
    base = (2 * level // 5 + 2) * move["base_power"] * A // D // 50
    if crit:
        base *= 2
    base = min(max(base, 1), 997) + 2
    if move["type"] in a["types"]:
        base += base // 2
    for t in d["types"]:
        m = CHART.get((move["type"], t), 1.0)
        if m == 0:
            return [0] * 39
        base = int(base * m)
    return [base * r // 255 for r in range(217, 256)]


def max_hp(name, level=100):
    return calc_stat(DEX[name]["hp"], level, hp=True)


def mean_frac(att, dfn, move, **kw):
    return np.mean(showdown_damage(att, dfn, move, **kw)) / max_hp(dfn)


def test_stat_formula():
    # Snorlax L100, DV 15, stat exp massima: HP 523, Atk 318 (valori noti del calcolatore Gen 2)
    assert calc_stat(160, 100, hp=True) == 523
    assert calc_stat(110, 100) == 318
    assert calc_stat(30, 100) == 158


@pytest.mark.parametrize("att,dfn,move", [
    ("snorlax", "zapdos", mv("bodyslam", "normal", 85)),
    ("zapdos", "snorlax", mv("thunderbolt", "electric", 95, "special")),
    ("tyranitar", "zapdos", mv("rockslide", "rock", 75)),
    ("starmie", "tyranitar", mv("surf", "water", 95, "special")),
    ("snorlax", "skarmory", mv("earthquake", "ground", 100)),
    ("gengar", "snorlax", mv("thunderbolt", "electric", 95, "special")),
    ("cloyster", "zapdos", mv("explosion", "normal", 250)),
])
def test_damage_matches_showdown(att, dfn, move):
    ours = damage_frac(DEX, CHART, att, dfn, move)
    ref = mean_frac(att, dfn, move)
    assert ref == 0 and ours == 0 or abs(ours / ref - 1) < 0.03, (ours, ref)


def test_immunities():
    assert damage_frac(DEX, CHART, "snorlax", "gengar", mv("bodyslam", "normal", 85)) == 0
    assert damage_frac(DEX, CHART, "snorlax", "skarmory", mv("earthquake", "ground", 100)) == 0
    # seismic toss (fighting) non colpisce i ghost; altrimenti danno = livello
    assert damage_frac(DEX, CHART, "snorlax", "gengar", mv("seismictoss", "fighting", 0)) == 0
    assert damage_frac(DEX, CHART, "snorlax", "zapdos", mv("seismictoss", "fighting", 0)) == pytest.approx(100 / max_hp("zapdos"))
    assert damage_frac(DEX, CHART, "gengar", "snorlax", mv("nightshade", "ghost", 0)) == 0


def test_category_by_type():
    # in Gen 2 la categoria dipende dal tipo: crunch (dark) è speciale anche se il dato dice altro
    ours = damage_frac(DEX, CHART, "tyranitar", "starmie", mv("crunch", "dark", 80, "physical"))
    ref = mean_frac("tyranitar", "starmie", mv("crunch", "dark", 80))
    assert abs(ours / ref - 1) < 0.03


def test_fixed_power_moves():
    # return/frustration a potenza massima 102, hiddenpower 70 col tipo inferito
    ours = damage_frac(DEX, CHART, "snorlax", "zapdos", mv("return", "normal", 0))
    assert abs(ours / mean_frac("snorlax", "zapdos", mv("return", "normal", 102)) - 1) < 0.03
    ours = damage_frac(DEX, CHART, "zapdos", "tyranitar", mv("hiddenpower", "normal", 0), hp_type="ice")
    assert abs(ours / mean_frac("zapdos", "tyranitar", mv("hiddenpower", "ice", 70)) - 1) < 0.03


def test_thick_club():
    move = mv("earthquake", "ground", 100)
    ours = damage_frac(DEX, CHART, "marowak", "snorlax", move, item="thickclub")
    assert abs(ours / mean_frac("marowak", "snorlax", move, item="thickclub") - 1) < 0.03


def test_boost_multipliers():
    # +1 = x1.5, +2 = x2, -1 = x0.66, -2 = x0.5 (Showdown: 100/66/50/40/33/28/25)
    assert [boost_mult(s) for s in (1, 2, 6)] == [1.5, 2.0, 4.0]
    for s, ref in zip(range(-1, -7, -1), (0.66, 0.5, 0.4, 0.33, 0.28, 0.25)):
        assert abs(boost_mult(s) - ref) < 0.01


def test_type_chart_gen2():
    # Gen 2: steel resiste ghost e dark; ghost immune a normal e fighting; ground immune a electric
    assert type_mult(CHART, "ghost", ["steel"]) == 0.5
    assert type_mult(CHART, "dark", ["steel"]) == 0.5
    assert type_mult(CHART, "normal", ["ghost"]) == 0
    assert type_mult(CHART, "electric", ["ground"]) == 0
    assert type_mult(CHART, "ice", ["dragon", "flying"]) == 4


def test_crit_ignores_boosts():
    move = mv("bodyslam", "normal", 85)
    plain = showdown_damage("snorlax", "zapdos", move, crit=True)
    boosted_def = showdown_damage("snorlax", "zapdos", move, crit=True, d_boost=2)
    assert plain == boosted_def
