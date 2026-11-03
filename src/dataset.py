"""Paths, reference context and the battle-state feature cache.

Everything learned from the data (pokédex, Hidden Power types, co-occurrence, move statistics,
bag-of-words vocabulary) comes from the training battles only; the test battles are only transformed.
"""
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, os.path.dirname(__file__))
from features import build_context, build_frame, build_pokedex, load_jsonl, load_type_chart

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.environ.get("POKEMON_DATA", os.path.join(ROOT, "data"))  # on Kaggle: /kaggle/input/...
OUT = os.environ.get("POKEMON_OUT", os.path.join(ROOT, "outputs"))
SEED = 42
SNAP_TURNS = (24, 26, 28)  # training battles truncated at these turns are used for augmentation


def load_battles():
    train = load_jsonl(os.path.join(DATA, "train.jsonl"))
    test = load_jsonl(os.path.join(DATA, "test.jsonl"))
    chart = load_type_chart(os.path.join(DATA, "type_chart_gen2.csv"))
    return train, test, chart


def build_reference(ref, chart):
    """Pokédex and context learned from the reference battles only: the whole training set for the
    submissions, the training fold inside cross-validation."""
    dex = build_pokedex(ref)
    return dex, build_context(ref, dex, chart)


def get_features():
    """Battle-state features at the last observed turn for train + test (snap_turn = 30) and for the
    training battles truncated at SNAP_TURNS. Cached in outputs/features.pkl."""
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "features.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    train, test, chart = load_battles()
    dex, ctx = build_reference(train, chart)
    df = build_frame(train + test, dex, chart, ctx)
    df["is_test"] = [0] * len(train) + [1] * len(test)
    df["player_won"] = [b["player_won"] for b in train] + [-1] * len(test)
    df["snap_turn"] = 30
    parts = [df]
    for t in SNAP_TURNS:
        s = build_frame(train, dex, chart, ctx, truncate=t)
        s["is_test"], s["player_won"], s["snap_turn"] = 0, [b["player_won"] for b in train], t
        parts.append(s)
    df = pd.concat(parts, ignore_index=True)
    df.to_pickle(cache)
    return df


def add_bags(df, min_df=20, fit_mask=None):
    """Bag-of-Pokémon / bag-of-moves count columns. The vocabulary is learned from the fit_mask rows
    only (default: the training rows, is_test == 0); the other rows are only transformed."""
    if fit_mask is None:
        fit_mask = df.is_test.values == 0
    parts = [df]
    for col in [c for c in df.columns if c.startswith("_")]:
        cv = CountVectorizer(token_pattern=r"\S+", min_df=min_df).fit(df.loc[fit_mask, col])
        m = cv.transform(df[col]).toarray()
        parts.append(pd.DataFrame(m, columns=[f"{col[1:]}__{v}" for v in cv.get_feature_names_out()], index=df.index))
    out = pd.concat(parts, axis=1).drop(columns=[c for c in df.columns if c.startswith("_")])
    for c in ["p1_active_status", "p2_active_status"]:
        out = pd.concat([out.drop(columns=c), pd.get_dummies(out[c], prefix=c, dtype=int)], axis=1)
    return out
