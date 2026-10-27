"""Analisi degli errori OOF di un esperimento (outputs/experiments/<nome>/predictions.csv).

Produce in outputs/experiments/<nome>/:
  errors.csv          battaglie sbagliate con probabilità, margine e feature di stato principali
  error_slices.csv    accuracy per fasce delle feature di stato (HP, KO, non visti, status, spikes, boost, duello)
  model_pairs.csv     per ogni coppia di base model: quante volte uno corregge l'altro, correlazione degli errori
  error_report.txt    riepilogo leggibile

Uso: python src/analyze_errors.py --name nested_full
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train import get_features
from evaluate_nested import EXP

KEY = ["diff_hp_sum", "diff_fainted", "p1_fainted", "p2_fainted", "p2_n_unseen", "diff_n_statused",
       "spikes_now_diff", "diff_active_boost_atk", "active_duel", "diff_eff_hp", "hp_diff_slope",
       "p1_rest_sleep", "p2_rest_sleep", "unphazed_setup_adv", "diff_hits_to_kill"]


def slices(df, correct, conf):
    """Accuracy per fasce: quantili per le continue, valori per le discrete."""
    rows = []
    act_boost = df.p1_active_boost_sum - df.p2_active_boost_sum
    spec = {
        "diff_hp_sum": pd.qcut(df.diff_hp_sum, 5, duplicates="drop"),
        "diff_fainted": df.diff_fainted.clip(-3, 2),
        "p2_n_unseen": df.p2_n_unseen.clip(0, 3),
        "n_statused_tot": (df.p1_n_statused + df.p2_n_statused).clip(0, 4),
        "spikes_now_diff": df.spikes_now_diff,
        "active_boost_diff": np.sign(act_boost),
        "active_duel": pd.qcut(df.active_duel.fillna(0), 4, duplicates="drop"),
        "rest_sleep_tot": (df.p1_rest_sleep + df.p2_rest_sleep).clip(0, 2),
        "setup_alive": df.p1_setup_alive * 2 + df.p2_setup_alive,
        "confidence": pd.cut(conf, [0, 0.1, 0.2, 0.3, 0.5], include_lowest=True),
    }
    for name, bins in spec.items():
        g = pd.DataFrame({"bin": bins.astype(str), "ok": correct, "conf": conf}).groupby("bin")
        for b, d in g:
            rows.append({"slice": name, "bin": b, "n": len(d), "acc": d.ok.mean(), "mean_conf": d.conf.mean()})
    return pd.DataFrame(rows)


def model_pairs(pred, y):
    cols = [c for c in pred.columns if c.startswith("p_") and c not in ("p_stack", "p_mean")]
    err = {c[2:]: ((pred[c].values > 0.5) != y) for c in cols}
    rows = []
    for a, b in itertools.combinations(err, 2):
        ea, eb = err[a], err[b]
        rows.append({"a": a, "b": b, "acc_a": 1 - ea.mean(), "acc_b": 1 - eb.mean(),
                     "both_wrong": (ea & eb).mean(), "a_fixes_b": (~ea & eb).mean(), "b_fixes_a": (ea & ~eb).mean(),
                     "err_corr": np.corrcoef(ea, eb)[0, 1]})
    return pd.DataFrame(rows).sort_values("err_corr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    d = os.path.join(EXP, args.name)
    pred = pd.read_csv(os.path.join(d, "predictions.csv"))
    feat = get_features().query("is_test == 0").set_index("battle_id").loc[pred.battle_id].reset_index()
    y = pred.y_true.values
    p = pred.p_stack.values
    correct = (p > 0.5) == y
    conf = np.abs(p - 0.5)

    err = pd.concat([pred[["battle_id", "y_true", "fold", "p_stack"]], feat[KEY]], axis=1)[~correct]
    err.insert(4, "margin", np.abs(err.p_stack - 0.5))
    err.sort_values("margin", ascending=False).to_csv(os.path.join(d, "errors.csv"), index=False)
    sl = slices(feat, correct, conf)
    sl.to_csv(os.path.join(d, "error_slices.csv"), index=False)
    mp = model_pairs(pred, y)
    mp.to_csv(os.path.join(d, "model_pairs.csv"), index=False)

    lines = [f"esperimento {args.name}: accuracy stack {correct.mean():.4f}, errori {(~correct).sum()}"]
    for lo, hi in [(0.3, 0.5), (0.2, 0.3), (0.1, 0.2), (0, 0.1)]:
        m = (conf > lo) & (conf <= hi) if lo else conf <= hi
        lines.append(f"  |p-0.5| in ({lo}, {hi}]: {m.sum():5d} battaglie, accuracy {correct[m].mean():.3f}, "
                     f"{(~correct[m]).sum()} errori")
    # errori sicuri: in quale direzione e con quale stato
    bad = (~correct) & (conf > 0.25)
    lines.append(f"errori con alta confidenza (|p-0.5|>0.25): {bad.sum()}; "
                 f"previsto p1 vince ma perde: {(bad & (p > 0.5)).sum()}, viceversa: {(bad & (p < 0.5)).sum()}")
    cmp = pd.DataFrame({"tutti": feat[KEY].mean(), "errori_sicuri": feat[KEY][bad].mean(),
                        "corretti_sicuri": feat[KEY][correct & (conf > 0.25)].mean()})
    lines.append(cmp.round(3).to_string())
    lines.append("\naccuracy per fascia:")
    lines.append(sl.round(3).to_string(index=False))
    lines.append("\ncoppie di modelli meno correlate negli errori:")
    lines.append(mp.head(10).round(4).to_string(index=False))
    lines.append("coppie più correlate:")
    lines.append(mp.tail(5).round(4).to_string(index=False))
    txt = "\n".join(lines)
    with open(os.path.join(d, "error_report.txt"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
