"""Infer p2's hidden team from a pool of complete teams (Phase 5).

Dataset observations:
  - 15,000 p1 teams contain only 5,607 distinct teams;
  - 75% of fully observed p2 teams (six Pokemon by turn 30) also appear as p1 teams.
Complete p2's missing slots using whole teams compatible with observed Pokemon
(a joint distribution), rather than summing pairwise co-occurrences.

Label-free corpus, transductive over train+test: all p1 teams and fully observed p2 teams.
P(team | observed) is proportional to corpus frequency among teams containing all
observed Pokemon. If none match, back off to one fewer observed member, then co-occurrence.

Usage: python src/team_infer.py
Reconstruction evaluation masks complete p2 teams, retaining the first k Pokemon in
appearance order, and compares pool versus co-occurrence recall and calibration.
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import build_cooccurrence, load_jsonl
from train import DATA


def seen_order(b, side="p2"):
    out = []
    for t in b["battle_timeline"]:
        n = t[side + "_pokemon_state"]["name"]
        if n not in out:
            out.append(n)
    return out


class TeamPool:
    def __init__(self, battles):
        """entries: (battle index, side, team) for every known complete team."""
        self.entries = []
        for i, b in enumerate(battles):
            self.entries.append((i, "p1", frozenset(p["name"] for p in b["p1_team_details"])))
            s = seen_order(b)
            if len(s) == 6:
                self.entries.append((i, "p2", frozenset(s)))
        self.post = defaultdict(set)
        for k, (_, _, t) in enumerate(self.entries):
            for n in t:
                self.post[n].add(k)

    def candidates(self, seen, exclude=None):
        """Return compatible teams weighted by frequency. Exclude (battle, side)
        for leave-one-out reconstruction. Return (Counter team->weight, backoff level)."""
        seen = set(seen)
        for drop in range(0, len(seen)):
            cnt = Counter()
            if drop == 0:
                ids = set.intersection(*(self.post[n] for n in seen)) if seen else set()
            else:
                # Share at least len(seen)-drop observed members.
                hits = Counter(k for n in seen for k in self.post[n])
                ids = {k for k, h in hits.items() if h >= len(seen) - drop}
            for k in ids:
                i, side, t = self.entries[k]
                if exclude is not None and (i, side) == exclude:
                    continue
                cnt[t] += 1
            if cnt:
                return cnt, drop
            if drop >= 1:
                break
        return Counter(), -1

    def member_probs(self, seen, exclude=None):
        """P(species on team | observed) for unseen species; sums to the expected missing slots."""
        cnt, level = self.candidates(seen, exclude)
        tot = sum(cnt.values())
        probs = Counter()
        for t, w in cnt.items():
            for n in t - set(seen):
                probs[n] += w / tot
        n_missing = 6 - len(seen)
        if level > 0:  # Backoff: teams lack some observed members; renormalize over missing slots.
            s = sum(probs.values())
            probs = Counter({n: p * n_missing / s for n, p in probs.items()}) if s else probs
        return dict(probs), level, len(cnt)


def cooc_probs(cooc, seen, n_missing):
    probs = Counter()
    for a in seen:
        for c, p in cooc.get(a, {}).items():
            if c not in seen:
                probs[c] += p
    s = sum(probs.values())
    return {c: p * n_missing / s for c, p in probs.items()} if s else {}


def blend(pp, pc, lam):
    keys = set(pp) | set(pc)
    return {c: lam * pp.get(c, 0) + (1 - lam) * pc.get(c, 0) for c in keys}


def reconstruction_test(battles, ks=(2, 3, 4, 5), lams=(0.5, 0.7, 0.85)):
    """Recall among the first n_missing candidates and Brier score across all species
    (sum of squared errors per battle, comparable across methods)."""
    pool = TeamPool(battles)
    cooc = build_cooccurrence(battles)[0]
    full = [(i, seen_order(b)) for i, b in enumerate(battles) if len(seen_order(b)) == 6]
    print(f"{len(pool.entries)} teams in the corpus, {len(set(t for *_, t in pool.entries))} distinct; "
          f"{len(full)} complete p2 teams used for leave-one-out evaluation")
    for k in ks:
        res, brier = defaultdict(list), defaultdict(list)
        levels = Counter()
        for i, order in full:
            seen, missing = order[:k], set(order[k:])
            pp, level, _ = pool.member_probs(seen, exclude=(i, "p2"))
            levels[level] += 1
            pc = cooc_probs(cooc, seen, 6 - k)
            methods = {"cooc": pc, "pool": pp or pc}
            methods.update({f"blend{l}": blend(pp, pc, l) if pp else pc for l in lams})
            for name, pr in methods.items():
                top = sorted(pr, key=lambda c: -pr[c])[:len(missing)]
                res[name].append(len(missing & set(top)) / len(missing))
                cand = set(pr) | missing
                brier[name].append(sum((min(pr.get(c, 0), 1) - (c in missing)) ** 2 for c in cand))
        print(f"k={k } observed (backoff {dict(sorted(levels.items()))})")
        for name in res:
            print(f"   {name:10s} recall {np.mean(res[name]):.3f} | total Brier {np.mean(brier[name]):.3f}")


if __name__ == "__main__":
    battles = load_jsonl(os.path.join(DATA, "train.jsonl")) + load_jsonl(os.path.join(DATA, "test.jsonl"))
    reconstruction_test(battles)
