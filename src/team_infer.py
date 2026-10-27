"""Inferenza del team nascosto di p2 da un pool di team completi (Fase 5).

Scoperta sui dati: i team non sono indipendenti, provengono da un pool ricorrente.
  - 15000 team p1 contengono solo 5607 team distinti;
  - il 75% dei team p2 visti per intero (6 pokemon entro il turno 30) compare anche come team p1.
Quindi il completamento dei posti mancanti di p2 si fa per *team interi* compatibili con i pokemon
visti (distribuzione congiunta), non sommando co-occorrenze a coppie.

Corpus (senza etichette, transduttivo su train+test): team p1 di tutte le battaglie + team p2
visti per intero. P(team | visti) è proporzionale alla frequenza del team nel corpus, tra i team
che contengono tutti i visti; se nessuno li contiene, si scende ai team con un visto in meno, poi
alle co-occorrenze.

Uso: python src/team_infer.py   test di ricostruzione: maschera i team p2 completi (lasciando i primi
k pokemon nell'ordine di apparizione) e confronta pool contro co-occorrenze (recall e calibrazione).
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
        """entries: (indice battaglia, lato, team) per ogni team completo noto."""
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
        """Team compatibili coi visti, pesati per frequenza. exclude=(battaglia, lato) da non usare
        (leave-one-out nel test di ricostruzione). Restituisce (Counter team->peso, livello di backoff)."""
        seen = set(seen)
        for drop in range(0, len(seen)):
            cnt = Counter()
            if drop == 0:
                ids = set.intersection(*(self.post[n] for n in seen)) if seen else set()
            else:
                # almeno len(seen)-drop visti in comune
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
        """P(specie nel team | visti) per le specie non viste; somma = posti mancanti attesi."""
        cnt, level = self.candidates(seen, exclude)
        tot = sum(cnt.values())
        probs = Counter()
        for t, w in cnt.items():
            for n in t - set(seen):
                probs[n] += w / tot
        n_missing = 6 - len(seen)
        if level > 0:  # backoff: i team non contengono tutti i visti, rinormalizza ai posti mancanti
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
    """Recall dei mancanti nelle prime n_mancanti candidate e Brier su tutte le specie
    (somma degli errori quadratici per battaglia: confrontabile tra metodi)."""
    pool = TeamPool(battles)
    cooc = build_cooccurrence(battles)[0]
    full = [(i, seen_order(b)) for i, b in enumerate(battles) if len(seen_order(b)) == 6]
    print(f"{len(pool.entries)} team nel corpus, {len(set(t for *_, t in pool.entries))} distinti; "
          f"{len(full)} team p2 completi usati per il test (leave-one-out)")
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
        print(f"k={k} visti (backoff {dict(sorted(levels.items()))})")
        for name in res:
            print(f"   {name:10s} recall {np.mean(res[name]):.3f} | Brier totale {np.mean(brier[name]):.3f}")


if __name__ == "__main__":
    battles = load_jsonl(os.path.join(DATA, "train.jsonl")) + load_jsonl(os.path.join(DATA, "test.jsonl"))
    reconstruction_test(battles)
