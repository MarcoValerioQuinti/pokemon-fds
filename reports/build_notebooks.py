"""Genera i due notebook di report (dati/significatività e costi/benefici dei metodi).

Uso: python reports/build_notebooks.py, poi
     jupyter nbconvert --to notebook --execute --inplace reports/*.ipynb
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (pokemon-fds)", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c.strip("\n"))
                for c in cells]
    return nb


SETUP = r'''
import os, sys, json, hashlib, warnings
from collections import Counter
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logit
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "reports" else os.getcwd()
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
RNG = np.random.default_rng(0)

def wilson(k, n, z=1.96):
    """Intervallo di Wilson al 95% per una proporzione."""
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h

def mcnemar(a_ok, b_ok):
    """Test di McNemar esatto su due vettori di correttezza appaiati."""
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p

def paired_bootstrap(a_ok, b_ok, n=5000):
    """IC al 95% della differenza di accuracy (b - a) ricampionando le battaglie."""
    d = b_ok.astype(float) - a_ok.astype(float)
    idx = RNG.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
'''

# --------------------------------------------------------------------------------------------
# Report 1: dati e significatività
# --------------------------------------------------------------------------------------------
R1 = [
r'''md:# Report statistico 1: i dati e la significatività dei risultati

Questo notebook descrive i dati della competizione (battaglie Pokémon Gen 2 OU, si predice `player_won` dallo stato al turno 30) e misura **quanto è significativo** ciò che vediamo: lo sbilanciamento delle classi, le relazioni tra stato e vittoria, il potere predittivo delle singole feature, lo spostamento tra train e test e infine l'incertezza sull'accuracy dei modelli.

Usa le cache già calcolate (`outputs/features.pkl`, `outputs/oof_mc_rec/`, `outputs/experiments/`): non riaddestra lo stack. Tutti i test sono bilaterali con α = 0.05; quando si testano molte ipotesi insieme si controlla il *false discovery rate* con Benjamini-Hochberg.''',
SETUP,
r'''
def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]
train = load_jsonl("data/train.jsonl"); test = load_jsonl("data/test.jsonl")
F = pd.read_pickle("outputs/features.pkl")
tr = F[F.is_test == 0].reset_index(drop=True); te = F[F.is_test == 1].reset_index(drop=True)
y = tr.player_won.values.astype(int)
assert (tr.battle_id.values == np.array([b["battle_id"] for b in train])).all()
print(f"train {len(train)} battaglie, test {len(test)}, feature base {F.shape[1] - 3}")
''',
r'''md:## 1. Bilanciamento delle classi

Se le classi fossero sbilanciate, l'accuracy andrebbe confrontata con la classe maggioritaria e non con 0.5. Test binomiale esatto contro p = 0.5.''',
r'''
k = y.sum(); n = len(y)
lo, hi = wilson(k, n)
print(f"p1 vince {k}/{n} = {k/n:.4f}  IC95% [{lo:.4f}, {hi:.4f}]  binomiale p = {stats.binomtest(k, n, .5).pvalue:.3f}")
''',
r'''md:**Lettura:** le classi sono bilanciate (nessuna differenza significativa da 50%). Il riferimento "a caso" per l'accuracy è quindi 0.50 e la soglia 0.5 delle probabilità non va spostata (lo conferma l'esperimento sulla soglia a mediana, senza effetto).''',
r'''md:## 2. Struttura delle battaglie

Lunghezza della timeline, quanti pokemon di p2 sono stati visti, quanti KO per lato al turno 30. Il confronto dei KO tra p1 e p2 è appaiato (stessa battaglia), quindi si usa il test di Wilcoxon per ranghi con segno.''',
r'''
lens = Counter(len(b["battle_timeline"]) for b in train + test)
print("lunghezza timeline:", dict(lens))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for d, lab in ((tr, "train"), (te, "test")):
    ax[0].hist(d.p2_n_seen, bins=np.arange(.5, 7.5), alpha=.6, density=True, label=lab)
ax[0].set(title="Pokemon di p2 visti al turno 30", xlabel="visti"); ax[0].legend()
ax[1].hist([tr.p1_fainted, tr.p2_fainted], bins=np.arange(-.5, 7.5), label=["p1", "p2"])
ax[1].set(title="KO subiti al turno 30 (train)", xlabel="pokemon svenuti"); ax[1].legend()
ax[2].hist(tr.diff_hp_sum, bins=60, color="gray"); ax[2].set(title="diff_hp_sum (p1 - p2)", xlabel="HP totali")
plt.tight_layout(); plt.show()
w = stats.wilcoxon(tr.p1_fainted, tr.p2_fainted)
print(f"KO medi al turno 30: p1 {tr.p1_fainted.mean():.2f}, p2 {tr.p2_fainted.mean():.2f}; Wilcoxon p = {w.pvalue:.1e}")
print(f"ma p1 vince il {y.mean():.1%}: la differenza di KO al turno 30 da sola non decide la partita")
''',
r'''md:**Lettura:** le timeline hanno sempre 30 turni, quindi la lunghezza non rivela l'esito. p1 subisce molti più KO di p2 nei primi 30 turni (differenza altamente significativa) eppure vince metà delle partite: le dinamiche dopo il turno 30 sono diverse da quelle osservabili, ed è questo il limite strutturale del problema.''',
r'''md:## 3. Duplicati e pool di team

Battaglie con timeline identica e quanto sono ripetuti i team. Queste proprietà condizionano la validazione (i duplicati vanno tenuti nello stesso fold) e giustificano l'inferenza dei team nascosti di p2.''',
r'''
key = lambda b: hashlib.md5(json.dumps(b["battle_timeline"], sort_keys=True).encode()).hexdigest()
groups = {}
for split, bs in (("train", train), ("test", test)):
    for b in bs:
        groups.setdefault(key(b), []).append((split, b.get("player_won")))
dup = [g for g in groups.values() if len(g) > 1]
kinds = Counter("-".join(sorted(s for s, _ in g)) for g in dup)
consistent = sum(len({w for s, w in g if s == "train"}) == 1 for g in dup if sum(s == "train" for s, _ in g) > 1)
print(f"gruppi duplicati: {len(dup)} {dict(kinds)}; gruppi con >1 battaglia di train ed etichetta coerente: {consistent}")

teams = Counter(frozenset(p["name"] for p in b["p1_team_details"]) for b in train + test)
p2_full = []
for b in train + test:
    seen = []
    for t in b["battle_timeline"]:
        n_ = t["p2_pokemon_state"]["name"]
        if n_ not in seen: seen.append(n_)
    if len(seen) == 6: p2_full.append(frozenset(seen))
in_pool = np.mean([t in teams for t in p2_full])
print(f"team p1: {sum(teams.values())} totali, {len(teams)} distinti; il team più frequente compare {teams.most_common(1)[0][1]} volte")
print(f"team p2 visti per intero: {len(p2_full)}, di cui {in_pool:.1%} compare anche come team p1")
freq = np.sort(np.array(list(teams.values())))[::-1]
plt.figure(figsize=(6, 3.4)); plt.loglog(np.arange(1, len(freq) + 1), freq)
plt.title("Rango-frequenza dei team p1"); plt.xlabel("rango"); plt.ylabel("occorrenze"); plt.show()
''',
r'''md:**Lettura:** i team non sono indipendenti: vengono da un pool ricorrente con una coda lunga (rango-frequenza quasi lineare in scala log-log). Per questo ricostruire i pokemon nascosti di p2 per team interi funziona meglio delle co-occorrenze a coppie. I pochi duplicati hanno sempre etichetta coerente; quelli tra train e test permettono di copiare l'etichetta in 7 battaglie di test.''',
r'''md:## 4. Relazione tra stato al turno 30 e vittoria

Probabilità di vittoria di p1 per fasce delle due variabili più informative, con intervalli di Wilson. Test χ² di indipendenza tra fascia e vittoria, e V di Cramér come misura dell'effetto.''',
r'''
def winrate_table(x, bins, name):
    cut = pd.cut(x, bins)
    t = pd.DataFrame({"bin": cut, "y": y}).groupby("bin", observed=True).y.agg(["sum", "count"])
    t["win"] = t["sum"] / t["count"]
    t[["lo", "hi"]] = [wilson(s, c) for s, c in zip(t["sum"], t["count"])]
    ct = pd.crosstab(cut, y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print(f"{name}: chi2 = {chi2:.0f}, dof = {dof}, p = {p:.1e}, V di Cramér = {v:.3f}")
    return t

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for a, (col, bins) in zip(ax, (("diff_hp_sum", np.quantile(tr.diff_hp_sum, np.linspace(0, 1, 11))),
                                ("diff_fainted", np.arange(-6.5, 6.5)))):
    t = winrate_table(tr[col], np.unique(bins), col)
    mid = np.arange(len(t))
    a.errorbar(mid, t.win, yerr=[t.win - t.lo, t.hi - t.win], fmt="o-", capsize=3)
    a.axhline(.5, color="gray", ls="--"); a.set_xticks(mid); a.set_xticklabels([str(i) for i in t.index], rotation=45, fontsize=7)
    a.set(title=f"P(p1 vince) per fasce di {col}", ylim=(0, 1))
plt.tight_layout(); plt.show()
''',
r'''md:**Lettura:** la relazione è monotona e molto significativa, ma l'effetto è moderato: anche nelle fasce più estreme la probabilità resta lontana da 0 e 1 per buona parte delle battaglie, e nelle fasce centrali (le più popolate) è vicina a 0.5. Molte partite sono genuinamente in bilico al turno 30.''',
r'''md:## 5. Significatività univariata delle feature

Per ognuna delle feature base: AUC rispetto a `player_won` (0.5 = nessun segnale) e test di Mann-Whitney (equivalente a testare AUC = 0.5). Con centinaia di test si corregge con Benjamini-Hochberg (FDR 5%).''',
r'''
num = tr.select_dtypes("number").columns
cols = [c for c in num if c not in ("battle_id", "is_test", "player_won") and tr[c].nunique() > 1]
rows = []
for c in cols:
    x = tr[c].fillna(tr[c].median()).values
    auc = roc_auc_score(y, x)
    p = stats.mannwhitneyu(x[y == 1], x[y == 0]).pvalue
    rows.append((c, auc, abs(auc - .5), p))
U = pd.DataFrame(rows, columns=["feature", "auc", "effetto", "p"])
U["q_bh"] = stats.false_discovery_control(U.p.fillna(1))
U = U.sort_values("effetto", ascending=False).reset_index(drop=True)
sig = (U.q_bh < .05).sum()
print(f"{len(U)} feature testate; significative dopo BH (q < 0.05): {sig} ({sig/len(U):.0%})")
print(f"con effetto |AUC-0.5| >= 0.1: {(U.effetto >= .1).sum()}; >= 0.05: {(U.effetto >= .05).sum()}")
U.head(20).style.format({"auc": "{:.3f}", "effetto": "{:.3f}", "p": "{:.1e}", "q_bh": "{:.1e}"})
''',
r'''
top = U.feature.head(15).tolist()
C = tr[top].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=90, fontsize=7)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7)
plt.colorbar(im); ax.set_title("Correlazione di Spearman tra le 15 feature più predittive"); plt.tight_layout(); plt.show()
off = C.values[np.triu_indices(len(top), 1)]
print(f"|rho| medio tra le top 15: {np.abs(off).mean():.2f}; coppie con |rho| > 0.8: {(np.abs(off) > .8).sum()}")
''',
r'''md:**Lettura:** gran parte delle feature è statisticamente significativa (con 10000 battaglie anche effetti piccoli lo sono), ma solo poche hanno un effetto grande, e le più forti sono fortemente correlate tra loro: misurano quasi tutte "chi è avanti negli HP/KO". Significatività univariata ≠ informazione nuova: è il motivo per cui le feature aggiunte in seguito (matchup, P(KO), storia per pokemon) sono significative da sole ma non migliorano lo stack.''',
r'''md:## 6. Train e test vengono dalla stessa distribuzione?

Due verifiche: test di Kolmogorov-Smirnov per feature (con BH) e *adversarial validation*, cioè un classificatore che prova a distinguere le battaglie di train da quelle di test. AUC ≈ 0.5 significa nessuno spostamento sfruttabile.''',
r'''
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import lightgbm as lgb
ks = pd.DataFrame([(c, *stats.ks_2samp(tr[c].dropna(), te[c].dropna())) for c in cols], columns=["feature", "ks", "p"])
ks["q_bh"] = stats.false_discovery_control(ks.p)
print(f"feature con distribuzione diversa tra train e test (BH q < 0.05): {(ks.q_bh < .05).sum()} su {len(ks)}")
X = pd.concat([tr[cols], te[cols]]).fillna(0).values; z = np.r_[np.zeros(len(tr)), np.ones(len(te))]
pa = cross_val_predict(lgb.LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=31, verbose=-1),
                       X, z, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"adversarial validation: AUC train vs test = {roc_auc_score(z, pa):.3f}")
ks.sort_values("ks", ascending=False).head(5)
''',
r'''md:**Lettura:** nessuna differenza rilevante tra train e test. La CV sul train è quindi una stima credibile dell'accuratezza sul test (al netto della varianza).''',
r'''md:## 7. Quanto si può prevedere? Curva di apprendimento e casi già decisi

Se l'accuracy salisse ancora con più dati, servirebbero modelli più espressivi; se è piatta, il limite è l'informazione contenuta nello stato al turno 30. Curva di apprendimento di un LightGBM sulle feature dense (5-fold, stesse condizioni per ogni dimensione).''',
r'''
from sklearn.model_selection import learning_curve
Xd = tr[cols].fillna(0).values
sizes, _, va = learning_curve(lgb.LGBMClassifier(n_estimators=400, learning_rate=.03, num_leaves=15,
                                                 colsample_bytree=.5, verbose=-1, n_jobs=-1),
                              Xd, y, train_sizes=[.05, .1, .2, .4, .6, .8, 1.0],
                              cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy")
m, s = va.mean(1), va.std(1)
plt.figure(figsize=(6, 3.6)); plt.errorbar(sizes, m, yerr=s, fmt="o-", capsize=3)
plt.xlabel("battaglie di training"); plt.ylabel("accuracy CV"); plt.title("Curva di apprendimento (LightGBM, feature dense)"); plt.show()
for n_, a, sd in zip(sizes, m, s): print(f"  {n_:5d}  {a:.4f} ± {sd:.4f}")
slope = (m[-1] - m[-3]) / (sizes[-1] - sizes[-3]) * 1000
print(f"pendenza finale: {slope*100:+.2f} punti di accuracy ogni 1000 battaglie in più")

done = (tr.p1_alive == 0) | (tr.p2_fainted == 6)
print(f"battaglie già finite al turno 30: {done.sum()} ({done.mean():.1%}); vincitore ovvio nel "
      f"{((tr.p1_alive[done] > 0).astype(int) == y[done]).mean():.0%} dei casi")
''',
r'''md:**Lettura:** la curva cresce sempre più lentamente: da 400 a 3200 battaglie guadagna circa 3 punti, da 3200 a 8000 poco più di 1. Con l'intero train la pendenza residua è di circa +0.15 punti ogni 1000 battaglie, e continua a diminuire: anche raddoppiando i dati il guadagno atteso sarebbe al massimo di circa 1 punto. Quasi nessuna partita è già decisa al turno 30 (0.6%). Il tetto di accuracy dipende soprattutto dall'informazione contenuta nello stato al turno 30, non dalla capacità del modello.''',
r'''md:## 8. Significatività dell'accuracy dei modelli

Qui si quantifica l'incertezza dei risultati principali:

- **IC dell'accuracy** del miglior stack (OOF sulle 10000 battaglie) e confronto con il caso (0.5) e con le baseline banali;
- **differenza minima rilevabile** in un confronto appaiato tra due modelli, che dipende da quante battaglie i due classificano in modo diverso;
- **varianza tra i fold** della stima annidata.''',
r'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
OOF = "outputs/oof_mc_rec"
names = sorted(f[:-4] for f in os.listdir(OOF) if f.endswith(".npz"))
Z = np.column_stack([logit(np.clip(np.load(f"{OOF}/{n}.npz")["oof"], 1e-4, 1 - 1e-4)) for n in names])
# predizione OOF del meta (logreg sui logit) mediata su 5 ripetizioni del 5-fold, come in stacking.meta_eval
pm = np.zeros(len(y)); accs = []
for r, (a, b) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=43).split(Z, y)):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Z[a], y[a]); p = m.predict_proba(Z[b])[:, 1]
    pm[b] += p / 5; accs.append(((p > .5) == y[b]).mean())
stack_ok = (pm > .5) == y
k = stack_ok.sum(); lo, hi = wilson(k, len(y))
print(f"stack a {len(names)} modelli: accuracy {np.mean(accs):.4f} (media dei fold, sd {np.std(accs):.4f}); "
      f"OOF aggregata {k/len(y):.4f} IC95% [{lo:.4f}, {hi:.4f}]")

base = {"caso (0.5)": None,
        "segno di diff_fainted": np.where(tr.diff_fainted != 0, tr.diff_fainted < 0, tr.diff_hp_sum > 0),
        "segno di diff_hp_sum": tr.diff_hp_sum.values > 0}
rows = []
for nm, pred in base.items():
    if pred is None:
        acc = .5; p = stats.binomtest(k, len(y), .5).pvalue; d = (k/len(y) - .5, np.nan, np.nan)
    else:
        ok = pred.astype(int) == y; acc = ok.mean(); p = mcnemar(ok, stack_ok)[2]; d = paired_bootstrap(ok, stack_ok)
    rows.append((nm, acc, d[0], d[1], d[2], p))
pd.DataFrame(rows, columns=["riferimento", "acc", "guadagno stack", "IC lo", "IC hi", "p"]).style.format(
    {"acc": "{:.4f}", "guadagno stack": "{:+.4f}", "IC lo": "{:+.4f}", "IC hi": "{:+.4f}", "p": "{:.1e}"})
''',
r'''
E = "outputs/experiments"
def nested(name):
    return pd.read_csv(f"{E}/{name}/predictions.csv").set_index("battle_id").sort_index()
b4, sl = nested("nested_base4"), nested("nested_base4_strat_lon")
ok_a = (b4.pred == b4.y_true).values; ok_b = (sl.pred == sl.y_true).values
disc = (ok_a != ok_b).mean()
mde = 1.96 * np.sqrt(disc / len(ok_a))
print(f"due stack simili classificano diversamente il {disc:.1%} delle battaglie")
print(f"=> differenza minima rilevabile al 5% (McNemar, approssimata): ±{mde*100:.2f} punti di accuracy")
print(f"   per rilevare +0.2 punti con questa discordanza servirebbero circa {int((1.96 / .002)**2 * disc):,} battaglie")
f = json.load(open(f"{E}/nested_base4/metrics.json"))["stack"]["acc_per_fold"]
print(f"stima annidata per fold: {np.round(f, 4)}, sd {np.std(f, ddof=1):.4f}: "
      f"la differenza tra il fold migliore e il peggiore è {(max(f)-min(f))*100:.1f} punti")
''',
r'''md:**Lettura:**

- Lo stack è molto sopra il caso e sopra le baseline banali basate su un solo indicatore (p ≪ 0.001).
- L'IC al 95% dell'accuracy assoluta è largo circa ±0.9 punti: **0.70 è dentro l'intervallo di 0.698**, quindi "superare 0.70" non è distinguibile dal non superarlo con questi dati.
- Nei confronti appaiati tra stack simili la soglia di significatività è circa ±0.5 punti; tutti gli esperimenti recenti stanno tra 0 e +0.3, e per dimostrare un +0.2 servirebbero circa 60 mila battaglie, sei volte il train.
- Tra un fold e l'altro l'accuracy varia di oltre un punto: un singolo split può mostrare 0.70 per caso.

## Conclusioni

1. Dati puliti: classi bilanciate, nessun leakage evidente, train e test con la stessa distribuzione, pochi duplicati gestiti.
2. Il segnale è reale e molto significativo ma di effetto moderato, ed è concentrato in poche dimensioni fortemente correlate (vantaggio di HP/KO).
3. La curva di apprendimento sempre più piatta e lo squilibrio dei KO prima/dopo il turno 30 indicano un tetto informativo vicino a 0.70.
4. Le differenze tra le varianti provate sono sotto la soglia rilevabile: per decidere tra loro conta la stima appaiata, non la singola accuracy.''',
]

# --------------------------------------------------------------------------------------------
# Report 2: metodi, modelli e architetture — resa contro spesa
# --------------------------------------------------------------------------------------------
R2 = [
r'''md:# Report statistico 2: metodi, modelli e architetture — la resa vale la spesa?

Per ogni componente del progetto si confronta il **guadagno di accuracy** (con test di significatività appaiati sulle stesse battaglie) con il **costo** (tempo di calcolo sul Mac M5 a 10 core, complessità). Componenti analizzati:

1. i 22 base model dello stack;
2. il meta-modello rispetto alla media e al miglior modello singolo;
3. la selezione di un sottoinsieme di modelli con una procedura annidata (stack ridotto);
4. i blocchi di feature e le architetture (Monte Carlo, modello ricorsivo, augmentation, matchup strategici, storia per pokemon, esperti di regime);
5. le politiche di fold e ripetizioni.

Costo misurato dai log delle run (`outputs/folds_experiment_log.txt` e i tempi delle singole pipeline sul M5). Le stime di accuratezza sono OOF (5-fold, seed 42) o annidate (`src/evaluate_nested.py`).''',
SETUP,
r'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
F = pd.read_pickle("outputs/features.pkl")
y = F[F.is_test == 0].player_won.values.astype(int)

def load_oof(d, names=None):
    names = names or sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npz"))
    return {n: np.load(f"{d}/{n}.npz") for n in names}

def to_Z(P, names):
    return np.column_stack([logit(np.clip(P[n]["oof"], 1e-4, 1 - 1e-4)) for n in names])

def meta_cv(Z, y, repeats=5, seed=43):
    """Meta logreg sui logit: (acc media dei fold, sd, predizione OOF mediata sulle ripetizioni)."""
    accs, pm = [], np.zeros(len(y))
    for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed).split(Z, y):
        p = LogisticRegression(C=1.0, max_iter=2000).fit(Z[a], y[a]).predict_proba(Z[b])[:, 1]
        pm[b] += p / repeats; accs.append(((p > .5) == y[b]).mean())
    return np.mean(accs), np.std(accs), pm

# tempi per modello (secondi, 5 fold) dalla run che ha prodotto outputs/oof_mc_rec
def parse_times(path):
    secs, sec, out = {}, None, {}
    for line in open(path):
        if line.startswith("=="): sec = line.strip("= \n")
        elif "OOF acc" in line and "|" in line:
            out.setdefault(sec, {})[line.split()[0]] = int(line.rsplit("|", 1)[1].strip().rstrip("s"))
        elif line.startswith("real") and sec: secs[sec] = float(line.split()[1])
    return out, secs
TIMES, TOTALS = parse_times("outputs/folds_experiment_log.txt")
T = TIMES["stacking folds=5 repeats=1"]
P = load_oof("outputs/oof_mc_rec"); names = sorted(P)
Z = to_Z(P, names)
print(f"{len(names)} base model; tempo totale dei base model {sum(T.values())/60:.1f} min")
''',
r'''md:## 1. I base model: accuratezza, overfitting e costo

`gap` = accuracy sul train − accuracy OOF (misura di overfitting). Il costo è il tempo per calcolare le 5 OOF più le predizioni di test.''',
r'''
rows = []
for n in names:
    p = P[n]["oof"]; ok = (p > .5) == y
    rows.append((n, ok.mean(), roc_auc_score(y, p), float(P[n]["train_acc"]) - ok.mean(), T.get(n, np.nan)))
B = pd.DataFrame(rows, columns=["modello", "acc OOF", "AUC", "gap", "tempo s"]).sort_values("acc OOF", ascending=False)
B.style.format({"acc OOF": "{:.4f}", "AUC": "{:.4f}", "gap": "{:+.3f}", "tempo s": "{:.0f}"}).background_gradient(subset=["acc OOF"], cmap="Greens")
''',
r'''
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(B["tempo s"], B["acc OOF"], c=B.gap, cmap="coolwarm", s=50)
for _, r in B.iterrows(): ax.annotate(r.modello, (r["tempo s"], r["acc OOF"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
ax.set_xscale("log"); ax.set(xlabel="tempo (s, scala log)", ylabel="accuracy OOF", title="Base model: accuracy contro costo (colore = gap train-OOF)")
plt.tight_layout(); plt.show()
''',
r'''md:**Lettura:** quasi tutti i modelli stanno in una fascia stretta (0.68-0.69) a costi che variano di due ordini di grandezza. I modelli lineari e generativi (lda, logreg_dense, gmm) costano secondi e quasi non vanno in overfitting; CatBoost, ExtraTrees, RF e SVM costano minuti senza essere più accurati da soli. Il loro valore va però misurato dentro lo stack (sezione 4): un modello meno accurato ma diverso può servire più di uno accurato e ridondante.''',
r'''md:## 2. Il meta-modello serve? Stack contro media e contro il miglior modello singolo

Confronti appaiati sulle stesse 10000 battaglie: differenza di accuracy con IC bootstrap e test di McNemar.''',
r'''
acc_s, sd_s, pm = meta_cv(Z, y)
stack_ok = (pm > .5) == y
best = B.iloc[0].modello
refs = {f"miglior singolo ({best})": (P[best]["oof"] > .5) == y,
        "media dei logit (22)": (Z.mean(1) > 0) == y,
        "lgbm da solo": (P["lgbm"]["oof"] > .5) == y,
        "logreg da sola": (P["logreg"]["oof"] > .5) == y}
rows = []
for nm, ok in refs.items():
    d, lo, hi = paired_bootstrap(ok, stack_ok); b, c, p = mcnemar(ok, stack_ok)
    rows.append((nm, ok.mean(), d, lo, hi, c, b, p))
print(f"stack (meta logreg, 22 modelli): {acc_s:.4f} ± {sd_s:.4f} (5x5 CV)")
pd.DataFrame(rows, columns=["riferimento", "acc", "guadagno stack", "IC lo", "IC hi", "corrette solo dallo stack", "solo dal rif.", "McNemar p"]).style.format(
    {"acc": "{:.4f}", "guadagno stack": "{:+.4f}", "IC lo": "{:+.4f}", "IC hi": "{:+.4f}", "McNemar p": "{:.3f}"})
''',
r'''md:**Lettura:** lo stack batte in modo significativo ogni modello singolo, compreso il migliore (xgb_aug, +0.6 punti, p ≈ 0.04), ma **non** si distingue in modo significativo dalla semplice media dei logit (+0.3 punti, p ≈ 0.19). Il meta-modello costa secondi, quindi conviene comunque; la gran parte del guadagno viene però dall'ensemble in sé, non dai pesi appresi.''',
r'''md:## 3. Diversità dei modelli: correlazione degli errori

Lo stacking guadagna solo se i modelli sbagliano su battaglie diverse. Correlazione tra i residui (y − p) delle OOF, ordinata per clustering gerarchico.''',
r'''
from scipy.cluster.hierarchy import linkage, leaves_list
R = np.column_stack([y - P[n]["oof"] for n in names])
C = np.corrcoef(R.T)
order = leaves_list(linkage(R.T, "average", metric="correlation"))
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(C[np.ix_(order, order)], cmap="viridis", vmin=.4, vmax=1)
ax.set_xticks(range(len(names))); ax.set_xticklabels(np.array(names)[order], rotation=90, fontsize=7)
ax.set_yticks(range(len(names))); ax.set_yticklabels(np.array(names)[order], fontsize=7)
plt.colorbar(im); ax.set_title("Correlazione dei residui OOF"); plt.tight_layout(); plt.show()
off = C[np.triu_indices(len(names), 1)]
print(f"correlazione media dei residui: {off.mean():.2f} (min {off.min():.2f}, max {off.max():.2f})")
''',
r'''md:## 4. Valore marginale di ogni modello nello stack (ablazione)

Per ogni base model si rifà il meta senza di lui, sugli **stessi split** (5×5): un valore negativo significa che toglierlo peggiora lo stack. Il rapporto con il tempo dice quanto costa ogni decimo di punto.''',
r'''
splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=43).split(Z, y))
def acc_on(cols):
    return np.array([((LogisticRegression(C=1.0, max_iter=2000).fit(Z[a][:, cols], y[a]).predict_proba(Z[b][:, cols])[:, 1] > .5) == y[b]).mean()
                     for a, b in splits])
full = acc_on(list(range(len(names))))
rows = []
for i, n in enumerate(names):
    d = acc_on([j for j in range(len(names)) if j != i]) - full
    rows.append((n, -d.mean(), d.std(ddof=1) / np.sqrt(len(d)), T.get(n, np.nan)))
A = pd.DataFrame(rows, columns=["modello", "contributo", "se", "tempo s"]).sort_values("contributo", ascending=False)
A["punti per minuto"] = A.contributo * 100 / (A["tempo s"] / 60)
print(f"stack completo: {full.mean():.4f}")
A.style.format({"contributo": "{:+.4f}", "se": "{:.4f}", "tempo s": "{:.0f}", "punti per minuto": "{:+.3f}"}).bar(subset=["contributo"], align="zero", color=["#d65f5f", "#5fba7d"])
''',
r'''md:**Lettura:**

- I residui sono correlati in media 0.97: i modelli sbagliano quasi sempre sulle stesse battaglie, e ogni contributo individuale è piccolo (al massimo +0.3 punti).
- Contro l'intuizione della sezione 1, i modelli che contribuiscono di più sono proprio alcuni dei **più costosi e meno accurati da soli**: ExtraTrees, RF e i CatBoost con augmentation. Sono i più diversi dai boosting (bagging con alberi profondi), e la loro diversità vale più della loro accuratezza.
- Hanno contributo nullo o negativo i modelli ridondanti con altri: lgbm, xgb, svm, catboost e catboost_dense senza augmentation, i GMM, lgbm_noise.
- Avvertenza: l'errore standard viene da 25 split ripetuti degli stessi dati, quindi è sottostimato; differenze sotto ~0.1 punti vanno considerate nulle.''',
r'''md:## 5. Uno stack ridotto è altrettanto buono? Selezione annidata

Selezione in avanti (greedy) dei base model **dentro** ogni fold esterno: si sceglie sul training esterno con una CV interna e si valuta sul fold esterno mai visto. Così la stima non è ottimistica (la selezione greedy fatta sulla stessa CV darebbe numeri gonfiati). Si ferma quando aggiungere un modello migliora la CV interna di meno di 0.05 punti.''',
r'''
def greedy(Zt, yt, max_k=8, tol=5e-4, seed=7):
    sp = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Zt, yt))
    def score(cols):
        return np.mean([((LogisticRegression(C=1.0, max_iter=2000).fit(Zt[a][:, cols], yt[a]).predict_proba(Zt[b][:, cols])[:, 1] > .5) == yt[b]).mean() for a, b in sp])
    chosen, cur = [], 0.0
    while len(chosen) < max_k:
        cand = [(score(chosen + [j]), j) for j in range(Zt.shape[1]) if j not in chosen]
        s, j = max(cand)
        if s - cur < tol: break
        chosen.append(j); cur = s
    return chosen

outer = StratifiedKFold(5, shuffle=True, random_state=42)
p_small, p_full, picks = np.zeros(len(y)), np.zeros(len(y)), []
for a, b in outer.split(Z, y):
    cols = greedy(Z[a], y[a]); picks.append([names[j] for j in cols])
    p_small[b] = LogisticRegression(C=1.0, max_iter=2000).fit(Z[a][:, cols], y[a]).predict_proba(Z[b][:, cols])[:, 1]
    p_full[b] = LogisticRegression(C=1.0, max_iter=2000).fit(Z[a], y[a]).predict_proba(Z[b])[:, 1]
ok_s, ok_f = (p_small > .5) == y, (p_full > .5) == y
d, lo, hi = paired_bootstrap(ok_f, ok_s); _, _, pv = mcnemar(ok_f, ok_s)
cnt = Counter(m for p in picks for m in p)
t_small = np.mean([sum(T[m] for m in p) for p in picks])
print(f"stack completo (22): {ok_f.mean():.4f}, tempo base model {sum(T.values())/60:.1f} min")
print(f"stack ridotto (in media {np.mean([len(p) for p in picks]):.1f} modelli): {ok_s.mean():.4f}, tempo {t_small/60:.1f} min")
print(f"differenza ridotto - completo: {d*100:+.2f} punti, IC95% [{lo*100:+.2f}, {hi*100:+.2f}], McNemar p = {pv:.2f}")
print("modelli scelti (fold esterni su 5):", dict(cnt.most_common()))
''',
r'''md:La selezione greedy si ferma troppo presto. Confronto anche alcuni sottoinsiemi fissati, sugli stessi split dello stack completo. I primi due sono decisi solo in base al costo, quindi la stima è onesta. Gli ultimi due sono suggeriti dall'ablazione della sezione 4, fatta sugli stessi dati: la loro stima è leggermente ottimistica e va confermata con una valutazione annidata.''',
r'''
subsets = {
    "economici (<= 20 s ciascuno)": [n for n in names if T[n] <= 20],
    "4 classici (logreg, lgbm, xgb, catboost)": ["logreg", "lgbm", "xgb", "catboost"],
    "senza svm e senza i catboost non _aug": [n for n in names if n not in ("svm", "catboost", "catboost_dense")],
    "solo boosting _aug + ET + RF + lineari": [n for n in names if n.endswith("_aug") or n in ("extratrees", "rf", "lda", "logreg", "logreg_dense")],
}
rows = []
for nm, ms in subsets.items():
    cols = [names.index(m) for m in ms]
    _, _, p_ = meta_cv(Z[:, cols], y)
    ok = (p_ > .5) == y
    d, lo, hi = paired_bootstrap(stack_ok, ok); _, _, pv = mcnemar(stack_ok, ok)
    t = sum(T[m] for m in ms)
    rows.append((nm, len(ms), acc_on(cols).mean(), d * 100, lo * 100, hi * 100, pv, t / 60, 1 - t / sum(T.values())))
SUB = pd.DataFrame(rows, columns=["sottoinsieme", "modelli", "acc", "Δ vs 22", "IC lo", "IC hi", "McNemar p", "tempo min", "risparmio"])
SUB.style.format({"acc": "{:.4f}", "Δ vs 22": "{:+.2f}", "IC lo": "{:+.2f}", "IC hi": "{:+.2f}", "McNemar p": "{:.3f}", "tempo min": "{:.1f}", "risparmio": "{:.0%}"})
''',
r'''md:**Lettura:**

- Ridurre lo stack ai modelli economici o ai 4 classici costa in modo **significativo** (−0.6 / −0.9 punti), e lo stesso vale per la selezione greedy annidata (−0.6 punti, p ≈ 0.02). Il risparmio di tempo (86%) si paga in accuratezza.
- Togliere i modelli ridondanti (svm, catboost senza augmentation, lgbm/xgb/gmm) non peggiora: con 10 modelli e il 43% di tempo in meno l'accuratezza è uguale (+0.15, non significativo).
- Quindi la diversità serve, la quantità no: conviene tenere i modelli diversi (boosting con augmentation, ExtraTrees, RF, lineari) e togliere i duplicati.''',
r'''md:## 6. Blocchi di feature e architetture

Ogni riga confronta una variante con la propria baseline **sugli stessi fold e sulle stesse battaglie**:

- **Monte Carlo e ricorsivo** (feature v3, 17 base model; e singoli modelli lgbm/xgb/lda/gmm/logreg_dense): OOF standard, meta rifatto qui con gli stessi split.
- **Augmentation** (modelli `_aug`, v4): stack v4 con e senza i 5 modelli `_aug`.
- **Matchup strategici, storia per pokemon, esperti di regime, pesi dell'augmentation:** stime annidate di `evaluate_nested.py` (4 o 3 base model).
- **Strat + lon nello stack completo a 22 modelli.**

Il costo è il tempo aggiuntivo della pipeline (generazione delle feature, più il tempo di training dei modelli in più).''',
r'''
def compare_stacks(Pa, Pb, names_a, names_b):
    _, _, pa = meta_cv(to_Z(Pa, names_a), y); _, _, pb = meta_cv(to_Z(Pb, names_b), y)
    return (pa > .5) == y, (pb > .5) == y

def nested_ok(name):
    d = pd.read_csv(f"outputs/experiments/{name}/predictions.csv").set_index("battle_id").sort_index()
    return (d.pred == d.y_true).values

rows = []
def add(label, ok_a, ok_b, cost_s, kind):
    d, lo, hi = paired_bootstrap(ok_a, ok_b); _, _, p = mcnemar(ok_a, ok_b)
    rows.append((label, kind, ok_a.mean(), ok_b.mean(), d * 100, lo * 100, hi * 100, p, cost_s))

# Monte Carlo + ricorsivo sullo stack v3 a 17 modelli (stessi fold)
v3, v3mr = load_oof("outputs/oof_v3"), load_oof("outputs/oof_v3_mc_rec")
add("Monte Carlo + ricorsivo (stack 17)", *compare_stacks(v3, v3mr, sorted(v3), sorted(v3)), 190 + 122, "OOF")
for extra, cost in (("mc", 190), ("rec", 122)):
    Pe = load_oof(f"outputs/oof_v3_{extra}"); ms = sorted(Pe)
    add(f"{'Monte Carlo' if extra == 'mc' else 'ricorsivo'} da solo (stack {len(ms)})", *compare_stacks(v3, Pe, ms, ms), cost, "OOF")
# augmentation: stack v4 senza/con i modelli _aug (stesse feature)
v4 = load_oof("outputs/oof_v4a", [n for n in names if n in os.listdir("outputs/oof_v4a") or f"{n}.npz" in os.listdir("outputs/oof_v4a")]); no_aug = sorted(n for n in v4 if "_aug" not in n)
aug_names = sorted(n for n in v4 if "_aug" in n)
add(f"augmentation ({len(aug_names)} modelli _aug)", *compare_stacks(v4, v4, no_aug, sorted(v4)), sum(T[n] for n in aug_names), "OOF")
# strat + lon nello stack completo
Psl = load_oof("outputs/oof_mc_rec_strat_lon")
add("strat + lon (stack 22)", *compare_stacks(P, Psl, names, names), 121 + 14, "OOF")
# esperimenti annidati
base4 = nested_ok("nested_base4")
for name, label, cost in (("nested_base4_strat", "matchup strategici + P(KO) + pool", 121),
                          ("nested_base4_strat_nopool", "matchup strategici senza pool", 121),
                          ("nested_base4_lon", "storia per pokemon", 14),
                          ("nested_base4_strat_lon", "strategici + storia", 135),
                          ("nested_base4_exp", "esperti di regime", 3 * 15),
                          ("nested_base4_exp_gate", "esperti + gating", 3 * 15)):
    add(label, base4, nested_ok(name), cost, "annidata")
w5 = nested_ok("nested_aug_w0.5")
for w in ("0.25", "1.0"):
    add(f"peso augmentation {w} (contro 0.5)", w5, nested_ok(f"nested_aug_w{w}"), 0, "annidata")
D = pd.DataFrame(rows, columns=["componente", "stima", "acc base", "acc con", "Δ punti", "IC lo", "IC hi", "McNemar p", "costo s"])
D.style.format({"acc base": "{:.4f}", "acc con": "{:.4f}", "Δ punti": "{:+.2f}", "IC lo": "{:+.2f}", "IC hi": "{:+.2f}", "McNemar p": "{:.3f}", "costo s": "{:.0f}"}).bar(subset=["Δ punti"], align="zero", color=["#d65f5f", "#5fba7d"])
''',
r'''
fig, ax = plt.subplots(figsize=(9, 4.8))
yy = np.arange(len(D))[::-1]
ax.errorbar(D["Δ punti"], yy, xerr=[D["Δ punti"] - D["IC lo"], D["IC hi"] - D["Δ punti"]], fmt="o", capsize=3)
ax.axvline(0, color="gray", ls="--"); ax.set_yticks(yy); ax.set_yticklabels(D.componente, fontsize=8)
ax.set(xlabel="Δ accuracy (punti) con IC 95% bootstrap appaiato", title="Effetto di ogni componente sulla propria baseline")
plt.tight_layout(); plt.show()
''',
r'''md:**Lettura:**

- **Nessun componente** ha un IC appaiato tutto sopra lo zero.
- L'**augmentation** è il più vicino (+0.30 punti, p ≈ 0.12) ed è anche il più semplice: riusa le feature esistenti sui turni 24/26/28.
- **Monte Carlo e ricorsivo:** nello stack v3 a 17 modelli il guadagno è nullo (−0.03). Migliorano di circa +0.3 punti alcuni modelli singoli, ma l'informazione è già nell'ensemble. Costano circa 5 minuti di calcolo e la parte più complessa del codice.
- **Matchup strategici, pool dei team, storia per pokemon:** +0.1/+0.2 punti, non dimostrabili. La storia per pokemon è l'unica economica (14 s).
- **Esperti di regime** e **peso 1.0 dell'augmentation** peggiorano. Il peso 0.25 sembra meglio nello stack a 3 modelli (p ≈ 0.10) ma non nello stack completo.''',
r'''md:## 7. Fold e ripetizioni

Tre configurazioni dei base model (22 modelli, feature + mc + rec), stesso meta valutato con gli stessi split. Le OOF hanno split diversi tra le configurazioni, quindi il confronto appaiato riguarda la stessa battaglia ma con modelli addestrati su partizioni diverse.''',
r'''
rows = []
ok0 = None
for d, sec in (("outputs/oof_mc_rec", "stacking folds=5 repeats=1"), ("outputs/oof_mc_rec_k10r1", "stacking folds=10 repeats=1"),
               ("outputs/oof_mc_rec_k5r3", "stacking folds=5 repeats=3")):
    Pd = load_oof(d); a, sd, pm_ = meta_cv(to_Z(Pd, sorted(Pd)), y); ok = (pm_ > .5) == y
    if ok0 is None: ok0 = ok
    dd, lo, hi = paired_bootstrap(ok0, ok)
    rows.append((sec.replace("stacking ", ""), a, sd, dd * 100, lo * 100, hi * 100, TOTALS.get(sec, np.nan) / 60))
pd.DataFrame(rows, columns=["configurazione", "meta acc", "sd", "Δ vs 5x1", "IC lo", "IC hi", "tempo min"]).style.format(
    {"meta acc": "{:.4f}", "sd": "{:.4f}", "Δ vs 5x1": "{:+.2f}", "IC lo": "{:+.2f}", "IC hi": "{:+.2f}", "tempo min": "{:.1f}"})
''',
r'''md:**Lettura:** né 10 fold né 3 ripetizioni migliorano lo stack: entrambi sono peggiori (non significativamente) e costano 2-3 volte il tempo. Il bagging delle ripetizioni migliora i singoli modelli ma rende le OOF più simili tra loro, togliendo diversità al meta.''',
r'''md:## 8. Quadro riassuntivo: resa contro spesa

Verdetto automatico per componente:

- **conviene** se l'IC appaiato è tutto sopra lo zero;
- **neutro, economico** se l'IC contiene lo zero ma il costo è basso (< 1 min) e la stima puntuale è ≥ 0;
- **guadagno non dimostrato** se l'IC contiene lo zero, la stima è positiva ma il costo è alto;
- **non conviene** se la stima è negativa.''',
r'''
def verdict(r):
    if r["IC lo"] > 0: return "conviene"
    if r["Δ punti"] < 0: return "non conviene"
    return "neutro, economico" if r["costo s"] < 60 else "guadagno non dimostrato"
S = D.copy(); S["verdetto"] = S.apply(verdict, axis=1)
S["punti per minuto"] = np.where(S["costo s"] > 0, S["Δ punti"] / (S["costo s"] / 60), np.nan)
print(f"stack completo 22 modelli: {acc_s:.4f} (base model {sum(T.values())/60:.1f} min)  |  "
      f"stack ridotto annidato: {ok_s.mean():.4f} ({t_small/60:.1f} min)")
S[["componente", "Δ punti", "IC lo", "IC hi", "McNemar p", "costo s", "punti per minuto", "verdetto"]].style.format(
    {"Δ punti": "{:+.2f}", "IC lo": "{:+.2f}", "IC hi": "{:+.2f}", "McNemar p": "{:.3f}", "costo s": "{:.0f}", "punti per minuto": "{:+.2f}"})
''',
r'''md:## Conclusioni

- **Ensemble:** è l'unica scelta con una resa dimostrata. Lo stack batte in modo significativo il miglior modello singolo (+0.6 punti) e, soprattutto, i sottoinsiemi economici (+0.6/+0.9 punti). Il meta-modello logreg in sé aggiunge poco rispetto alla media semplice, ma costa secondi.
- **Quali base model:** contano la diversità, non il numero né l'accuratezza individuale. ExtraTrees, RF e i CatBoost con augmentation, pur costosi e meno accurati da soli, sono quelli che contribuiscono di più. SVM, i GMM, lgbm/xgb/catboost senza augmentation e le varianti con rumore sono ridondanti: si possono togliere risparmiando circa il 40% del tempo, senza perdite misurabili (da confermare con una valutazione annidata).
- **Augmentation con snapshot:** il componente con il miglior rapporto resa/spesa (+0.3 punti, al limite della significatività, costo circa 3.5 minuti).
- **Monte Carlo e modello ricorsivo:** nessun guadagno misurabile nello stack, a fronte del costo maggiore (5 minuti e la parte più complessa del codice). Hanno valore come strumento di analisi, non come leva di accuratezza.
- **Matchup strategici, pool dei team, storia per pokemon:** segnale reale ma ridondante (+0.1/+0.2, non dimostrabile). Solo la storia per pokemon è abbastanza economica da poterla tenere senza rimpianti.
- **Esperti di regime, 10 fold, ripetizioni del k-fold:** spesa senza resa, da non usare.
- **In sintesi:** il tetto non dipende dai modelli. Con questi dati nessuna aggiunta oltre l'ensemble di modelli diversi e l'augmentation ha una resa che giustifica la spesa.''',
]


def main():
    for fname, cells in (("01_report_statistico_dati.ipynb", R1), ("02_report_costi_benefici.ipynb", R2)):
        nbf.write(notebook(cells), os.path.join(HERE, fname))
        print("scritto", fname)


if __name__ == "__main__":
    main()
