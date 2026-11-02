"""Generate the two analysis notebooks (data/significance and method costs/benefits).

Usage: python reports/build_notebooks.py, then
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
    """95% Wilson interval for a proportion."""
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h

def mcnemar(a_ok, b_ok):
    """Exact McNemar test on paired correctness vectors."""
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p

def paired_bootstrap(a_ok, b_ok, n=5000):
    """95% CI for accuracy difference (b - a), resampling battles."""
    d = b_ok.astype(float) - a_ok.astype(float)
    idx = RNG.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
'''

# --------------------------------------------------------------------------------------------
# Report 1: data and significance
# --------------------------------------------------------------------------------------------
R1 = [
r'''md:# Statistical report 1: data and significance of the results

This notebook describes the competition data (Gen 2 OU Pokemon battles, predicting `player_won` from the turn-30 state) and quantifies the significance of class balance, state/outcome relationships, individual feature predictive power, train/test shift and model-accuracy uncertainty.

It uses existing caches (`outputs/features.pkl`, `outputs/oof_mc_rec/`, `outputs/experiments/`) without retraining the stack. Tests are two-sided with alpha = 0.05; multiple comparisons use Benjamini-Hochberg false-discovery-rate control.

Historical analysis: prose and textual outputs have been translated without rerunning calculations. Saved plot images remain unchanged; their source labels are translated for future regeneration.''',
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
print(f"train {len(train)} battles, test {len(test)}, base features {F.shape[1] - 3}")
''',
r'''md:## 1. Class balance

With imbalanced classes, accuracy should be compared with the majority-class baseline rather than 0.5. Use an exact binomial test against p = 0.5.''',
r'''
k = y.sum(); n = len(y)
lo, hi = wilson(k, n)
print(f"p1 wins {k}/{n} = {k/n:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  binomial p = {stats.binomtest(k, n, .5).pvalue:.3f}")
''',
r'''md:**Interpretation:** classes are balanced, with no significant difference from 50%. Random-reference accuracy is therefore 0.50. There is no reason to move the probability threshold from 0.5; the median-threshold experiment also found no effect.''',
r'''md:## 2. Battle structure

Examine timeline lengths, observed p2 Pokemon and KOs per side by turn 30. KO counts are paired within each battle, so compare p1 and p2 using the Wilcoxon signed-rank test.''',
r'''
lens = Counter(len(b["battle_timeline"]) for b in train + test)
print("timeline length:", dict(lens))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for d, lab in ((tr, "train"), (te, "test")):
    ax[0].hist(d.p2_n_seen, bins=np.arange(.5, 7.5), alpha=.6, density=True, label=lab)
ax[0].set(title="p2 Pokemon observed by turn 30", xlabel="observed"); ax[0].legend()
ax[1].hist([tr.p1_fainted, tr.p2_fainted], bins=np.arange(-.5, 7.5), label=["p1", "p2"])
ax[1].set(title="KOs suffered by turn 30 (train)", xlabel="fainted Pokemon"); ax[1].legend()
ax[2].hist(tr.diff_hp_sum, bins=60, color="gray"); ax[2].set(title="diff_hp_sum (p1 - p2)", xlabel="total HP")
plt.tight_layout(); plt.show()
w = stats.wilcoxon(tr.p1_fainted, tr.p2_fainted)
print(f"mean KOs at turn 30: p1 {tr.p1_fainted.mean():.2f}, p2 {tr.p2_fainted.mean():.2f}; Wilcoxon p = {w.pvalue:.1e}")
print(f"yet p1 wins {y.mean():.1%}: turn-30 KO difference alone does not decide the battle")
''',
r'''md:**Interpretation:** timelines always contain 30 turns, so length does not reveal the outcome. p1 suffers significantly more KOs in the first 30 turns but wins half the battles. Dynamics after turn 30 differ from those observed, which is a structural limitation of this problem.''',
r'''md:## 3. Duplicates and the team pool

Inspect identical timelines and recurring teams. These affect validation (duplicates should stay in the same fold) and motivate inference of p2's hidden team.''',
r'''
key = lambda b: hashlib.md5(json.dumps(b["battle_timeline"], sort_keys=True).encode()).hexdigest()
groups = {}
for split, bs in (("train", train), ("test", test)):
    for b in bs:
        groups.setdefault(key(b), []).append((split, b.get("player_won")))
dup = [g for g in groups.values() if len(g) > 1]
kinds = Counter("-".join(sorted(s for s, _ in g)) for g in dup)
consistent = sum(len({w for s, w in g if s == "train"}) == 1 for g in dup if sum(s == "train" for s, _ in g) > 1)
print(f"duplicate groups: {len(dup)} {dict(kinds)}; groups with >1 training battle and consistent labels: {consistent}")

teams = Counter(frozenset(p["name"] for p in b["p1_team_details"]) for b in train + test)
p2_full = []
for b in train + test:
    seen = []
    for t in b["battle_timeline"]:
        n_ = t["p2_pokemon_state"]["name"]
        if n_ not in seen: seen.append(n_)
    if len(seen) == 6: p2_full.append(frozenset(seen))
in_pool = np.mean([t in teams for t in p2_full])
print(f"p1 teams: {sum(teams.values())} total, {len(teams)} distinct; most frequent team appears {teams.most_common(1)[0][1]} times")
print(f"fully observed p2 teams: {len(p2_full)}, of which {in_pool:.1%} also occur as p1 teams")
freq = np.sort(np.array(list(teams.values())))[::-1]
plt.figure(figsize=(6, 3.4)); plt.loglog(np.arange(1, len(freq) + 1), freq)
plt.title("p1 team rank-frequency"); plt.xlabel("rank"); plt.ylabel("occurrences"); plt.show()
''',
r'''md:**Interpretation:** teams come from a recurring pool with a long tail, approximately linear in a log-log rank/frequency plot. Whole-team reconstruction therefore works better than pairwise co-occurrence. The small number of duplicates have consistent labels; seven exact train/test matches allow the original pipeline to copy training labels.''',
r'''md:## 4. Turn-30 state and victory

Estimate p1 win probabilities across bins of the two most informative variables, with Wilson intervals. Use a chi-squared independence test and Cramer's V as an effect-size measure.''',
r'''
def winrate_table(x, bins, name):
    cut = pd.cut(x, bins)
    t = pd.DataFrame({"bin": cut, "y": y}).groupby("bin", observed=True).y.agg(["sum", "count"])
    t["win"] = t["sum"] / t["count"]
    t[["lo", "hi"]] = [wilson(s, c) for s, c in zip(t["sum"], t["count"])]
    ct = pd.crosstab(cut, y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print(f"{name}: chi2 = {chi2:.0f}, dof = {dof}, p = {p:.1e}, Cramer's V = {v:.3f}")
    return t

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for a, (col, bins) in zip(ax, (("diff_hp_sum", np.quantile(tr.diff_hp_sum, np.linspace(0, 1, 11))),
                                ("diff_fainted", np.arange(-6.5, 6.5)))):
    t = winrate_table(tr[col], np.unique(bins), col)
    mid = np.arange(len(t))
    a.errorbar(mid, t.win, yerr=[t.win - t.lo, t.hi - t.win], fmt="o-", capsize=3)
    a.axhline(.5, color="gray", ls="--"); a.set_xticks(mid); a.set_xticklabels([str(i) for i in t.index], rotation=45, fontsize=7)
    a.set(title=f"P(p1 wins) by bins of {col}", ylim=(0, 1))
plt.tight_layout(); plt.show()
''',
r'''md:**Interpretation:** the relationship is monotonic and highly significant, but its effect is moderate. Even extreme bins often have win probabilities far from 0 and 1; central, more populated bins are close to 0.5. Many battles remain undecided at turn 30.''',
r'''md:## 5. Univariate feature significance

For each base feature, compute AUC against `player_won` (0.5 means no signal) and a Mann-Whitney test, equivalent to testing AUC = 0.5. Correct the hundreds of comparisons using Benjamini-Hochberg at 5% FDR.''',
r'''
num = tr.select_dtypes("number").columns
cols = [c for c in num if c not in ("battle_id", "is_test", "player_won") and tr[c].nunique() > 1]
rows = []
for c in cols:
    x = tr[c].fillna(tr[c].median()).values
    auc = roc_auc_score(y, x)
    p = stats.mannwhitneyu(x[y == 1], x[y == 0]).pvalue
    rows.append((c, auc, abs(auc - .5), p))
U = pd.DataFrame(rows, columns=["feature", "auc", "effect", "p"])
U["q_bh"] = stats.false_discovery_control(U.p.fillna(1))
U = U.sort_values("effect", ascending=False).reset_index(drop=True)
sig = (U.q_bh < .05).sum()
print(f"{len(U)} features tested; significant after BH (q < 0.05): {sig} ({sig/len(U):.0%})")
print(f"with effect |AUC-0.5| >= 0.1: {(U.effect >= .1).sum()}; >= 0.05: {(U.effect >= .05).sum()}")
U.head(20).style.format({"auc": "{:.3f}", "effect": "{:.3f}", "p": "{:.1e}", "q_bh": "{:.1e}"})
''',
r'''
top = U.feature.head(15).tolist()
C = tr[top].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=90, fontsize=7)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7)
plt.colorbar(im); ax.set_title("Spearman correlation among the 15 most predictive features"); plt.tight_layout(); plt.show()
off = C.values[np.triu_indices(len(top), 1)]
print(f"mean |rho| among top 15: {np.abs(off).mean():.2f}; pairs with |rho| > 0.8: {(np.abs(off) > .8).sum()}")
''',
r'''md:**Interpretation:** many features are statistically significant because 10,000 battles can reveal small effects. Few effects are large, and the strongest features are highly correlated: most describe HP/KO advantage. Univariate significance is not new information. Later matchup, KO-probability and per-Pokemon-history features can be significant individually without improving the stack.''',
r'''md:## 6. Do train and test follow the same distribution?

Use feature-wise Kolmogorov-Smirnov tests with BH correction and adversarial validation: a classifier tries to distinguish training from test battles. AUC near 0.5 indicates no exploitable shift.''',
r'''
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import lightgbm as lgb
ks = pd.DataFrame([(c, *stats.ks_2samp(tr[c].dropna(), te[c].dropna())) for c in cols], columns=["feature", "ks", "p"])
ks["q_bh"] = stats.false_discovery_control(ks.p)
print(f"features with different train/test distributions (BH q < 0.05): {(ks.q_bh < .05).sum()} out of {len(ks)}")
X = pd.concat([tr[cols], te[cols]]).fillna(0).values; z = np.r_[np.zeros(len(tr)), np.ones(len(te))]
pa = cross_val_predict(lgb.LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=31, verbose=-1),
                       X, z, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"adversarial validation: AUC train vs test = {roc_auc_score(z, pa):.3f}")
ks.sort_values("ks", ascending=False).head(5)
''',
r'''md:**Interpretation:** these historical checks found no relevant train/test difference. Subject to sampling variability, training CV can therefore estimate test accuracy.''',
r'''md:## 7. Predictability: learning curve and already decided battles

A rising learning curve would suggest benefit from more data or expressive models; a flat curve suggests limited information in the turn-30 state. Fit a dense-feature LightGBM learning curve with five-fold CV under the same conditions at each training size.''',
r'''
from sklearn.model_selection import learning_curve
Xd = tr[cols].fillna(0).values
sizes, _, va = learning_curve(lgb.LGBMClassifier(n_estimators=400, learning_rate=.03, num_leaves=15,
                                                 colsample_bytree=.5, verbose=-1, n_jobs=-1),
                              Xd, y, train_sizes=[.05, .1, .2, .4, .6, .8, 1.0],
                              cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy")
m, s = va.mean(1), va.std(1)
plt.figure(figsize=(6, 3.6)); plt.errorbar(sizes, m, yerr=s, fmt="o-", capsize=3)
plt.xlabel("training battles"); plt.ylabel("accuracy CV"); plt.title("Learning curve (LightGBM, dense features)"); plt.show()
for n_, a, sd in zip(sizes, m, s): print(f"  {n_:5d}  {a:.4f} ± {sd:.4f}")
slope = (m[-1] - m[-3]) / (sizes[-1] - sizes[-3]) * 1000
print(f"final slope: {slope*100:+.2f} accuracy points per 1,000 additional battles")

done = (tr.p1_alive == 0) | (tr.p2_fainted == 6)
print(f"battles already over at turn 30: {done.sum()} ({done.mean():.1%}); obvious winner in "
      f"{((tr.p1_alive[done] > 0).astype(int) == y[done]).mean():.0%} of cases")
''',
r'''md:**Interpretation:** growth slows: accuracy gains about 3 points from 400 to 3,200 battles and slightly over 1 point from 3,200 to 8,000. At full training size the remaining slope is about +0.15 points per 1,000 battles and continues to decrease; doubling the data would add at most about 1 point. Only 0.6% of battles are already decided by turn 30. This historical analysis attributes the accuracy ceiling mainly to observed information rather than model capacity.''',
r'''md:## 8. Statistical significance of model accuracy

Quantify uncertainty in the main results:

- Accuracy confidence intervals for the best stack on 10,000 OOF battles, compared with chance (0.5) and simple baselines.
- Minimum detectable paired differences, depending on how often two models disagree.
- Between-fold variability in nested estimates.''',
r'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
OOF = "outputs/oof_mc_rec"
names = sorted(f[:-4] for f in os.listdir(OOF) if f.endswith(".npz"))
Z = np.column_stack([logit(np.clip(np.load(f"{OOF}/{n}.npz")["oof"], 1e-4, 1 - 1e-4)) for n in names])
# Average logistic-meta OOF predictions over five repeats of five-fold CV, as in stacking.meta_eval
pm = np.zeros(len(y)); accs = []
for r, (a, b) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=43).split(Z, y)):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Z[a], y[a]); p = m.predict_proba(Z[b])[:, 1]
    pm[b] += p / 5; accs.append(((p > .5) == y[b]).mean())
stack_ok = (pm > .5) == y
k = stack_ok.sum(); lo, hi = wilson(k, len(y))
print(f"stack with {len(names)} models: accuracy {np.mean(accs):.4f} (mean across folds, SD {np.std(accs):.4f}); "
      f"aggregated OOF {k/len(y):.4f} 95% CI [{lo:.4f}, {hi:.4f}]")

base = {"chance (0.5)": None,
        "sign of diff_fainted": np.where(tr.diff_fainted != 0, tr.diff_fainted < 0, tr.diff_hp_sum > 0),
        "sign of diff_hp_sum": tr.diff_hp_sum.values > 0}
rows = []
for nm, pred in base.items():
    if pred is None:
        acc = .5; p = stats.binomtest(k, len(y), .5).pvalue; d = (k/len(y) - .5, np.nan, np.nan)
    else:
        ok = pred.astype(int) == y; acc = ok.mean(); p = mcnemar(ok, stack_ok)[2]; d = paired_bootstrap(ok, stack_ok)
    rows.append((nm, acc, d[0], d[1], d[2], p))
pd.DataFrame(rows, columns=["reference", "acc", "stack gain", "CI low", "CI high", "p"]).style.format(
    {"acc": "{:.4f}", "stack gain": "{:+.4f}", "CI low": "{:+.4f}", "CI high": "{:+.4f}", "p": "{:.1e}"})
''',
r'''
E = "outputs/experiments"
def nested(name):
    return pd.read_csv(f"{E}/{name}/predictions.csv").set_index("battle_id").sort_index()
b4, sl = nested("nested_base4"), nested("nested_base4_strat_lon")
ok_a = (b4.pred == b4.y_true).values; ok_b = (sl.pred == sl.y_true).values
disc = (ok_a != ok_b).mean()
mde = 1.96 * np.sqrt(disc / len(ok_a))
print(f"two similar stacks disagree on {disc:.1%} of battles")
print(f"=> minimum detectable difference at 5% (approximate McNemar): ±{mde*100:.2f} accuracy points")
print(f"   detecting +0.2 points at this disagreement rate requires about {int((1.96 / .002)**2 * disc):,} battles")
f = json.load(open(f"{E}/nested_base4/metrics.json"))["stack"]["acc_per_fold"]
print(f"nested estimate per fold: {np.round(f, 4)}, sd {np.std(f, ddof=1):.4f}: "
      f"best-to-worst fold difference is {(max(f)-min(f))*100:.1f} points")
''',
r'''md:**Interpretation:**

- The stack exceeds chance and simple single-indicator baselines (p much smaller than 0.001).
- The 95% absolute-accuracy CI is about +/-0.9 points: **0.70 lies inside the interval for 0.698**, so these data cannot reliably distinguish crossing 0.70 from not crossing it.
- Similar stacks need roughly +/-0.5 points for a significant paired difference. Recent experiments gain 0 to +0.3; detecting +0.2 would require about 60,000 battles, six times the training set.
- Accuracy varies by over one point across folds, so a single split can reach 0.70 by chance.

## Conclusions

1. The historical analysis found balanced classes, no obvious leakage, similar train/test distributions and a few duplicates requiring appropriate handling.
2. Signal is real and significant but moderate, concentrated in correlated HP/KO-advantage dimensions.
3. The flattening learning curve and KO imbalance before/after turn 30 suggest an information ceiling near 0.70.
4. Differences between tested variants fall below detectable levels; decisions should use paired estimates rather than isolated accuracies.''',
]

# --------------------------------------------------------------------------------------------
# Report 2: methods, models and architectures — benefit versus cost
# --------------------------------------------------------------------------------------------
R2 = [
r'''md:# Statistical report 2: methods, models and architectures — is the gain worth the cost?

Compare each component's **accuracy gain**, using paired significance tests on the same battles, with its **cost**: runtime on a ten-core Mac M5 and implementation complexity.

1. The stack's 22 base models.
2. The meta-model versus averaging and the best individual model.
3. Nested model-subset selection (a reduced stack).
4. Feature blocks and architectures: Monte Carlo, recursive model, augmentation, strategic matchups, per-Pokemon history and regime experts.
5. Fold and repetition policies.

Costs come from `outputs/folds_experiment_log.txt` and individual M5 pipeline timings. Accuracy estimates are five-fold OOF (seed 42) or nested (`src/evaluate_nested.py`).

Historical analysis: text is translated without recomputing results. Saved plot images remain unchanged; source labels are translated for future regeneration.''',
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
    """Logistic meta on logits: (mean fold accuracy, SD, averaged repeated OOF predictions)."""
    accs, pm = [], np.zeros(len(y))
    for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed).split(Z, y):
        p = LogisticRegression(C=1.0, max_iter=2000).fit(Z[a], y[a]).predict_proba(Z[b])[:, 1]
        pm[b] += p / repeats; accs.append(((p > .5) == y[b]).mean())
    return np.mean(accs), np.std(accs), pm

# Per-model timings (seconds, five folds) from the outputs/oof_mc_rec run
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
print(f"{len(names)} base models; total base-model time {sum(T.values())/60:.1f} min")
''',
r'''md:## 1. Base models: accuracy, overfitting and cost

`gap` is training accuracy minus OOF accuracy, a measure of overfitting. Cost includes five-fold OOF fitting and test prediction.''',
r'''
rows = []
for n in names:
    p = P[n]["oof"]; ok = (p > .5) == y
    rows.append((n, ok.mean(), roc_auc_score(y, p), float(P[n]["train_acc"]) - ok.mean(), T.get(n, np.nan)))
B = pd.DataFrame(rows, columns=["model", "acc OOF", "AUC", "gap", "time s"]).sort_values("acc OOF", ascending=False)
B.style.format({"acc OOF": "{:.4f}", "AUC": "{:.4f}", "gap": "{:+.3f}", "time s": "{:.0f}"}).background_gradient(subset=["acc OOF"], cmap="Greens")
''',
r'''
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(B["time s"], B["acc OOF"], c=B.gap, cmap="coolwarm", s=50)
for _, r in B.iterrows(): ax.annotate(r.model, (r["time s"], r["acc OOF"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
ax.set_xscale("log"); ax.set(xlabel="time (s, log scale)", ylabel="accuracy OOF", title="Base models: accuracy versus cost (color = train-OOF gap)")
plt.tight_layout(); plt.show()
''',
r'''md:**Interpretation:** most models lie in a narrow accuracy range (0.68–0.69) while costs vary by two orders of magnitude. Linear and generative models (lda, logreg_dense, gmm) take seconds and show little overfitting. CatBoost, ExtraTrees, RF and SVM take minutes without better standalone accuracy. Their value must be measured within the stack: a weaker but different model can contribute more than a strong redundant one.''',
r'''md:## 2. Does the meta-model help? Stacking versus averaging and the best single model

Paired comparisons on the same 10,000 battles use bootstrap accuracy-difference intervals and McNemar's test.''',
r'''
acc_s, sd_s, pm = meta_cv(Z, y)
stack_ok = (pm > .5) == y
best = B.iloc[0].model
refs = {f"best single model ({best})": (P[best]["oof"] > .5) == y,
        "mean logits (22)": (Z.mean(1) > 0) == y,
        "lgbm alone": (P["lgbm"]["oof"] > .5) == y,
        "logreg alone": (P["logreg"]["oof"] > .5) == y}
rows = []
for nm, ok in refs.items():
    d, lo, hi = paired_bootstrap(ok, stack_ok); b, c, p = mcnemar(ok, stack_ok)
    rows.append((nm, ok.mean(), d, lo, hi, c, b, p))
print(f"stack (meta logreg, 22 models): {acc_s:.4f} ± {sd_s:.4f} (5x5 CV)")
pd.DataFrame(rows, columns=["reference", "acc", "stack gain", "CI low", "CI high", "correct only with stack", "only with reference", "McNemar p"]).style.format(
    {"acc": "{:.4f}", "stack gain": "{:+.4f}", "CI low": "{:+.4f}", "CI high": "{:+.4f}", "McNemar p": "{:.3f}"})
''',
r'''md:**Interpretation:** the stack significantly beats every single model, including xgb_aug (+0.6 points, p about 0.04), but does **not** significantly beat simple mean logits (+0.3 points, p about 0.19). Meta fitting takes seconds, so it is still worthwhile; most gains come from ensembling rather than learned weights.''',
r'''md:## 3. Model diversity: error correlation

Stacking helps when models make errors on different battles. Correlate OOF residuals (y - p), ordered by hierarchical clustering.''',
r'''
from scipy.cluster.hierarchy import linkage, leaves_list
R = np.column_stack([y - P[n]["oof"] for n in names])
C = np.corrcoef(R.T)
order = leaves_list(linkage(R.T, "average", metric="correlation"))
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(C[np.ix_(order, order)], cmap="viridis", vmin=.4, vmax=1)
ax.set_xticks(range(len(names))); ax.set_xticklabels(np.array(names)[order], rotation=90, fontsize=7)
ax.set_yticks(range(len(names))); ax.set_yticklabels(np.array(names)[order], fontsize=7)
plt.colorbar(im); ax.set_title("OOF residual correlation"); plt.tight_layout(); plt.show()
off = C[np.triu_indices(len(names), 1)]
print(f"mean residual correlation: {off.mean():.2f} (min {off.min():.2f}, max {off.max():.2f})")
''',
r'''md:## 4. Marginal value of each model: ablation

Refit the meta-model without each base model on the **same splits** (5 x 5). A negative removal difference means that dropping the model hurts. Comparing contribution with runtime measures the cost of each tenth of an accuracy point.''',
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
A = pd.DataFrame(rows, columns=["model", "contribution", "se", "time s"]).sort_values("contribution", ascending=False)
A["points per minute"] = A.contribution * 100 / (A["time s"] / 60)
print(f"full stack: {full.mean():.4f}")
A.style.format({"contribution": "{:+.4f}", "se": "{:.4f}", "time s": "{:.0f}", "points per minute": "{:+.3f}"}).bar(subset=["contribution"], align="zero", color=["#d65f5f", "#5fba7d"])
''',
r'''md:**Interpretation:**

- Residual correlations average 0.97: models usually fail on the same battles, and individual contributions are small (at most +0.3 points).
- Contrary to standalone rankings, expensive, weaker models contribute most: ExtraTrees, RF and augmented CatBoost provide diversity through deep-tree bagging and different fitting behavior.
- Redundant lgbm, xgb, svm, non-augmented catboost/catboost_dense, GMM and lgbm_noise have zero or negative contributions.
- Standard errors from 25 repeated splits of the same data are underestimated; differences below about 0.1 points should be treated as negligible.''',
r'''md:## 5. Can a reduced stack match the full stack? Nested selection

Greedy forward selection occurs **inside** each outer fold, using inner CV on the outer training partition and evaluation on the held-out outer fold. This avoids the optimism of selecting and evaluating on the same CV. Stop when adding a model improves inner CV by less than 0.05 points.''',
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
print(f"full stack (22): {ok_f.mean():.4f}, base-model time {sum(T.values())/60:.1f} min")
print(f"reduced stack (mean {np.mean([len(p) for p in picks]):.1f} models): {ok_s.mean():.4f}, time {t_small/60:.1f} min")
print(f"reduced - full difference: {d*100:+.2f} points, 95% CI [{lo*100:+.2f}, {hi*100:+.2f}], McNemar p = {pv:.2f}")
print("selected models (out of five outer folds):", dict(cnt.most_common()))
''',
r'''md:Greedy selection stops too soon. Also compare fixed subsets using the full stack's splits. The first two depend only on cost. The last two are suggested by the same-data ablation in section 4, so their estimates are somewhat optimistic and require nested confirmation.''',
r'''
subsets = {
    "cheap (<= 20 s each)": [n for n in names if T[n] <= 20],
    "four classical models (logreg, lgbm, xgb, catboost)": ["logreg", "lgbm", "xgb", "catboost"],
    "without svm or non-augmented catboost models": [n for n in names if n not in ("svm", "catboost", "catboost_dense")],
    "augmented boosting + ET + RF + linear models only": [n for n in names if n.endswith("_aug") or n in ("extratrees", "rf", "lda", "logreg", "logreg_dense")],
}
rows = []
for nm, ms in subsets.items():
    cols = [names.index(m) for m in ms]
    _, _, p_ = meta_cv(Z[:, cols], y)
    ok = (p_ > .5) == y
    d, lo, hi = paired_bootstrap(stack_ok, ok); _, _, pv = mcnemar(stack_ok, ok)
    t = sum(T[m] for m in ms)
    rows.append((nm, len(ms), acc_on(cols).mean(), d * 100, lo * 100, hi * 100, pv, t / 60, 1 - t / sum(T.values())))
SUB = pd.DataFrame(rows, columns=["subset", "models", "acc", "Δ vs 22", "CI low", "CI high", "McNemar p", "time min", "saving"])
SUB.style.format({"acc": "{:.4f}", "Δ vs 22": "{:+.2f}", "CI low": "{:+.2f}", "CI high": "{:+.2f}", "McNemar p": "{:.3f}", "time min": "{:.1f}", "saving": "{:.0%}"})
''',
r'''md:**Interpretation:**

- Cheap models or the four classical models significantly reduce accuracy (-0.6 / -0.9 points). Nested greedy selection also loses -0.6 points (p about 0.02); saving 86% runtime costs accuracy.
- Removing redundant models (svm, non-augmented catboost and lgbm/xgb/gmm) preserves accuracy: ten models save 43% runtime for +0.15 points, not statistically significant.
- Diversity matters more than quantity: retain augmented boosting, ExtraTrees, RF and linear models, and remove redundant components.''',
r'''md:## 6. Feature blocks and architectures

Each row compares a variant with its baseline **on the same folds and battles**:

- **Monte Carlo and recursive features:** v3, 17-model stack and individual lgbm/xgb/lda/gmm/logreg_dense models; standard OOF with meta refitted on matching splits.
- **Augmentation:** v4 with and without five `_aug` models.
- **Strategic matchups, per-Pokemon history, regime experts and augmentation weights:** nested estimates from `evaluate_nested.py`, using four or three base models.
- **Strat + lon in the complete 22-model stack.**

Cost is additional feature-generation time plus fitting time for added models.''',
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

# Monte Carlo + recursion in the v3 17-model stack (matching folds)
v3, v3mr = load_oof("outputs/oof_v3"), load_oof("outputs/oof_v3_mc_rec")
add("Monte Carlo + recursive (17-model stack)", *compare_stacks(v3, v3mr, sorted(v3), sorted(v3)), 190 + 122, "OOF")
for extra, cost in (("mc", 190), ("rec", 122)):
    Pe = load_oof(f"outputs/oof_v3_{extra}"); ms = sorted(Pe)
    add(f"{'Monte Carlo' if extra == 'mc' else 'recursive'} alone (stack {len(ms)})", *compare_stacks(v3, Pe, ms, ms), cost, "OOF")
# Augmentation: v4 without/with _aug models (same features)
v4 = load_oof("outputs/oof_v4a", [n for n in names if n in os.listdir("outputs/oof_v4a") or f"{n}.npz" in os.listdir("outputs/oof_v4a")]); no_aug = sorted(n for n in v4 if "_aug" not in n)
aug_names = sorted(n for n in v4 if "_aug" in n)
add(f"augmentation ({len(aug_names)} models _aug)", *compare_stacks(v4, v4, no_aug, sorted(v4)), sum(T[n] for n in aug_names), "OOF")
# strat + lon in the full stack
Psl = load_oof("outputs/oof_mc_rec_strat_lon")
add("strat + lon (stack 22)", *compare_stacks(P, Psl, names, names), 121 + 14, "OOF")
# Nested experiments
base4 = nested_ok("nested_base4")
for name, label, cost in (("nested_base4_strat", "strategic matchups + P(KO) + pool", 121),
                          ("nested_base4_strat_nopool", "strategic matchups without pool", 121),
                          ("nested_base4_lon", "per-Pokemon history", 14),
                          ("nested_base4_strat_lon", "strategic + history", 135),
                          ("nested_base4_exp", "regime experts", 3 * 15),
                          ("nested_base4_exp_gate", "experts + gating", 3 * 15)):
    add(label, base4, nested_ok(name), cost, "nested")
w5 = nested_ok("nested_aug_w0.5")
for w in ("0.25", "1.0"):
    add(f"augmentation weight {w} (versus 0.5)", w5, nested_ok(f"nested_aug_w{w}"), 0, "nested")
D = pd.DataFrame(rows, columns=["component", "estimate", "acc base", "variant acc", "Δ points", "CI low", "CI high", "McNemar p", "cost s"])
D.style.format({"acc base": "{:.4f}", "variant acc": "{:.4f}", "Δ points": "{:+.2f}", "CI low": "{:+.2f}", "CI high": "{:+.2f}", "McNemar p": "{:.3f}", "cost s": "{:.0f}"}).bar(subset=["Δ points"], align="zero", color=["#d65f5f", "#5fba7d"])
''',
r'''
fig, ax = plt.subplots(figsize=(9, 4.8))
yy = np.arange(len(D))[::-1]
ax.errorbar(D["Δ points"], yy, xerr=[D["Δ points"] - D["CI low"], D["CI high"] - D["Δ points"]], fmt="o", capsize=3)
ax.axvline(0, color="gray", ls="--"); ax.set_yticks(yy); ax.set_yticklabels(D.component, fontsize=8)
ax.set(xlabel="Accuracy difference (points), 95% paired bootstrap CI", title="Effect of each component relative to its baseline")
plt.tight_layout(); plt.show()
''',
r'''md:**Interpretation:**

- **No component** has a paired CI entirely above zero.
- **Augmentation** comes closest (+0.30 points, p about 0.12), and is simple: reuse features at turns 24/26/28.
- **Monte Carlo and recursion:** no gain in the v3 17-model stack (-0.03). Some single models improve about +0.3 points, but the ensemble already captures the information. These methods cost about five minutes and substantial code complexity.
- **Strategic matchups, team pool and per-Pokemon history:** +0.1/+0.2 points without demonstrated significance. History is the only inexpensive component (14 seconds).
- **Regime experts** and **augmentation weight 1.0** hurt. Weight 0.25 looks better in a three-model stack (p about 0.10), but not in the full stack.''',
r'''md:## 7. Folds and repetitions

Three base-model configurations (22 models, mc + rec features) share the same meta-evaluation splits. Base OOF partitions differ, so comparisons are paired by battle but involve models trained on different partitions.''',
r'''
rows = []
ok0 = None
for d, sec in (("outputs/oof_mc_rec", "stacking folds=5 repeats=1"), ("outputs/oof_mc_rec_k10r1", "stacking folds=10 repeats=1"),
               ("outputs/oof_mc_rec_k5r3", "stacking folds=5 repeats=3")):
    Pd = load_oof(d); a, sd, pm_ = meta_cv(to_Z(Pd, sorted(Pd)), y); ok = (pm_ > .5) == y
    if ok0 is None: ok0 = ok
    dd, lo, hi = paired_bootstrap(ok0, ok)
    rows.append((sec.replace("stacking ", ""), a, sd, dd * 100, lo * 100, hi * 100, TOTALS.get(sec, np.nan) / 60))
pd.DataFrame(rows, columns=["configuration", "meta acc", "sd", "Δ vs 5x1", "CI low", "CI high", "time min"]).style.format(
    {"meta acc": "{:.4f}", "sd": "{:.4f}", "Δ vs 5x1": "{:+.2f}", "CI low": "{:+.2f}", "CI high": "{:+.2f}", "time min": "{:.1f}"})
''',
r'''md:**Interpretation:** neither ten folds nor three repetitions improve the stack. Both are worse, without significant differences, and cost 2–3 times as much. Repetition bagging improves individual models but makes OOF predictions more alike, reducing meta-model diversity.''',
r'''md:## 8. Summary: benefit versus cost

Automatic component verdict:

- **Worthwhile:** paired CI entirely above zero.
- **Neutral, cheap:** CI includes zero, cost below one minute and nonnegative point estimate.
- **Gain not established:** CI includes zero, positive point estimate and high cost.
- **Not worthwhile:** negative point estimate.''',
r'''
def verdict(r):
    if r["CI low"] > 0: return "worthwhile"
    if r["Δ points"] < 0: return "not worthwhile"
    return "neutral, cheap" if r["cost s"] < 60 else "gain not established"
S = D.copy(); S["verdict"] = S.apply(verdict, axis=1)
S["points per minute"] = np.where(S["cost s"] > 0, S["Δ points"] / (S["cost s"] / 60), np.nan)
print(f"full 22-model stack: {acc_s:.4f} (base model {sum(T.values())/60:.1f} min)  |  "
      f"nested reduced stack: {ok_s.mean():.4f} ({t_small/60:.1f} min)")
S[["component", "Δ points", "CI low", "CI high", "McNemar p", "cost s", "points per minute", "verdict"]].style.format(
    {"Δ points": "{:+.2f}", "CI low": "{:+.2f}", "CI high": "{:+.2f}", "McNemar p": "{:.3f}", "cost s": "{:.0f}", "points per minute": "{:+.2f}"})
''',
r'''md:## Conclusions

- **Ensembling:** the demonstrated gain. Stacking significantly beats the best single model (+0.6 points) and cheap subsets (+0.6/+0.9). Logistic meta fitting adds little over mean logits but takes seconds.
- **Base-model choice:** diversity matters more than count or standalone accuracy. ExtraTrees, RF and augmented CatBoost contribute most. SVM, GMM, non-augmented lgbm/xgb/catboost and noise variants can be removed, saving about 40% runtime without measurable loss, subject to nested confirmation.
- **Snapshot augmentation:** the best benefit/cost component (+0.3 points, borderline significance, about 3.5 minutes).
- **Monte Carlo and recursion:** no measurable stack gain, about five minutes and substantial complexity; useful for analysis rather than established accuracy improvement.
- **Strategic matchups, team pool and Pokemon history:** real but redundant signal (+0.1/+0.2, not established). Only history is cheap enough to retain readily.
- **Regime experts, ten folds and repeated k-fold:** cost without demonstrated return.
- **Summary:** these experiments attribute the ceiling mainly to available information. Beyond diverse ensembling and augmentation, no tested addition justifies its cost.''',
]


def main():
    for fname, cells in (("01_report_statistico_dati.ipynb", R1), ("02_report_costi_benefici.ipynb", R2)):
        nbf.write(notebook(cells), os.path.join(HERE, fname))
        print("written", fname)


if __name__ == "__main__":
    main()
