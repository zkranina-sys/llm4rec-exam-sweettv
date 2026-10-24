"""Diagnostics (fold A, rule 'new').
Part 1 - CBF: is the low MAP a code bug or a data property?
  1a independent recomputation of one cosine from raw token lists (pure Python, no matrices)
  1b what CBF puts in the Top-5: familiar (watched in train) vs new, popularity of the items, ties
  1c where hits come from: precision of familiar vs new items in each method's Top-5
  1d ablations: which attribute group carries signal; no IDF
  1e attribute coverage
Part 2 - validation truth: why popularity scores 0.66
  2a definition and sizes, 2b how many users watched each popular show, 2c random baseline
"""
import sys, math
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
import lib
from lib import *
from methods import *

print("watched rule:", lib.RULE)
inter, sch, test_users = load()
TR, VA = list(range(11, 25)), list(range(25, 31))
tr, truth = split(inter, TR, VA)
cand = candidates(sch, VA)
EU = [u for u in test_users if u in truth]
d = Data(tr, cand, EU, weight="weeks")
fill = list(d.cand[np.argsort(-d.pop[d.cand_col], kind="stable")][:50])
F, vocab = item_features(sch, d.items, use_channel=True)
S_cbf = score_cbf(d, F)
recs = {"Популярность": rank5(d, score_pop(d), fill), "История": rank5(d, score_hist(d), fill),
        "CBF": rank5(d, S_cbf, fill), "PPR": rank5(d, score_ppr(d, alpha=0.7), fill)}
print("check: MAP@5 CBF =", round(mapk(truth, recs["CBF"], EU), 4))

# ---------- 1a independent cosine ----------
inv = {j: t for t, j in vocab.items()}
Fd = F.tocsr()
def vec(it):  # rebuild tf-idf vector of an item from the matrix row into a {token: weight} dict
    row = Fd[d.iix[it]]
    return {inv[j]: v for j, v in zip(row.indices, row.data)}
u0 = EU[0]; r0 = 0
H = d.user_hist().toarray()[r0]
prof = Counter()
for col in np.nonzero(H)[0]:
    for t, x in vec(d.items[col]).items():
        prof[t] += H[col] * x
pn = math.sqrt(sum(x * x for x in prof.values()))
j_top = int(np.argmax(S_cbf[r0])); it = int(d.cand[j_top])
cv = vec(it)
cos = sum(prof.get(t, 0) * x for t, x in cv.items()) / pn / math.sqrt(sum(x * x for x in cv.values()))
print(f"\n1a user {u0}: CBF top item {it}; matrix score {S_cbf[r0, j_top]:.6f}, pure-Python cosine {cos:.6f}")
raw = sch[sch.tv_show_id == it].iloc[0]
print("   raw attributes of that item:", {k: raw[k] for k in ['tv_show_title', 'tv_show_category', 'tv_show_genre_1', 'director', 'actors', 'year_of_production']})

# ---------- 1b/1c ----------
seen = watched(tr).groupby("user_id").tv_show_id.agg(set).to_dict()
pop_rank = pd.Series(np.arange(len(fill)), index=fill)
popc = pd.Series(d.pop[d.cand_col], index=d.cand)
print("\n1b/1c Top-5 composition (averages over", len(EU), "users):")
print(f"{'method':14s} {'familiar share':>15s} {'P@5 familiar':>13s} {'P@5 new':>9s} {'median users/item':>18s}")
for k, r in recs.items():
    fam_n = fam_h = new_n = new_h = 0; pops = []
    for u in EU:
        for i in r[u]:
            f = i in seen.get(u, ())
            h = i in truth[u]
            if f: fam_n += 1; fam_h += h
            else: new_n += 1; new_h += h
            pops.append(popc.get(i, 0))
    print(f"{k:14s} {fam_n/(fam_n+new_n):15.3f} {fam_h/max(fam_n,1):13.3f} {new_h/max(new_n,1):9.3f} {np.median(pops):18.0f}")
ties = np.mean([np.sort(S_cbf[r])[-5] == np.sort(S_cbf[r])[-6] for r in range(len(EU))])
print(f"   CBF: share of users whose 5th and 6th scores are exactly tied: {ties:.3f}")
rep = np.mean([len(truth[u] & seen.get(u, set())) / len(truth[u]) for u in EU])
print(f"   share of a user's validation shows already watched in train: {rep:.3f}")

# ---------- 1d ablations ----------
def feats_subset(prefixes, idf=True):
    F2, voc2 = item_features(sch, d.items, use_channel=True)
    keep = [j for t, j in voc2.items() if t.split("=")[0] in prefixes]
    G = F2[:, keep]
    if not idf:
        G = G.copy(); G.data[:] = 1.0
    n = np.sqrt(np.asarray(G.multiply(G).sum(1)).ravel()); n[n == 0] = 1
    return sp.diags(1 / n) @ G
print("\n1d CBF ablations (MAP@5):")
for name, pref, idf in [("all attributes, TF-IDF (used)", {"cat", "gen", "dir", "act", "dec", "ch"}, True),
                        ("all attributes, binary (no IDF)", {"cat", "gen", "dir", "act", "dec", "ch"}, False),
                        ("without channel", {"cat", "gen", "dir", "act", "dec"}, True),
                        ("category + genres only", {"cat", "gen"}, True),
                        ("director + actors only", {"dir", "act"}, True),
                        ("channel only", {"ch"}, True)]:
    G = feats_subset(pref, idf).tocsr()
    print(f"   {name:35s} {mapk(truth, rank5(d, score_cbf(d, G), fill), EU):.4f}")

# ---------- 1e coverage ----------
first = sch.drop_duplicates("tv_show_id").set_index("tv_show_id").reindex(d.cand)
for c in ["tv_show_genre_1", "director", "actors", "year_of_production"]:
    print(f"1e candidates with non-empty {c}: {(first[c].fillna('') != '').mean():.3f}")
print("1e category shares among candidates:", first.tv_show_category.value_counts(normalize=True).round(3).to_dict())
print("1e category shares among validation hits (all users' truth):",
      pd.Series([first.tv_show_category.get(i) for u in EU for i in truth[u]]).value_counts(normalize=True).round(3).to_dict())

# ---------- Part 2 ----------
sizes = np.array([len(truth[u]) for u in EU])
print(f"\n2a truth = distinct tv_show_ids with (overlap >= 300 s OR >= 90% of airing) in weeks 25-30; "
      f"users {len(EU)}, |R_u| median {np.median(sizes):.0f}, 10% {np.quantile(sizes,.1):.0f}, 90% {np.quantile(sizes,.9):.0f}; "
      f"candidates {len(cand)}; |R_u|/|cand| median {np.median(sizes)/len(cand):.4f}")
top5 = recs["Популярность"][EU[0]]
meta = sch.drop_duplicates("tv_show_id").set_index("tv_show_id")
print("2b popularity Top-5 and share of evaluation users who watched it in validation:")
for i in top5:
    print(f"   {i:9d} {meta.tv_show_title[i][:40]:40s} {np.mean([i in truth[u] for u in EU]):.3f}")
ap = np.array([apk(truth[u], recs['Популярность'][u]) for u in EU])
print(f"2b popularity AP@5 distribution: share AP=1: {(ap==1).mean():.3f}, AP=0: {(ap==0).mean():.3f}")
rng = np.random.default_rng(42)
cl = list(cand)
rand = {u: list(rng.choice(cl, K, replace=False)) for u in EU}
print(f"2c random 5 candidates (seed 42): MAP@5 = {mapk(truth, rand, EU):.4f}")
