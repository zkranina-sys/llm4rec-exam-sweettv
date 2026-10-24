"""Step 7: verification.
(1) Label-threshold sensitivity on fold A: re-score the SAME Top-5 lists of each method against truth built with
    other "watched" thresholds (60 s, 900 s, 1800 s, >=50% of the airing). Models are unchanged (trained at 300 s).
(2) PPR check: power iteration vs exact linear solve p = (1-a)(I - a M)^-1 v for 3 users.
(3) Hand-checkable AP@5 for one user: print the list, the hit flags, and the arithmetic.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
from lib import *
from methods import *
import importlib; ltr = importlib.import_module("50_ltr")

inter, sch, test_users = load()
TRb, VAb = list(range(11, 19)), list(range(19, 25))
TR, VA = list(range(11, 25)), list(range(25, 31))
tr, truth = split(inter, TR, VA)
cand = candidates(sch, VA)
EU = [u for u in test_users if u in truth]
d = Data(tr, cand, EU, weight="weeks")
fill = list(d.cand[np.argsort(-d.pop[d.cand_col], kind="stable")][:50])
F, _ = item_features(sch, d.items, use_channel=True)
P = score_ppr(d, alpha=0.7)
recs = {"popularity": rank5(d, score_pop(d), fill), "history": rank5(d, score_hist(d), fill),
        "CBF": rank5(d, score_cbf(d, F), fill), "PPR": rank5(d, P, fill)}
dfB = pd.read_pickle("out/ltr_feats_B.pkl"); dfA = pd.read_pickle("out/ltr_feats_A.pkl")
sc, m = ltr.fit(dfB)
recs["LR fusion"] = ltr.to_recs(dfA, m.predict_proba(sc.transform(dfA[ltr.FEATS].values))[:, 1], fill)

va = inter[inter.week.isin(VA)]
defs = {"ov>=60s": va.ov >= 60, "ov>=300s (main)": va.ov >= 300, "ov>=900s": va.ov >= 900,
        "ov>=1800s": va.ov >= 1800, "frac>=0.5": va.frac >= 0.5}
tab = {}
for name, mask in defs.items():
    t = va[mask].groupby("user_id").tv_show_id.agg(lambda x: set(x.tolist())).to_dict()
    tab[name] = {k: round(mapk(t, r, EU), 4) for k, r in recs.items()}
    sizes = [len(t.get(u, ())) for u in EU]
    print(f"{name:16s} median relevant/user {np.median(sizes):5.0f} | " + " | ".join(f"{k} {v:.4f}" for k, v in tab[name].items()))
json.dump(tab, open("out/threshold_sensitivity.json", "w"), indent=1)

# (2) PPR exact check
Q, dang = coview_transition(d)
Q = Q.toarray()
n = Q.shape[0]
H = d.user_hist().toarray()
for r in [0, 1, 2]:
    v = H[r] / H[r].sum()
    M = Q + np.outer(v, dang.astype(float))  # dangling mass returns to v
    exact = 0.3 * np.linalg.solve(np.eye(n) - 0.7 * M, v)
    it = np.zeros(n); it[:] = 0
    approx_full = None
    # compare on candidate columns with what score_ppr returned
    err = np.abs(exact[d.cand_col] - P[r]).sum()
    print(f"PPR user row {r}: L1(power iteration vs exact solve) on candidates = {err:.2e}; exact sums to {exact.sum():.6f}; "
          f"top-5 equal: {list(np.argsort(-exact[d.cand_col], kind='stable')[:5]) == list(np.argsort(-P[r], kind='stable')[:5])}")
print("PPR iterations used:", d.ppr_iters)

# (3) hand-checkable AP@5: choose the first evaluation user whose LR list has a miss before a hit
names = sch.drop_duplicates("tv_show_id").set_index("tv_show_id").tv_show_title
for u in EU:
    lst = recs["LR fusion"][u]; flags = [int(i in truth[u]) for i in lst]
    if 0 < sum(flags) < 5 and flags[0] == 1 and 0 in flags[:4]:
        break
print(f"\nHand check user {u}: |relevant in val| = {len(truth[u])}, denominator min(5, {len(truth[u])}) = {min(5, len(truth[u]))}")
hits, s = 0, 0.0
for k, (i, f) in enumerate(zip(lst, flags), 1):
    if f:
        hits += 1; s += hits / k
    print(f"  rank {k}: {i} {names.get(i, '?')[:40]!r} relevant={f}" + (f"  precision@{k} = {hits}/{k}" if f else ""))
print(f"  AP@5 = {s:.4f} / {min(5, len(truth[u]))} = {s / min(5, len(truth[u])):.4f}; apk() returns {apk(truth[u], lst):.4f}")
