"""Single source of every number in the report (rule 'new').
Fold A: train weeks 11-24 -> validation 25-30; fold B: train 11-18 -> validation 19-24.
MAP@5 over all 1260 test users (empty answer key -> AP 0). Two answer keys: K1 = ours (airing >= 300 s or >= 90 %),
K2 = user's top-5 shows by number of distinct days watched in the window (calibrated to the Kaggle popularity score).
LR fusion v1 (15 features) and final v2 (+2 daily features): fit on B, evaluated on A; reverse: fit on A, evaluated on B.
Also: PPR alpha grid, linear hybrid grid (chosen on A, in-sample), Top-5 of every method for example users A and B."""
import sys, runpy, io, contextlib, time, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
T0 = time.time()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path("/home/claude/work/92_crosskey.py")  # fold-A lists, dfA/dfB, fit_predict, day_feats (verified there)
G = lambda k: g[k]
apk, mapk, users, inter, key_fn, dfA, dfB, fillA, fill, hist_users, recs, ltr, K = (G(k) for k in
    ["apk", "mapk", "users", "inter", "key_fn", "dfA", "dfB", "fillA", "fill", "hist_users", "recs", "ltr", "K"])
fit_predict, day_feats, F2 = G("fit_predict"), G("day_feats"), G("F2")
g90 = g["g"]  # namespace of 90_keys.py as loaded inside 92_crosskey.py
d, N, comp, sch = g90["d"], g90["N"], g90["comp"], g90["sch"]
from methods import score_ppr, rank5
from lib import watched, split, candidates
from methods import Data
TRb, VAb, TR, VA = list(range(11, 19)), list(range(19, 25)), list(range(11, 25)), list(range(25, 31))
KEY1, KEY2 = ("ours", None), ("topk_days", 5)
tA = {k: key_fn(k)(inter[inter.week.isin(VA)]) for k in (KEY1, KEY2)}
tB = {k: key_fn(k)(inter[inter.week.isin(VAb)]) for k in (KEY1, KEY2)}

# --- data accounting for fold A
print("=== accounting (fold A, rule new)")
print("candidates (labelled shows airing in weeks 25-30):", len(d.cand))
print("test users:", len(users), "| with watched history in train 11-24:", len(hist_users & set(users)),
      "| with >=1 answer under K1:", sum(u in tA[KEY1] for u in users), "| under K2:", sum(u in tA[KEY2] for u in users))
print("median |R_u| K1:", np.median([len(tA[KEY1][u]) for u in users if u in tA[KEY1]]),
      "K2:", np.median([len(tA[KEY2][u]) for u in users if u in tA[KEY2]]))

# --- PPR alpha grid on fold A
print("\n=== PPR alpha grid (fold A)")
for a in [0.15, 0.3, 0.5, 0.7, 0.85]:
    t = time.time(); r = rank5(d, score_ppr(d, alpha=a), fill)
    print(f"alpha {a}: K1 {mapk(tA[KEY1], r, users):.4f} | K2 {mapk(tA[KEY2], r, users):.4f} | iters {d.ppr_iters} | {time.time()-t:.0f} s")

# --- linear hybrid (weights chosen on fold A under K1: in-sample)
best, bw = -1, None
for b, c, e in itertools.product([0, 0.05, 0.1, 0.2, 0.4, 0.8], [0, 0.25, 0.5, 1.0], [0, 0.05, 0.1, 0.2]):
    m_ = mapk(tA[KEY1], rank5(d, N["hist"] + b * N["pop"] + c * N["ppr"] + e * N["cbf"], fill), users)
    if m_ > best: best, bw = m_, (b, c, e)
r_lin = rank5(d, N["hist"] + bw[0] * N["pop"] + bw[1] * N["ppr"] + bw[2] * N["cbf"], fill)

# --- LR v1 / v2
r_v1 = recs["Слияние LR (текущий финал)"]
dfA2, dfB2 = day_feats(dfA, TR), day_feats(dfB, TRb)
r_v2 = fit_predict(dfB2, dfA2, F2)
allrecs = {"Popularity": recs["Популярность"], "History (weeks)": recs["История"], "CBF": recs["CBF"],
           "PPR a=0.7": recs["PPR"], f"Linear hybrid {bw}": r_lin, "LR fusion v1 (final)": r_v1, "LR fusion v2": r_v2}
print("\n=== main table: fold A, all 1260 users")
pop1, pop2 = mapk(tA[KEY1], allrecs["Popularity"], users), mapk(tA[KEY2], allrecs["Popularity"], users)
for k, r in allrecs.items():
    m1, m2 = mapk(tA[KEY1], r, users), mapk(tA[KEY2], r, users)
    print(f"{k:32s} K1 {m1:.4f} ({(m1/pop1-1)*100:+.1f}% vs pop) | K2 {m2:.4f} ({(m2/pop2-1)*100:+.1f}% vs pop)")

# --- reverse direction: fit on A, evaluate on B (fold B lists for single methods too)
print("\n=== fold B (train 11-18, val 19-24), all 1260 users")
trB = inter[inter.week.isin(TRb)]
dB = Data(trB, candidates(sch, VAb), users, weight="weeks")
fillB = list(dB.cand[np.argsort(-dB.pop[dB.cand_col], kind="stable")][:50])
from methods import score_pop, score_hist
import lib
for nm, S in [("Popularity", score_pop(dB)), ("History (weeks)", score_hist(dB)), ("PPR a=0.7", score_ppr(dB, alpha=0.7))]:
    r = rank5(dB, S, fillB)
    print(f"{nm:32s} K1 {mapk(tB[KEY1], r, users):.4f} | K2 {mapk(tB[KEY2], r, users):.4f}")
dfB_full, fillB2, _ = ltr.build(inter.drop(columns="day"), sch, TRb, VAb, users, tB[KEY1])
hist_B = set(watched(trB).user_id)
def fp(train_df, test_df, feats):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(train_df[feats].values)
    m = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(sc.transform(train_df[feats].values), train_df.y.values)
    r = ltr.to_recs(test_df, m.predict_proba(sc.transform(test_df[feats].values))[:, 1], fillB2)
    for u in users:
        if u not in hist_B: r[u] = [int(x) for x in fillB[:K]]
    return r
for nm, tr_df, te_df, feats in [("LR fusion v1 (fit on A)", dfA, dfB_full, ltr.FEATS),
                                ("LR fusion v2 (fit on A)", dfA2, day_feats(dfB_full, TRb), F2)]:
    r = fp(tr_df, te_df, feats)
    print(f"{nm:32s} K1 {mapk(tB[KEY1], r, users):.4f} | K2 {mapk(tB[KEY2], r, users):.4f}")

# --- example users
meta = sch.drop_duplicates("tv_show_id").set_index("tv_show_id")
w = watched(inter[inter.week.isin(TR)])
for lab, u in [("A", 2725163102947201526), ("B", 8150816946177670084)]:
    wu = w[w.user_id == u]
    print(f"\n=== user {lab} {u}: train airings {len(wu)}, shows {wu.tv_show_id.nunique()}, weeks {wu.week.nunique()}, "
          f"|R| K1 {len(tA[KEY1].get(u, ()))}, |R| K2 {len(tA[KEY2].get(u, ()))}")
    top = wu.groupby("tv_show_id").week.nunique().sort_values(ascending=False).head(5)
    print("   most regular train shows:", [(int(i), meta.tv_show_title[i][:30], int(n)) for i, n in top.items()])
    print("   K2 answers:", [(int(i), meta.tv_show_title[i][:30]) for i in sorted(tA[KEY2].get(u, ()))])
    for k, r in allrecs.items():
        lst = r[u]
        f1 = "".join("1" if i in tA[KEY1].get(u, ()) else "0" for i in lst)
        f2 = "".join("1" if i in tA[KEY2].get(u, ()) else "0" for i in lst)
        print(f"   {k:32s} AP K1 {apk(tA[KEY1].get(u, set()), lst):.3f} hits {f1} | AP K2 {apk(tA[KEY2].get(u, set()), lst):.3f} hits {f2}")
        for i in lst:
            seen = "seen" if i in set(wu.tv_show_id) else "new "
            print(f"        {i:9d} {seen} {meta.tv_show_category[i]:14s} {meta.tv_show_title[i][:55]}")
print(f"\ntotal time {time.time()-T0:.0f} s")
