"""Calibrating the validation answer key to ONE Kaggle number (popularity public 0.28078), as decided by Zlata.
Models are NOT retrained for step 1: the fold-A Top-5 lists of each method are fixed; only the answer key changes.
MAP@5 is computed over ALL 1260 test users (users with an empty key score 0), as Kaggle presumably does.
Step 2 (for keys whose popularity MAP is within 0.03 of 0.281): all methods, linear hybrid weights re-chosen on
fold A under that key, and LR fusion re-fitted on fold B labels under that key (features unchanged), evaluated on A.
Kaggle's final-method score (0.41170) is NOT used for anything here."""
import sys, itertools, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
import lib
from lib import *
from methods import *
import importlib; ltr = importlib.import_module("50_ltr")

KAGGLE_POP = 0.28078
inter, sch, test_users = load()
inter = inter.assign(day=inter.st // 86400)
users = list(test_users)
TRb, VAb = list(range(11, 19)), list(range(19, 25))
TR, VA = list(range(11, 25)), list(range(25, 31))

def T(df): return df.groupby("user_id").tv_show_id.agg(lambda x: set(x.tolist())).to_dict()
def key_fn(name):
    kind, val = name
    def f(win):
        if kind == "ours":
            return T(watched(win))
        if kind == "airing_sec":
            return T(win[win.ov >= val])
        g = win.groupby(["user_id", "tv_show_id"]).agg(sec=("ov", "sum"), days=("day", "nunique")).reset_index()
        if kind == "show_sec":
            return T(g[g.sec >= val])
        if kind == "show_days":
            return T(g[g.days >= val])
        if kind == "topk_sec":
            return T(g.sort_values(["user_id", "sec", "tv_show_id"], ascending=[True, False, True]).groupby("user_id").head(val))
        if kind == "topk_days":
            g2 = g[g.sec >= 300]
            return T(g2.sort_values(["user_id", "days", "sec"], ascending=[True, False, False]).groupby("user_id").head(val))
    return f
KEYS = [("ours", None)] + [("airing_sec", t) for t in (1800, 3600)] + [("show_sec", t) for t in (1800, 3600, 7200, 14400)] \
     + [("show_days", d) for d in (2, 3, 5, 8)] + [("topk_sec", k) for k in (5, 10, 20)] + [("topk_days", k) for k in (5, 10, 20)]
label = lambda k: {"ours": "наше: показ ≥300 с или ≥90%", "airing_sec": f"показ ≥ {k[1]} с",
                   "show_sec": f"передача суммарно ≥ {k[1]} с", "show_days": f"передача ≥ {k[1]} разных дней",
                   "topk_sec": f"топ-{k[1]} передач по времени", "topk_days": f"топ-{k[1]} передач по дням"}[k[0]]

# fixed fold-A lists (rule 'new'), same pipeline as before
tr = inter[inter.week.isin(TR)]
cand = candidates(sch, VA)
d = Data(tr, cand, users, weight="weeks")
fill = list(d.cand[np.argsort(-d.pop[d.cand_col], kind="stable")][:50])
F, _ = item_features(sch, d.items, use_channel=True)
comp = {"hist": score_hist(d), "pop": score_pop(d).astype(float), "ppr": score_ppr(d, alpha=0.7), "cbf": score_cbf(d, F)}
N = {k: minmax_rows(v) for k, v in comp.items()}
hist_users = set(watched(tr).user_id)
truth_ours_A = key_fn(("ours", None))(inter[inter.week.isin(VA)])
truth_ours_B = key_fn(("ours", None))(inter[inter.week.isin(VAb)])
dfB, _, _ = ltr.build(inter, sch, TRb, VAb, [u for u in users if u in truth_ours_B], truth_ours_B)
dfA, fillA, _ = ltr.build(inter, sch, TR, VA, users, truth_ours_A)
def lr_recs(dfB_labels):
    sc, m = ltr.fit(dfB_labels)
    r = ltr.to_recs(dfA, m.predict_proba(sc.transform(dfA[ltr.FEATS].values))[:, 1], fillA)
    for u in users:
        if u not in hist_users: r[u] = [int(x) for x in fill[:K]]
    return r
recs = {"Популярность": rank5(d, comp["pop"], fill), "История": rank5(d, comp["hist"], fill),
        "CBF": rank5(d, comp["cbf"], fill), "PPR": rank5(d, comp["ppr"], fill),
        "Слияние LR (текущий финал)": lr_recs(dfB)}

# ---------- step 1 ----------
print("STEP 1: fixed lists, answer key varies; MAP@5 over all 1260 test users")
rows = []
for k in KEYS:
    t = key_fn(k)(inter[inter.week.isin(VA)])
    sz = [len(t[u]) for u in users if u in t]
    r = {"ключ ответов": label(k), "польз. с ответами": len(sz), "медиана |R|": int(np.median(sz))}
    for nm, rr in recs.items():
        r[nm] = round(mapk(t, rr, users), 4)
    r["|поп − 0,281|"] = round(abs(r["Популярность"] - KAGGLE_POP), 4)
    rows.append((k, r))
df1 = pd.DataFrame([r for _, r in rows])
pd.set_option("display.width", 250)
print(df1.to_string(index=False))
df1.to_csv("out/keys_step1.csv", index=False)

# ---------- step 2 ----------
close = [k for k, r in rows if r["|поп − 0,281|"] <= 0.03]
print("\nSTEP 2 keys with |pop - 0.281| <= 0.03:", [label(k) for k in close])
grid = list(itertools.product([0, 0.05, 0.1, 0.2, 0.4, 0.8], [0, 0.25, 0.5, 1.0, 2.0], [0, 0.05, 0.1, 0.2]))
out = []
for k in close:
    tA = key_fn(k)(inter[inter.week.isin(VA)])
    tB = key_fn(k)(inter[inter.week.isin(VAb)])
    res = {"ключ ответов": label(k)}
    for nm, rr in recs.items():
        res[nm] = round(mapk(tA, rr, users), 4)
    best, bw = -1, None
    for b, c, e in grid:
        m_ = mapk(tA, rank5(d, N["hist"] + b * N["pop"] + c * N["ppr"] + e * N["cbf"], fill), users)
        if m_ > best: best, bw = m_, (b, c, e)
    res["лин. гибрид (веса на A)"] = round(best, 4); res["веса b,c,e"] = str(bw)
    dfB2 = dfB.copy()
    dfB2["y"] = [float(i in tB.get(u, ())) for u, i in zip(dfB2.user_id, dfB2.tv_show_id)]
    res["LR, переобучена на B под ключ"] = round(mapk(tA, lr_recs(dfB2), users), 4)
    out.append(res); print(res, flush=True)
df2 = pd.DataFrame(out)
print(df2.to_string(index=False))
df2.to_csv("out/keys_step2.csv", index=False)
