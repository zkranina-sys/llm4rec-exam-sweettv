"""Six candidate users for the recommendation examples (fold A: train weeks 11-24, validation 25-30, rule 'new').
Selection rule (deterministic): among test users with >=1 relevant show in validation, rank by number of watched
labelled airings in train; heavy = users at the 99th, 95th and 90th percentile of that count (heavy but not the
single extreme TV-left-on account); light = the one test user with NO watched history in train who still has
validation truth (if several, the one with most validation shows), plus the two users with the smallest non-zero
history. Prints per user: train airings, distinct shows, active weeks, hours, top categories, |R_u|, AP@5 per method.
Also writes, for each candidate, the final method's Top-5 and the full list of validation shows (for a hand AP@5)."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
import lib
from lib import *
from methods import *
import importlib; ltr = importlib.import_module("50_ltr")

print("watched rule:", lib.RULE)
inter, sch, test_users = load()
TRb, VAb = list(range(11, 19)), list(range(19, 25))
TR, VA = list(range(11, 25)), list(range(25, 31))
tr, truth = split(inter, TR, VA)
cand = candidates(sch, VA)
users = list(test_users)
meta = sch.drop_duplicates("tv_show_id").set_index("tv_show_id")
w = watched(tr)
st = w.groupby("user_id").agg(airings=("tv_show_id", "size"), shows=("tv_show_id", "nunique"),
                              weeks=("week", "nunique"), hours=("ov", lambda x: x.sum() / 3600))
st = st.reindex(users).fillna(0)
st["n_val"] = [len(truth.get(u, ())) for u in users]
el = st[st.n_val > 0]
print("test users with validation truth:", len(el), "; of them without watched train history:", int((el.airings == 0).sum()))
print("train watched airings per eligible user: quantiles 10/50/90/95/99% =",
      np.quantile(el.airings, [.1, .5, .9, .95, .99]).round(0))
hist_el = el[el.airings > 0].sort_values("airings", kind="stable")
heavy = []
for q in [0.99, 0.95, 0.90]:
    target = np.quantile(hist_el.airings, q)
    u = (hist_el.airings - target).abs().idxmin(); heavy.append(u)
cold = el[el.airings == 0].sort_values("n_val", ascending=False, kind="stable")
light = list(cold.index[:1]) + [u for u in hist_el.index if u not in heavy][:2]
picked = [("много", u) for u in heavy] + [("мало/нет", u) for u in light]

# recommendations of all methods for these users (same pipeline as validation)
d = Data(tr, cand, users, weight="weeks")
fill = list(d.cand[np.argsort(-d.pop[d.cand_col], kind="stable")][:50])
F, _ = item_features(sch, d.items, use_channel=True)
recs = {"Популярность": rank5(d, score_pop(d), fill), "История": rank5(d, score_hist(d), fill),
        "CBF": rank5(d, score_cbf(d, F), fill), "PPR": rank5(d, score_ppr(d, alpha=0.7), fill)}
_, truthB = split(inter, TRb, VAb)
dfB, _, _ = ltr.build(inter, sch, TRb, VAb, [u for u in users if u in truthB], truthB)
dfA, fillA, _ = ltr.build(inter, sch, TR, VA, users, truth)
sc, m = ltr.fit(dfB)
recs["Слияние (финал)"] = ltr.to_recs(dfA, m.predict_proba(sc.transform(dfA[ltr.FEATS].values))[:, 1], fillA)
no_hist = set(st.index[st.airings == 0])
for u in no_hist:  # same popularity fallback for cold users in every method (as in 70_final.py)
    recs["Слияние (финал)"][u] = [int(x) for x in fill[:K]]

rows = []
cat_of = meta.tv_show_category
for grp, u in picked:
    wu = w[w.user_id == u]
    cats = cat_of.reindex(wu.tv_show_id).value_counts(normalize=True)
    topc = ", ".join(f"{c} {p:.0%}" for c, p in cats.head(3).items()) if len(cats) else "— (нет истории)"
    r = dict(группа=grp, user_id=str(u), показов_с_id_в_train=int(st.airings[u]), передач=int(st.shows[u]),
             активных_недель=int(st.weeks[u]), часов=round(float(st.hours[u]), 1), любимые_категории=topc,
             передач_на_валидации=int(st.n_val[u]))
    for k, rr in recs.items():
        r[f"AP@5 {k}"] = round(apk(truth.get(u, set()), rr[u]), 3)
    rows.append(r)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(df.to_string(index=False))
df.to_csv("out/candidates6.csv", index=False)

# per-candidate files for the hand calculation: final Top-5 + full validation list (no AP shown here)
with open("out/candidates6_lists.md", "w") as f:
    f.write("# Final method (LR fusion, fold A, rule 'new'): Top-5 and validation shows, weeks 25-30\n\n")
    for grp, u in picked:
        f.write(f"## user {u} ({grp})\n\nTop-5 финального метода:\n\n| ранг | id | название | категория |\n|---|---|---|---|\n")
        for i, it in enumerate(recs["Слияние (финал)"][u], 1):
            f.write(f"| {i} | {it} | {meta.tv_show_title.get(it, '?')} | {cat_of.get(it, '?')} |\n")
        tv = sorted(truth.get(u, set()))
        f.write(f"\nРеально смотрел(а) на валидации: {len(tv)} передач (отсортировано по id)\n\n| id | название | категория |\n|---|---|---|\n")
        for it in tv:
            f.write(f"| {it} | {meta.tv_show_title.get(it, '?')} | {cat_of.get(it, '?')} |\n")
        f.write("\n")
json.dump({k: {str(u): [int(x) for x in v[u]] for _, u in picked} for k, v in recs.items()}, open("out/candidates6_recs.json", "w"), ensure_ascii=False, indent=1)
print("wrote out/candidates6.csv, out/candidates6_lists.md")
