"""Compare the old 'watched' rule (overlap >= 300 s) with the new one (>= 300 s OR >= 90% of the airing).
The rule is used in two places: (1) building the models (training interactions + LR fusion labels on fold B),
(2) building the validation truth on fold A. Changing (2) changes the answer key itself, so MAP@5 is reported for
all 4 combinations: model rule x truth rule. Same evaluation users (test users with >=1 relevant show under the OLD
rule on fold A), same candidates, same fallback for every cell."""
import sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
import lib
from lib import *
from methods import *
import importlib; ltr = importlib.import_module("50_ltr")

inter, sch, test_users = load()
TRb, VAb = list(range(11, 19)), list(range(19, 25))
TR, VA = list(range(11, 25)), list(range(25, 31))

# how much the new rule changes the interaction table
old = inter.ov >= 300; new = old | (inter.frac >= 0.9)
print(f"pairs watched: old {old.sum()}, new {new.sum()} (+{(new & ~old).sum()} pairs, "
      f"all with ov < 300 s and frac >= 0.9); airings < 300 s that can now count: "
      f"{(inter[new & ~old].dur < 300).mean():.3f} of the added pairs are airings shorter than 300 s")
added = inter[new & ~old]
print("added pairs by category:", sch.drop_duplicates("tv_show_id").set_index("tv_show_id")
      .tv_show_category.reindex(added.tv_show_id).value_counts().to_dict())

truthA = {r: split(inter, TR, VA, rule=r)[1] for r in ["old", "new"]}
EU = [u for u in test_users if u in truthA["old"]]
print("evaluation users", len(EU), "; of them with empty truth under new rule:", sum(u not in truthA["new"] for u in EU))
for r in ["old", "new"]:
    print(f"truth rule {r}: median relevant shows/user {np.median([len(truthA[r].get(u, ())) for u in EU]):.0f}")

cand = candidates(sch, VA)
table = {}
for mrule in ["old", "new"]:
    t0 = time.time()
    lib.RULE = mrule
    tr = inter[inter.week.isin(TR)]
    d = Data(tr, cand, EU, weight="weeks")
    fill = list(d.cand[np.argsort(-d.pop[d.cand_col], kind="stable")][:50])
    F, _ = item_features(sch, d.items, use_channel=True)
    recs = {"Популярность": rank5(d, score_pop(d), fill), "История": rank5(d, score_hist(d), fill),
            "CBF": rank5(d, score_cbf(d, F), fill), "PPR α=0.7": rank5(d, score_ppr(d, alpha=0.7), fill)}
    # LR fusion: fit on fold B with labels under the model rule, apply to fold A features
    _, truthB = split(inter, TRb, VAb)
    EUB = [u for u in test_users if u in truthB]
    dfB, _, _ = ltr.build(inter, sch, TRb, VAb, EUB, truthB)
    dfA, fillA, _ = ltr.build(inter, sch, TR, VA, EU, truthA[mrule])
    sc, m = ltr.fit(dfB)
    recs["Слияние (LR)"] = ltr.to_recs(dfA, m.predict_proba(sc.transform(dfA[ltr.FEATS].values))[:, 1], fillA)
    for trule in ["old", "new"]:
        for meth, r in recs.items():
            table.setdefault(meth, {})[f"model={mrule} truth={trule}"] = round(mapk(truthA[trule], r, EU), 4)
    print(f"model rule {mrule}: done in {time.time()-t0:.0f} s", flush=True)
lib.RULE = "old"

df = pd.DataFrame(table).T
cols = ["model=old truth=old", "model=new truth=old", "model=old truth=new", "model=new truth=new"]
df = df[cols]
df["Δ (truth old)"] = df[cols[1]] - df[cols[0]]
df["Δ (truth new)"] = df[cols[3]] - df[cols[2]]
pd.set_option("display.width", 200)
print(df.round(4).to_string())
json.dump(df.round(4).to_dict(), open("out/rule_compare.json", "w"), ensure_ascii=False, indent=1)
