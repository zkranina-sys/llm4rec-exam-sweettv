"""Final model v2 (M2, passed the cross-key rule in 92_crosskey.log): LR fusion with the 15 original features +
2 daily-regularity features, original labels (rule 'new'), fitted on folds A and B; applied to features from
train weeks 11-30, candidates airing in weeks 31-42. Same format checks as 70_final.py."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
import lib
from lib import *
from methods import *
import importlib; ltr = importlib.import_module("50_ltr")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

t0 = time.time()
print("watched rule:", lib.RULE)
inter, sch, test_users = load()
inter = inter.assign(day=inter.st // 86400)
users = list(test_users)
FOLD = {"B": list(range(11, 19)), "A": list(range(11, 25))}
TR, TW = list(range(11, 31)), list(range(31, 43))

def day_feats(df, TRw):  # identical to 92_crosskey.py
    w = watched(inter[inter.week.isin(TRw)])
    last_day = w.day.max()
    a = w.groupby(["user_id", "tv_show_id"]).day.nunique().rename("days_all")
    b = w[w.day > last_day - 14].groupby(["user_id", "tv_show_id"]).day.nunique().rename("days_last14")
    out = df.merge(a, left_on=["user_id", "tv_show_id"], right_index=True, how="left") \
            .merge(b, left_on=["user_id", "tv_show_id"], right_index=True, how="left")
    out["log_days_all"] = np.log1p(out.days_all.fillna(0)) / np.log1p(7 * len(TRw))
    out["days_last14"] = out.days_last14.fillna(0) / 14
    return out
F2 = ltr.FEATS + ["log_days_all", "days_last14"]
train = pd.concat([day_feats(pd.read_pickle(f"out/ltr_feats_{f}.pkl"), FOLD[f]) for f in ["A", "B"]])
sc = StandardScaler().fit(train[F2].values)
m = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(sc.transform(train[F2].values), train.y.values)
print("fitted on", len(train), "rows; new-feature coefficients:",
      {k: round(c, 3) for k, c in zip(F2, m.coef_[0]) if k in ("log_days_all", "days_last14")})
df, fill, d = ltr.build(inter.drop(columns="day"), sch, TR, TW, users, truth=None)
df = day_feats(df, TR)
recs = ltr.to_recs(df, m.predict_proba(sc.transform(df[F2].values))[:, 1], fill)
hist_users = set(watched(inter[inter.week.isin(TR)]).user_id)
for u in users:
    if u not in hist_users: recs[u] = [int(x) for x in fill[:K]]
path = "out/submission_lr_fusion_v2.csv"
pd.DataFrame([(u, " ".join(str(int(x)) for x in recs[u])) for u in users], columns=["user_id", "tv_show_id"]).to_csv(path, index=False)
chk = pd.read_csv(path, dtype={"user_id": "uint64"})
pool = set(d.cand.tolist())
lists = chk.tv_show_id.str.split()
print(f"{path}: rows {len(chk)}, all 5 ids {bool((lists.map(len) == 5).all())}, all in week 31-42 pool "
      f"{bool(lists.map(lambda l: all(int(x) in pool for x in l)).all())}, duplicate ids {int(lists.map(lambda l: len(set(l)) != 5).sum())}, "
      f"user order == template {bool((chk.user_id.values == np.array(users)).all())}")
old = pd.read_csv("out/submission_lr_fusion.csv")
print("users whose Top-5 differs from current final:", int((old.tv_show_id != chk.tv_show_id).sum()), "of", len(chk))
print("time", round(time.time() - t0), "s")
