"""Step 8: final model for Kaggle. LR fusion fitted on folds A and B (labels from weeks 19-24 and 25-30 only),
applied to features from train weeks 11-30; candidates = labelled programmes airing in weeks 31-42 (schedule only;
no test-period viewing exists in the data or is used). Also writes popularity and PPR submissions as references."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
from lib import *
from methods import *
import importlib; ltr = importlib.import_module("50_ltr")

t0 = time.time()
inter, sch, test_users = load()
TR, TW = list(range(11, 31)), list(range(31, 43))
users = list(test_users)
hist_users = set(watched(inter[inter.week.isin(TR)]).user_id)
print("test users", len(users), "with watched history in weeks 11-30:", len(hist_users & set(users)))
df, fill, d = ltr.build(inter, sch, TR, TW, users, truth=None)
train = pd.concat([pd.read_pickle("out/ltr_feats_A.pkl"), pd.read_pickle("out/ltr_feats_B.pkl")])
sc, m = ltr.fit(train)
print("fitted on", len(train), "rows; coefficients:", dict(zip(ltr.FEATS, np.round(m.coef_[0], 3))))
recs = ltr.to_recs(df, m.predict_proba(sc.transform(df[ltr.FEATS].values))[:, 1], fill)
cold = [u for u in users if u not in hist_users]
for u in cold:
    recs[u] = [int(x) for x in fill[:K]]  # same popularity fallback as every method

def write(recs, path):
    rows = [(u, " ".join(str(int(x)) for x in recs[u])) for u in users]
    pd.DataFrame(rows, columns=["user_id", "tv_show_id"]).to_csv(path, index=False)
    chk = pd.read_csv(path, dtype={"user_id": "uint64"})
    lens = chk.tv_show_id.str.split().map(len)
    ok_ids = chk.tv_show_id.str.split().map(lambda l: all(int(x) in set(d.cand.tolist()) for x in l))
    dup = chk.tv_show_id.str.split().map(lambda l: len(set(l)) != len(l))
    print(f"{path}: rows {len(chk)}, all 5 ids {bool((lens == 5).all())}, all ids in week 31-42 pool {bool(ok_ids.all())}, "
          f"rows with duplicate ids {int(dup.sum())}, user order == template {bool((chk.user_id.values == np.array(users)).all())}")

write(recs, "out/submission_lr_fusion.csv")
write(rank5(d, score_pop(d), fill), "out/submission_popularity.csv")
write(rank5(d, score_ppr(d, alpha=0.7), fill), "out/submission_ppr.csv")
top = pd.Series([x for r in recs.values() for x in r]).value_counts()
names = sch.drop_duplicates("tv_show_id").set_index("tv_show_id").tv_show_title
print("distinct programmes recommended:", len(top), "; most frequent:", [(i, names[i][:25], c) for i, c in top.head(5).items()])
print("time", round(time.time() - t0), "s")
