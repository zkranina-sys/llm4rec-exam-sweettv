"""Step 3: temporal split sanity + popularity baseline + personal-history method; hand-check of AP@5."""
import time
import numpy as np, pandas as pd
from lib import *

inter, sch, test_users = load()
TR = list(range(11, 25)); VA = list(range(25, 31))
tr, truth = split(inter, TR, VA)
cand = candidates(sch, VA)
users = list(test_users)
n_with = sum(1 for u in users if u in truth)
n_hist = len(set(watched(tr).user_id) & set(users))
print(f"split train weeks {TR[0]}-{TR[-1]} / val weeks {VA[0]}-{VA[-1]}")
print(f"test users {len(users)}; with >=1 relevant show in val {n_with}; with watched history in train {n_hist}")
print(f"candidate pool (labelled shows airing in val weeks): {len(cand)}")
rel_sizes = [len(truth[u]) for u in users if u in truth]
print("relevant set size per user: median", np.median(rel_sizes), "mean", round(np.mean(rel_sizes), 2))
tru_in_cand = np.mean([len(truth[u] & cand) / len(truth[u]) for u in users if u in truth])
print("share of relevant shows inside candidate pool:", round(tru_in_cand, 4))
seen_tr = watched(tr).groupby("user_id").tv_show_id.agg(set).to_dict()
rep = [len(truth[u] & seen_tr.get(u, set())) / len(truth[u]) for u in users if u in truth]
print("share of a user's val shows already watched by them in train (repeat rate):", round(np.mean(rep), 4))

# evaluate only on test users that have >=1 relevant show (users with empty truth add 0 for every method)
EU = [u for u in users if u in truth]
print("evaluation users:", len(EU))
res = {}
for wb in [None, 4, 2, 1]:
    pop = popularity(tr, cand, wb)
    top = list(pop.index[:K])
    res[f"pop(last {wb or 'all'} wks)"] = mapk(truth, {u: top for u in EU}, EU)
    print(f"popularity weeks_back={wb}: MAP@5={res[f'pop(last {wb or 'all'} wks)']:.4f} top5={top}")
