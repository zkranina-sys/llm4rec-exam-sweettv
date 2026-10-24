"""Diagnose why history < popularity: weighting by seconds vs airings vs weeks; effect of target-window airing count."""
import sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "/home/claude/work")
from lib import *
from methods import *

inter, sch, test_users = load()
TR = list(range(11, 25)); VA = list(range(25, 31))
tr, truth = split(inter, TR, VA)
cand = candidates(sch, VA)
EU = [u for u in test_users if u in truth]
d = Data(tr, cand, EU, half_life=None)
fill = list(d.cand[np.argsort(-d.pop[d.cand_col], kind="stable")][:50])
air = sch[sch.week.isin(VA)].groupby("tv_show_id").size()          # airings in target window (schedule only)
A = air.reindex(d.cand).fillna(0).values
w = watched(tr)
def mat(col):
    g = w.groupby(["user_id", "tv_show_id"])[col].agg(col_agg[col]).reset_index() if False else None
uidx = {u: i for i, u in enumerate(EU)}
ww = w[w.user_id.isin(uidx) & w.tv_show_id.isin(set(d.cand))]
cix = {c: i for i, c in enumerate(d.cand)}
def M(series):
    s = series.reset_index()
    r = s.user_id.map(uidx).values; c = s.tv_show_id.map(cix).values
    return sp.csr_matrix((s.iloc[:, -1].values.astype(float), (r, c)), shape=(len(EU), len(d.cand))).toarray()
S_sec = M(ww.groupby(["user_id", "tv_show_id"]).ov.sum())
S_air = M(ww.groupby(["user_id", "tv_show_id"]).size())
S_wk = M(ww.groupby(["user_id", "tv_show_id"]).week.nunique())
# share of user's train airings of the show: how loyal (watched airings / aired airings in train weeks)
air_tr = sch[sch.week.isin(TR)].groupby("tv_show_id").size().reindex(d.cand).fillna(1).values
def ev(name, S):
    print(f"{name:45s} MAP@5={mapk(truth, rank5(d, S, fill), EU):.4f}")
ev("pop", score_pop(d))
ev("pop x airings_val", score_pop(d) * A)
ev("pop/airings_train x airings_val", score_pop(d) / air_tr * A)
for nm, S in [("seconds", S_sec), ("airings watched", S_air), ("weeks watched", S_wk)]:
    ev(f"hist {nm}", S)
    ev(f"hist {nm} x airings_val", S * A)
    ev(f"hist {nm} / airings_train x airings_val", S / air_tr * A)
# loyalty-rate model: rate = watched airings / aired airings in train; expected views in val = rate * airings_val
rate = S_air / air_tr
ev("rate x airings_val", rate * A)
ev("1-(1-rate)^airings_val", 1 - (1 - np.clip(rate, 0, 1)) ** A)
