"""Step 4: evaluate every method individually on the same users / candidates / fallback; small grids.
Selection is by validation MAP@5 only (train weeks 11-24, validation weeks 25-30)."""
import sys, time, json
import numpy as np
sys.path.insert(0, "/home/claude/work")
from lib import *
from methods import *

inter, sch, test_users = load()
TR = list(range(11, 25)); VA = list(range(25, 31))
tr, truth = split(inter, TR, VA)
cand = candidates(sch, VA)
EU = [u for u in test_users if u in truth]
res = {}

def ev(name, d, S, fill):
    t = time.time()
    r = rank5(d, S, fill)
    m = mapk(truth, r, EU)
    res[name] = round(m, 4)
    print(f"{name:45s} MAP@5={m:.4f}", flush=True)
    return r

def mk(**kw):
    d = Data(tr, cand, EU, **kw)
    fill = list(d.cand[np.argsort(-d.pop[d.cand_col], kind="stable")][:50])
    return d, fill

t0 = time.time()
d, fill = mk(weight="weeks")
ev("popularity", d, score_pop(d), fill)
for wt in ["seconds", "airings", "weeks"]:
    for hl in [None, 8, 4]:
        dd, ff = mk(weight=wt, half_life=hl)
        ev(f"history w={wt} hl={hl}", dd, score_hist(dd), ff)
print("history grid", round(time.time() - t0, 1), "s")

# CBF
t0 = time.time()
for ch in [False, True]:
    F, vocab = item_features(sch, d.items, use_channel=ch)
    print(f"CBF features: {F.shape[1]} tokens (channel={ch})")
    ev(f"CBF channel={ch}", d, score_cbf(d, F), fill)
print("CBF", round(time.time() - t0, 1), "s")

# PPR
for a in [0.15, 0.3, 0.5, 0.7, 0.85]:
    t0 = time.time()
    S = score_ppr(d, alpha=a)
    ev(f"PPR alpha={a}", d, S, fill)
    print(f"  iters {d.ppr_iters}, {time.time() - t0:.1f} s")
json.dump(res, open("out/val_stage1.json", "w"), indent=1, ensure_ascii=False)
