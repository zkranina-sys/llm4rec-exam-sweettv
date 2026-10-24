"""Step 5: score-fusion hybrid. Components (each min-max normalised per user over candidates):
history (distinct weeks watched), popularity, PPR (alpha), CBF (with channel token).
S = hist + b*pop + c*ppr + e*cbf. Weights chosen on fold A (train 11-24 / val 25-30);
fold B (train 11-18 / val 19-24) is a robustness check run with the same grid."""
import sys, time, json, itertools
import numpy as np
sys.path.insert(0, "/home/claude/work")
from lib import *
from methods import *

inter, sch, test_users = load()
FOLDS = {"A": (list(range(11, 25)), list(range(25, 31))), "B": (list(range(11, 19)), list(range(19, 25)))}
grid_b = [0, 0.05, 0.1, 0.2, 0.4]
grid_c = [0, 0.25, 0.5, 1.0]
grid_e = [0, 0.05, 0.1, 0.2]
out = {}
for fname, (TR, VA) in FOLDS.items():
    t0 = time.time()
    tr, truth = split(inter, TR, VA)
    cand = candidates(sch, VA)
    EU = [u for u in test_users if u in truth]
    d = Data(tr, cand, EU, weight="weeks")
    fill = list(d.cand[np.argsort(-d.pop[d.cand_col], kind="stable")][:50])
    F, _ = item_features(sch, d.items, use_channel=True)
    comp = {"hist": score_hist(d), "pop": score_pop(d).astype(float), "ppr": score_ppr(d, alpha=0.7),
            "cbf": score_cbf(d, F)}
    N = {k: minmax_rows(v) for k, v in comp.items()}
    print(f"fold {fname}: train {TR[0]}-{TR[-1]} val {VA[0]}-{VA[-1]} users {len(EU)} cand {len(cand)} "
          f"(components {time.time()-t0:.0f} s)", flush=True)
    single = {k: mapk(truth, rank5(d, comp[k], fill), EU) for k in comp}
    print("  single:", {k: round(v, 4) for k, v in single.items()})
    res = {}
    for b, c, e in itertools.product(grid_b, grid_c, grid_e):
        S = N["hist"] + b * N["pop"] + c * N["ppr"] + e * N["cbf"]
        res[(b, c, e)] = mapk(truth, rank5(d, S, fill), EU)
    best = sorted(res.items(), key=lambda t: -t[1])[:8]
    for (b, c, e), m in best:
        print(f"  b={b} c={c} e={e}: {m:.4f}")
    out[fname] = {"single": single, "grid": {f"{k}": v for k, v in res.items()}}
    print(f"  fold time {time.time()-t0:.0f} s", flush=True)
json.dump(out, open("out/hybrid_grid.json", "w"), indent=1)
