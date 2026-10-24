"""Paired standard error of the difference (LR re-fitted under the key) - (current final) over the 1260 test users,
plus paired bootstrap 95% interval (2000 resamples, seed 42), for the 3 calibrated keys of 90_keys.py."""
import sys, runpy
import numpy as np
sys.argv = ["90_keys.py"]
sys.path.insert(0, "/home/claude/work")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path("/home/claude/work/90_keys.py")
apk, users, inter, VA, VAb, key_fn, dfB, lr_recs, recs = (g[k] for k in ["apk", "users", "inter", "VA", "VAb", "key_fn", "dfB", "lr_recs", "recs"])
rng = np.random.default_rng(42)
for k in [("show_sec", 7200), ("show_days", 8), ("topk_days", 5)]:
    tA = key_fn(k)(inter[inter.week.isin(VA)]); tB = key_fn(k)(inter[inter.week.isin(VAb)])
    dfB2 = dfB.copy(); dfB2["y"] = [float(i in tB.get(u, ())) for u, i in zip(dfB2.user_id, dfB2.tv_show_id)]
    r_new = lr_recs(dfB2); r_cur = recs["Слияние LR (текущий финал)"]
    diff = np.array([apk(tA.get(u, set()), r_new[u]) - apk(tA.get(u, set()), r_cur[u]) for u in users])
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    print(f"{k}: mean diff {diff.mean():+.4f}, paired SE {diff.std(ddof=1)/np.sqrt(len(diff)):.4f}, "
          f"bootstrap 95% [{np.quantile(bs,.025):+.4f}, {np.quantile(bs,.975):+.4f}], users with changed AP {np.mean(diff!=0):.3f}")
