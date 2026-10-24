"""Cross-key robustness test (decision rule set by Zlata): a challenger replaces the current final only if on fold A it is
not worse on ANY of the 5 answer keys and better on at least 2 (point estimates; paired bootstrap 95% CI shown).
Challengers:
  M1 = LR fusion with the same features, labels on fold B re-defined by key 'show total >= 7200 s'
  M2 = LR fusion with our original labels + 2 daily-regularity features (distinct days watched in train window,
       distinct days in its last 2 weeks) - no Kaggle-derived choice involved
All evaluated over all 1260 test users, cold users -> popularity fallback (as in the final)."""
import sys, runpy, io, contextlib
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path("/home/claude/work/90_keys.py")  # rebuilds fixed fold-A lists, dfA, dfB (same seeds/data)
apk, mapk, users, inter, key_fn, dfA, dfB, fillA, fill, hist_users, recs, ltr, K, watched = (g[k] for k in
    ["apk", "mapk", "users", "inter", "key_fn", "dfA", "dfB", "fillA", "fill", "hist_users", "recs", "ltr", "K", "watched"])
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
TRb, VAb = list(range(11, 19)), list(range(19, 25))
TR, VA = list(range(11, 25)), list(range(25, 31))

def fit_predict(train_df, test_df, feats):
    sc = StandardScaler().fit(train_df[feats].values)
    m = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(sc.transform(train_df[feats].values), train_df.y.values)
    r = ltr.to_recs(test_df, m.predict_proba(sc.transform(test_df[feats].values))[:, 1], fillA)
    for u in users:
        if u not in hist_users: r[u] = [int(x) for x in fill[:K]]
    return r

# sanity: refitting the current final here must reproduce it exactly
r_cur = recs["Слияние LR (текущий финал)"]
r_chk = fit_predict(dfB, dfA, ltr.FEATS)
print("sanity: re-fit reproduces current final lists for", sum(r_chk[u] == r_cur[u] for u in users), "of", len(users), "users")

# M1
tB7200 = key_fn(("show_sec", 7200))(inter[inter.week.isin(VAb)])
dfB1 = dfB.copy(); dfB1["y"] = [float(i in tB7200.get(u, ())) for u, i in zip(dfB1.user_id, dfB1.tv_show_id)]
r_m1 = fit_predict(dfB1, dfA, ltr.FEATS)

# M2: daily-regularity features from the TRAIN window of each fold only
def day_feats(df, TRw):
    w = watched(inter[inter.week.isin(TRw)])
    last_day = w.day.max()
    a = w.groupby(["user_id", "tv_show_id"]).day.nunique().rename("days_all")
    b = w[w.day > last_day - 14].groupby(["user_id", "tv_show_id"]).day.nunique().rename("days_last14")
    out = df.merge(a, left_on=["user_id", "tv_show_id"], right_index=True, how="left") \
            .merge(b, left_on=["user_id", "tv_show_id"], right_index=True, how="left")
    out["log_days_all"] = np.log1p(out.days_all.fillna(0)) / np.log1p(7 * len(TRw))  # scaled by window length
    out["days_last14"] = out.days_last14.fillna(0) / 14
    return out
dfB2, dfA2 = day_feats(dfB, TRb), day_feats(dfA, TR)
F2 = ltr.FEATS + ["log_days_all", "days_last14"]
r_m2 = fit_predict(dfB2, dfA2, F2)

KEYS = [("ours", None), ("airing_sec", 1800), ("show_sec", 7200), ("show_days", 8), ("topk_days", 5)]
names = {("ours", None): "наш исходный", ("airing_sec", 1800): "показ ≥ 1800 с", ("show_sec", 7200): "передача ≥ 7200 с",
         ("show_days", 8): "≥ 8 разных дней", ("topk_days", 5): "топ-5 по дням"}
rng = np.random.default_rng(42)
rows, verdict = [], {}
for k in KEYS:
    t = key_fn(k)(inter[inter.week.isin(VA)])
    base = np.array([apk(t.get(u, set()), r_cur[u]) for u in users])
    row = {"ключ": names[k], "текущий финал": round(base.mean(), 4)}
    for nm, r in [("M1", r_m1), ("M2", r_m2)]:
        x = np.array([apk(t.get(u, set()), r[u]) for u in users]); dlt = x - base
        idx = rng.integers(0, len(dlt), (2000, len(dlt)))
        lo, hi = np.quantile(dlt[idx].mean(1), [.025, .975])
        row[nm] = round(x.mean(), 4); row[f"{nm}−финал"] = f"{dlt.mean():+.4f} [{lo:+.4f}; {hi:+.4f}]"
        verdict.setdefault(nm, []).append(dlt.mean())
    rows.append(row)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
for nm, ds in verdict.items():
    ds = np.array(ds)
    ok = (ds >= 0).all() and (ds > 0).sum() >= 2
    print(f"{nm}: worse on {(ds < 0).sum()} keys, better on {(ds > 0).sum()} keys -> rule {'PASSED' if ok else 'NOT passed'}")
