"""Step 6: learned score fusion (candidate generation + logistic-regression re-ranking).
Candidates per user = union of Top-30 history, Top-30 PPR, Top-20 popularity (all restricted to programmes airing
in the target window). Features combine the individual methods' scores. The LR is fitted on fold B
(train 11-18 -> labels 19-24) and evaluated on fold A (train 11-24 -> val 25-30): selection never sees fold A labels.
"""
import sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
import lib
from lib import *
from methods import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATS = ["wk_ratio", "wk_last4", "log_air", "log_hours", "since_last", "never", "log_pop", "log_pop_last2",
         "ppr", "cbf", "ch_aff", "log_air_target", "hist_n", "ppr_rank", "hist_rank"]


def build(inter, sch, TR, TW, users, truth=None, alpha=0.7):
    t0 = time.time()
    tr = inter[inter.week.isin(TR)]
    cand = candidates(sch, TW)
    d = Data(tr, cand, users, weight="weeks")
    F, _ = item_features(sch, d.items, use_channel=True)
    H = score_hist(d); P = score_ppr(d, alpha=alpha); C = score_cbf(d, F); POP = d.pop[d.cand_col].astype(float)
    w = watched(tr)
    last = max(TR)
    ww = w[w.user_id.isin(set(users)) & w.tv_show_id.isin(cand)]
    g = ww.groupby(["user_id", "tv_show_id"]).agg(air=("ov", "size"), sec=("ov", "sum"), lastw=("week", "max"))
    g4 = ww[ww.week > last - 4].drop_duplicates(["user_id", "tv_show_id", "week"]).groupby(["user_id", "tv_show_id"]).size()
    act = w.groupby("user_id").week.nunique()
    wl = watched(tr)
    pop2 = wl[wl.week > last - 2].groupby("tv_show_id").user_id.nunique()
    # channel affinity: share of the user's watched (show,week) pairs on the show's main channel in target window
    main_ch = sch[sch.week.isin(TW)].groupby("tv_show_id").channel_id.agg(lambda x: x.value_counts().index[0])
    uw = w[w.user_id.isin(set(users))].drop_duplicates(["user_id", "tv_show_id", "week"])
    uch = uw.groupby(["user_id", "channel_id"]).size()
    utot = uw.groupby("user_id").size()
    air_t = sch[sch.week.isin(TW)].groupby("tv_show_id").size()
    rows = []
    K_H, K_P, K_POP = 30, 30, 20
    pop_top = np.argsort(-POP, kind="stable")[:K_POP]
    for r, u in enumerate(d.users):
        hs = np.argsort(-H[r], kind="stable")[:K_H]; hs = hs[H[r][hs] > 0]
        ps = np.argsort(-P[r], kind="stable")[:K_P]; ps = ps[P[r][ps] > 0]
        cidx = np.unique(np.concatenate([hs, ps, pop_top]))
        hmax = H[r].max() or 1; pmax = P[r].max() or 1; cmax = C[r].max() or 1
        prank = np.empty(len(d.cand)); prank[np.argsort(-P[r], kind="stable")] = np.arange(len(d.cand))
        hrank = np.empty(len(d.cand)); hrank[np.argsort(-H[r], kind="stable")] = np.arange(len(d.cand))
        a_u = act.get(u, 0)
        for j in cidx:
            it = int(d.cand[j])
            key = (u, it)
            gi = g.loc[key] if key in g.index else None
            mc = main_ch.get(it, -1)
            rows.append(dict(
                user_id=u, tv_show_id=it,
                wk_ratio=H[r, j] / a_u if a_u else 0.0, wk_last4=g4.get(key, 0) / 4,
                log_air=np.log1p(gi.air) if gi is not None else 0.0,
                log_hours=np.log1p(gi.sec / 3600) if gi is not None else 0.0,
                since_last=(last - gi.lastw) / len(TR) if gi is not None else 1.0, never=float(gi is None),
                log_pop=np.log1p(POP[j]), log_pop_last2=np.log1p(pop2.get(it, 0)),
                ppr=P[r, j] / pmax, cbf=C[r, j] / cmax,
                ch_aff=uch.get((u, mc), 0) / utot.get(u, 1) if u in utot.index else 0.0,
                log_air_target=np.log1p(air_t.get(it, 0)), hist_n=H[r, j] / hmax,
                ppr_rank=np.log1p(prank[j]), hist_rank=np.log1p(hrank[j]),
                y=float(it in truth.get(u, set())) if truth is not None else np.nan))
    df = pd.DataFrame(rows)
    fill = list(d.cand[np.argsort(-POP, kind="stable")][:50])
    print(f"  built {len(df)} candidate rows for {len(users)} users in {time.time()-t0:.0f} s", flush=True)
    return df, fill, d


def to_recs(df, score, fill):
    df = df.assign(s=score).sort_values(["user_id", "s", "tv_show_id"], ascending=[True, False, True])
    recs = {u: g.tv_show_id.head(K).tolist() for u, g in df.groupby("user_id", sort=False)}
    for u in recs:
        for f in fill:
            if len(recs[u]) >= K: break
            if f not in recs[u]: recs[u].append(int(f))
    return recs


def fit(df):
    sc = StandardScaler().fit(df[FEATS].values)
    m = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(sc.transform(df[FEATS].values), df.y.values)
    return sc, m


if __name__ == "__main__":
    inter, sch, test_users = load()
    out = {}
    folds = {"B": (list(range(11, 19)), list(range(19, 25))), "A": (list(range(11, 25)), list(range(25, 31)))}
    data = {}
    for f, (TR, VA) in folds.items():
        _, truth = split(inter, TR, VA)
        EU = [u for u in test_users if u in truth]
        df, fill, d = build(inter, sch, TR, VA, EU, truth)
        rec_cov = np.mean([min(len(set(df[df.user_id == u].tv_show_id) & truth[u]), K) / min(K, len(truth[u])) for u in EU[:300]])
        print(f"fold {f} (rule {lib.RULE}): candidate rows/user {len(df)/len(EU):.1f}, positives {df.y.mean():.3f}, "
              f"share of users' first min(5,|R|) slots fillable from candidates (first 300 users) {rec_cov:.3f}")
        data[f] = (df, fill, truth, EU)
    sc, m = fit(data["B"][0])
    print("LR coefficients (standardised features):")
    for n_, c_ in sorted(zip(FEATS, m.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {n_:15s} {c_:+.3f}")
    for f in ["B", "A"]:
        df, fill, truth, EU = data[f]
        s = m.predict_proba(sc.transform(df[FEATS].values))[:, 1]
        mm = mapk(truth, to_recs(df, s, fill), EU)
        base_h = mapk(truth, to_recs(df, df.hist_n.values + 1e-6 * df.log_pop.values, fill), EU)
        base_p = mapk(truth, to_recs(df, df.ppr.values, fill), EU)
        print(f"fold {f} ({'fit' if f=='B' else 'held-out'}): LR fusion MAP@5={mm:.4f} | inside same candidates: hist={base_h:.4f}, ppr={base_p:.4f}")
        out[f] = mm
    # reverse direction as extra check: fit on A, evaluate on B
    sc2, m2 = fit(data["A"][0])
    df, fill, truth, EU = data["B"]
    mm = mapk(truth, to_recs(df, m2.predict_proba(sc2.transform(df[FEATS].values))[:, 1], fill), EU)
    print(f"reverse check: fit on A, evaluate on B: MAP@5={mm:.4f}")
    for f in data:
        data[f][0].to_pickle(f"out/ltr_feats_{f}.pkl")
    json.dump(out, open("out/ltr.json", "w"))
