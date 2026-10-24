"""Recommendation methods. Each scorer returns a dense matrix S[users x items] over the candidate items;
lists of 5 are produced by `rank5` identically for all methods (repeats allowed, cold users -> popularity)."""
import numpy as np, pandas as pd, scipy.sparse as sp
from lib import watched, K

DEC = 0.0  # placeholder


class Data:
    """Index users/items of the training window and build the user x item weight matrix."""

    def __init__(self, tr, cand, users, half_life=None, thresh=300, weight="weeks"):
        """weight: 'weeks' = number of distinct weeks the user watched the show (regularity),
        'airings' = number of airings watched, 'seconds' = hours watched. Optional recency decay."""
        self.users = list(users)
        self.uix = {u: i for i, u in enumerate(self.users)}
        self.cand = np.array(sorted(cand))
        w = watched(tr, thresh).copy()
        last = tr.week.max()
        if weight == "weeks":
            w = w.drop_duplicates(["user_id", "tv_show_id", "week"]).assign(base=1.0)
        elif weight == "airings":
            w = w.assign(base=1.0)
        else:
            w = w.assign(base=w.ov / 3600.0)
        # recency weight: halves every `half_life` weeks before the end of the train window
        w["wt"] = w.base * (0.5 ** ((last - w.week) / half_life)) if half_life else w.base
        all_users = np.unique(np.concatenate([w.user_id.unique(), np.array(self.users, dtype=np.uint64)]))
        self.all_users = all_users
        self.aix = {u: i for i, u in enumerate(all_users)}
        items = np.unique(np.concatenate([w.tv_show_id.unique(), self.cand]))
        self.items = items
        self.iix = {it: i for i, it in enumerate(items)}
        g = w.groupby(["user_id", "tv_show_id"]).wt.sum().reset_index()
        r = g.user_id.map(self.aix).values; c = g.tv_show_id.map(self.iix).values
        self.X = sp.csr_matrix((g.wt.values, (r, c)), shape=(len(all_users), len(items)))
        self.Xb = self.X.copy(); self.Xb.data[:] = 1.0  # binary watched
        self.cand_col = np.array([self.iix[i] for i in self.cand])
        self.urows = np.array([self.aix[u] for u in self.users])
        self.pop = np.asarray(self.Xb.sum(0)).ravel()  # distinct users per item (train)

    def user_hist(self):
        return self.X[self.urows]  # sparse [eval users x items]


def score_pop(d):
    return np.tile(d.pop[d.cand_col], (len(d.users), 1))


def score_hist(d):
    return d.user_hist()[:, d.cand_col].toarray()


def item_features(sch, items, use_channel=False):
    """Bag of content attributes per programme: category, 3 genres, director names, actor names,
    decade of production (and optionally the channel). TF-IDF weighted, L2-normalised rows."""
    s = sch[sch.tv_show_id.isin(items)]
    first = s.drop_duplicates("tv_show_id").set_index("tv_show_id")
    toks = {}
    for it, r in first.iterrows():
        t = [f"cat={r.tv_show_category}"]
        for g in (r.tv_show_genre_1, r.tv_show_genre_2, r.tv_show_genre_3):
            if g:
                t.append(f"gen={g}")
        for f, pre in ((r.director, "dir"), (r.actors, "act")):
            if f:
                t += [f"{pre}={x.strip()}" for x in str(f).split(",") if x.strip()][:10]
        y = str(r.year_of_production)[:4]
        if y.isdigit():
            t.append(f"dec={int(y)//10*10}")
        toks[it] = t
    if use_channel:
        chs = s.groupby("tv_show_id").channel_id.agg(lambda x: x.value_counts().index[0])
        for it, c in chs.items():
            toks[it].append(f"ch={c}")
    vocab = {}
    rows, cols = [], []
    item_ix = {it: i for i, it in enumerate(items)}
    for it, t in toks.items():
        for x in set(t):
            j = vocab.setdefault(x, len(vocab))
            rows.append(item_ix[it]); cols.append(j)
    F = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(items), len(vocab)))
    df = np.asarray((F > 0).sum(0)).ravel()
    idf = np.log((1 + len(items)) / (1 + df)) + 1
    F = F.multiply(idf).tocsr()
    nrm = np.sqrt(np.asarray(F.multiply(F).sum(1)).ravel()); nrm[nrm == 0] = 1
    F = sp.diags(1 / nrm) @ F
    return F.tocsr(), vocab


def score_cbf(d, F):
    """User profile = history-weighted sum of item vectors (L2-normalised); score = cosine with candidate vectors."""
    H = d.user_hist()
    P = H @ F
    n = np.sqrt(np.asarray(P.multiply(P).sum(1)).ravel()); n[n == 0] = 1
    P = sp.diags(1 / n) @ P
    return (P @ F[d.cand_col].T).toarray()


def coview_transition(d, min_co=1):
    """Item-item co-view graph: w_ij = number of users who watched both i and j (diag removed).
    Column-stochastic transition Q (Q[j, i] = P(i -> j)); dangling items keep their mass via d*v."""
    C = (d.Xb.T @ d.Xb).tocsr().astype(np.float64)
    C.setdiag(0); C.eliminate_zeros()
    if min_co > 1:
        C.data[C.data < min_co] = 0; C.eliminate_zeros()
    out = np.asarray(C.sum(1)).ravel()
    inv = np.where(out > 0, 1 / np.maximum(out, 1e-12), 0)
    T = sp.diags(inv) @ C  # row-stochastic: T[i, j] = P(i -> j)
    return T.T.tocsr(), out == 0


def score_ppr(d, alpha=0.5, tol=1e-6, max_iter=500, min_co=1):
    """Personalized PageRank: p <- alpha*(Q p + dangling mass * v) + (1-alpha)*v, v = user's normalised history.
    Run for all evaluation users at once (columns). Stop when max L1 change < tol or max_iter."""
    Q, dangling = coview_transition(d, min_co)
    Q = Q.toarray().astype(np.float64)  # co-view graph is dense; dense BLAS is much faster
    H = d.user_hist().toarray().T  # items x users
    s = H.sum(0); s[s == 0] = 1
    V = H / s
    p = V.copy()
    for it in range(max_iter):
        dm = p[dangling].sum(0)
        pn = alpha * (Q @ p + V * dm) + (1 - alpha) * V
        diff = np.abs(pn - p).sum(0).max()
        p = pn
        if diff < tol:
            break
    d.ppr_iters = it + 1
    return p[d.cand_col].T


def rank5(d, S, cold_fill):
    """Turn a score matrix into Top-5 lists over candidates. Users with no history (all-zero row) and
    any unfilled slots get the popularity list `cold_fill`. Ties broken by popularity then item id."""
    recs = {}
    popc = d.pop[d.cand_col]
    order_tb = np.lexsort((d.cand, -popc))  # tie-break order
    for r, u in enumerate(d.users):
        row = S[r]
        if not np.any(row > 0):
            recs[u] = list(cold_fill[:K]); continue
        idx = order_tb[np.argsort(-row[order_tb], kind="stable")[:K]]
        idx = [i for i in idx if row[i] > 0]
        lst = [int(d.cand[i]) for i in idx]
        for f in cold_fill:
            if len(lst) >= K:
                break
            if f not in lst:
                lst.append(int(f))
        recs[u] = lst
    return recs


def minmax_rows(S):
    mx = S.max(1, keepdims=True); mx[mx == 0] = 1
    return S / mx
