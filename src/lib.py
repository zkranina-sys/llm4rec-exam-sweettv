"""Shared code: loading, temporal split, candidate pool, MAP@5, fallbacks. Imported by all method scripts."""
import numpy as np, pandas as pd

K = 5
OUT = "/home/claude/work/out"
DATA = "/home/claude/data"
THRESH_SEC = 300  # an airing counts as "watched" if the user's summed overlap with it is >= 300 s


def load():
    inter = pd.read_pickle(f"{OUT}/interactions.pkl")
    sch = pd.read_pickle(f"{OUT}/schedule.pkl")
    test_users = pd.read_csv(f"{DATA}/submission.csv", dtype={"user_id": "uint64"}).user_id.values
    return inter, sch, test_users


RULE = "new"       # chosen 2026-10-24 (65_rule_compare.log). "old": ov >= 300 s;  "new": ov >= 300 s OR frac >= 0.9 (fixes airings shorter than 300 s)
FRAC_MIN = 0.9


def watched(inter, thresh=THRESH_SEC, rule=None):
    rule = rule or RULE
    m = inter.ov >= thresh
    if rule == "new":
        m = m | (inter.frac >= FRAC_MIN)
    return inter[m]


def split(inter, train_weeks, target_weeks, thresh=THRESH_SEC, rule=None):
    """train = all labelled overlaps in train_weeks (models may use raw seconds);
    target = per-user set of distinct tv_show_ids watched (by `rule`) in target_weeks."""
    tr = inter[inter.week.isin(train_weeks)]
    tg = watched(inter[inter.week.isin(target_weeks)], thresh, rule)
    truth = tg.groupby("user_id").tv_show_id.agg(lambda x: set(x.tolist())).to_dict()
    return tr, truth


def candidates(sch, weeks):
    """Candidate pool = labelled programmes that air at least once in the target weeks (schedule only)."""
    return set(sch.loc[sch.week.isin(weeks), "tv_show_id"].unique().tolist())


def apk(actual, pred, k=K):
    """AP@k with denominator min(k, |actual|); duplicates in pred are not double-counted."""
    if not actual:
        return 0.0
    hits, s, seen = 0, 0.0, set()
    for i, p in enumerate(pred[:k]):
        if p in actual and p not in seen:
            hits += 1
            s += hits / (i + 1)
        seen.add(p)
    return s / min(k, len(actual))


def mapk(truth, recs, users, k=K):
    """Mean AP over `users`; users missing from truth score 0 (they are kept, as on Kaggle)."""
    return float(np.mean([apk(truth.get(u, set()), recs.get(u, []), k) for u in users]))


def topk_from_scores(score: dict, k=K, fill=()):
    """score: {item: value}. Highest first; ties by item id; fill remaining slots from `fill` (no duplicates)."""
    out = [i for i, _ in sorted(score.items(), key=lambda t: (-t[1], t[0]))[:k]]
    for f in fill:
        if len(out) >= k:
            break
        if f not in out:
            out.append(f)
    return out


def popularity(tr, cand, weeks_back=None, thresh=THRESH_SEC):
    """Items ranked by number of distinct users who watched them (>= thresh) in train, restricted to candidates."""
    w = watched(tr, thresh)
    if weeks_back:
        w = w[w.week > w.week.max() - weeks_back]
    pop = w.groupby("tv_show_id").user_id.nunique()
    pop = pop[pop.index.isin(cand)].sort_values(ascending=False, kind="mergesort")
    return pop
