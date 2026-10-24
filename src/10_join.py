"""Step 2: join view log with the broadcast schedule.

Rule: a view (user, channel c, [a, b)) is matched to every schedule row on the SAME channel
whose airing interval [s, s+duration) overlaps [a, b); overlap seconds = min(b, e) - max(a, s) > 0.
Rows with tv_show_id = 0 (unlabelled airtime) are dropped: they are not items.
Overlaps of the same user with the same airing are summed (a user may zap away and come back).
Output: one row per (user, airing) with summed overlap seconds -> interactions.pkl
Also writes the cleaned, de-duplicated schedule -> schedule.pkl (all weeks, labelled rows only).
"""
import sys, time
import numpy as np, pandas as pd

D, OUT = sys.argv[1], sys.argv[2]
t0 = time.time()
K = 10**10  # key = channel * K + unix seconds

def sec(x):  # pandas 3 stores datetimes in microseconds -> convert explicitly to seconds
    return x.astype("datetime64[s]").astype("int64")

parts = []
for f in ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv"]:
    s = pd.read_csv(f"{D}/{f}", dtype={"tv_show_id": "int64", "year_of_production": "str"}, keep_default_na=False)
    s["st"] = sec(pd.to_datetime(s.start_time, format="%d.%m.%Y %H:%M:%S"))
    s["src"] = f[11:16]
    parts.append(s)
sch = pd.concat(parts, ignore_index=True)
n0 = len(sch)
dup = sch.duplicated(["channel_id", "st"], keep=False)
d2 = sch[dup].groupby(["channel_id", "st"]).agg(n_ids=("tv_show_id", "nunique"), n_dur=("duration", "nunique"))
print(f"schedule rows {n0}; rows in duplicated (channel,start) keys {dup.sum()}; keys with conflicting id {(d2.n_ids>1).sum()}, conflicting duration {(d2.n_dur>1).sum()}")
sch = sch.drop_duplicates(["channel_id", "st"], keep="first").sort_values(["channel_id", "st"]).reset_index(drop=True)
sch["en"] = sch.st + sch.duration
print("schedule rows after de-dup", len(sch))

v = pd.read_csv(f"{D}/dataset11-30.csv", dtype={"user_id": "uint64", "vsetv_id": "int64"})
v["a"] = sec(pd.to_datetime(v.start_time)); v["b"] = sec(pd.to_datetime(v.stop_time))
print("views", len(v))

key = sch.channel_id.values * K + sch.st.values
ch = sch.channel_id.values
va = v.vsetv_id.values * K + v.a.values
vb = v.vsetv_id.values * K + v.b.values
i0 = np.searchsorted(key, va, side="right") - 1
i0 = np.maximum(i0, 0)
i0 = np.where(ch[i0] != v.vsetv_id.values, i0 + 1, i0)  # view begins before channel's first programme
i1 = np.searchsorted(key, vb, side="left") - 1          # last programme starting before b
n = i1 - i0 + 1
has = (n > 0) & (i0 < len(key))
print("views with no schedule on their channel/time:", int((~has).sum()))
vi = np.repeat(np.nonzero(has)[0], n[has])
off = np.arange(len(vi)) - np.repeat(np.cumsum(n[has]) - n[has], n[has])
si = np.repeat(i0[has], n[has]) + off
assert (ch[si] == v.vsetv_id.values[vi]).all(), "channel mismatch in join"
ov = np.minimum(v.b.values[vi], sch.en.values[si]) - np.maximum(v.a.values[vi], sch.st.values[si])
keep = ov > 0
vi, si, ov = vi[keep], si[keep], ov[keep]
print("view x airing overlap pairs", len(vi))
lab = sch.tv_show_id.values[si] != 0
tot = pd.Series(ov).groupby(vi).sum()
print("share of watched seconds falling on labelled airtime:", round(ov[lab].sum() / ov.sum(), 4))
vi, si, ov = vi[lab], si[lab], ov[lab]
pairs = pd.DataFrame({"user_id": v.user_id.values[vi], "sidx": si, "ov": ov})
inter = pairs.groupby(["user_id", "sidx"], as_index=False).ov.sum()
inter["tv_show_id"] = sch.tv_show_id.values[inter.sidx]
inter["channel_id"] = sch.channel_id.values[inter.sidx]
inter["st"] = sch.st.values[inter.sidx]
inter["dur"] = sch.duration.values[inter.sidx]
inter["ov"] = np.minimum(inter.ov, inter.dur)  # cannot watch more than the airing
inter["frac"] = inter.ov / inter.dur.clip(lower=1)
inter["week"] = pd.to_datetime(inter.st, unit="s").dt.isocalendar().week.astype(int).values
print("user-airing interactions (labelled)", len(inter), "users", inter.user_id.nunique(), "shows", inter.tv_show_id.nunique())
print("overlap seconds quantiles", inter.ov.quantile([.1, .25, .5, .75, .9]).to_dict())
print("fraction-of-programme quantiles", inter.frac.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
inter.to_pickle(f"{OUT}/interactions.pkl")
lab_sch = sch[sch.tv_show_id != 0].copy()
lab_sch["week"] = pd.to_datetime(lab_sch.st, unit="s").dt.isocalendar().week.astype(int).values
lab_sch.to_pickle(f"{OUT}/schedule.pkl")
print("time", round(time.time() - t0, 1), "s")
