"""Check clock alignment between view log and schedule: for shifts of -3..+3 h, how many
view starts fall within 60 s after a programme start on the same channel (people tune in at the start)."""
import sys
import numpy as np, pandas as pd
D = sys.argv[1]
v = pd.read_csv(f"{D}/dataset11-30.csv", usecols=["vsetv_id", "start_time"], parse_dates=["start_time"])
v = v[(v.start_time >= "2020-04-06") & (v.start_time < "2020-04-20")]  # 2 weeks, DST already active
s = pd.read_csv(f"{D}/export_arh_11-20-final.csv", usecols=["channel_id", "start_time"])
s["st"] = pd.to_datetime(s.start_time, format="%d.%m.%Y %H:%M:%S")
vs = (v.vsetv_id.astype(np.int64) * 10**10 + v.start_time.astype("datetime64[s]").astype("int64")).values
for h in range(-3, 4):
    ss = np.sort((s.channel_id.astype(np.int64) * 10**10 + s.st.astype("datetime64[s]").astype("int64") + h * 3600).values)
    i = np.searchsorted(ss, vs, side="right") - 1
    ok = i >= 0
    gap = vs[ok] - ss[i[ok]]
    print(f"shift {h:+d} h: share of view starts 0-60 s after a programme start = {(gap <= 60).mean():.4f}, 0-300s = {(gap<=300).mean():.4f}")
# schedule continuity
s = s.sort_values(["channel_id", "st"])
dur = pd.read_csv(f"{D}/export_arh_11-20-final.csv", usecols=["duration"]).loc[s.index, "duration"].values
nxt = s.groupby("channel_id").st.shift(-1)
diff = (nxt - s.st).dt.total_seconds().values - dur
print("schedule: next_start - (start+duration) quantiles:", np.nanquantile(diff, [.01, .1, .5, .9, .99]))
