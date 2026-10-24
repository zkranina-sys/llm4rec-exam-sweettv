"""Are view intervals already cut at programme boundaries? Check exact equality of view start/stop with schedule start/end."""
import sys
import numpy as np, pandas as pd
D = sys.argv[1]
v = pd.read_csv(f"{D}/dataset11-30.csv", parse_dates=["start_time","stop_time"])
v = v[(v.start_time >= "2020-04-06") & (v.start_time < "2020-04-20")]
s = pd.read_csv(f"{D}/export_arh_11-20-final.csv", usecols=["channel_id","start_time","duration","tv_show_id"])
s["st"] = pd.to_datetime(s.start_time, format="%d.%m.%Y %H:%M:%S").astype("datetime64[s]").astype("int64")
s["en"] = s.st + s.duration
K = 10**10
starts = set((s.channel_id.astype(np.int64)*K + s.st).values)
ends = set((s.channel_id.astype(np.int64)*K + s.en).values)
vst = v.vsetv_id.astype(np.int64)*K + v.start_time.astype("datetime64[s]").astype("int64")
ven = v.vsetv_id.astype(np.int64)*K + v.stop_time.astype("datetime64[s]").astype("int64")
a = vst.isin(starts); b = ven.isin(ends)
print("views", len(v), "start==prog start", a.mean().round(4), "stop==prog end", b.mean().round(4), "both", (a&b).mean().round(4))
# for views not starting at boundary: are they starting at an exact second near the boundary? print examples
print(v[~a].head(8).to_string())
print(v[a].head(8).to_string())
