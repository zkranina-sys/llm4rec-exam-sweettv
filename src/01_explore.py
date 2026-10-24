"""Step 1: data accounting. Every number here is printed so it can be cited in the report."""
import sys, time
import numpy as np, pandas as pd

D = sys.argv[1]  # data dir
t0 = time.time()
v = pd.read_csv(f"{D}/dataset11-30.csv", dtype={"user_id": "uint64", "vsetv_id": "int32", "duraton": "int64"},
                parse_dates=["start_time", "stop_time"])
print("view rows", len(v), "users", v.user_id.nunique(), "channels", v.vsetv_id.nunique())
print("view time range", v.start_time.min(), v.stop_time.max())
d = v.duraton
print("duraton check (stop-start==duraton):", ((v.stop_time - v.start_time).dt.total_seconds() == d).mean())
print("duraton quantiles", d.quantile([.1, .25, .5, .75, .9, .99]).to_dict(), "max", d.max())
for s in [60, 120, 300, 600, 900, 1800]:
    print(f"  share of views < {s}s: {(d < s).mean():.3f}")
iso = v.start_time.dt.isocalendar()
print("views per ISO week:\n", iso.week.value_counts().sort_index().to_dict())

sub = pd.read_csv(f"{D}/submission.csv", dtype={"user_id": "uint64"})
print("submission rows", len(sub), "unique users", sub.user_id.nunique())
tu = set(sub.user_id)
print("test users present in view log:", len(tu & set(v.user_id)))

for f in ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv"]:
    s = pd.read_csv(f"{D}/{f}", dtype={"tv_show_id": "int64"})
    s["st"] = pd.to_datetime(s.start_time, format="%d.%m.%Y %H:%M:%S")
    lab = s.tv_show_id != 0
    print(f"== {f}: rows {len(s)}, labelled {lab.sum()} ({lab.mean():.3%}), distinct ids {s.loc[lab,'tv_show_id'].nunique()}, "
          f"channels {s.channel_id.nunique()}, range {s.st.min()} .. {s.st.max()}")
    print("   labelled share of airtime(seconds):", round(s.loc[lab, 'duration'].sum() / s.duration.sum(), 3))
    print("   categories:", s.loc[lab, 'tv_show_category'].value_counts().to_dict())
    print("   programme duration quantiles (labelled):", s.loc[lab, 'duration'].quantile([.1, .5, .9]).to_dict())
    print("   view channels covered by schedule:", len(set(v.vsetv_id) & set(s.channel_id)), "of", v.vsetv_id.nunique())
print("time", round(time.time() - t0, 1), "s")
