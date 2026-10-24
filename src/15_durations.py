"""Duration distributions behind the 'watched' threshold. Prints every number shown in the figure.
Panel 1: length of a single view record (duraton). Panel 2: summed overlap of a user with one labelled airing
(after the join, before the threshold) - this is the quantity the 300 s threshold is applied to."""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
v = pd.read_csv(f"{D}/dataset11-30.csv", usecols=["duraton"]).duraton.values
inter = pd.read_pickle(f"{OUT}/interactions.pkl")
ov, frac = inter.ov.values, inter.frac.values
print("view records", len(v), "min", v.min(), "share exactly", (v == v.min()).mean().round(4))
print("view-record min-duration: values < 70 s:", np.bincount(v[v < 70])[60:70] if (v < 70).any() else "none")
for name, x in [("view record duraton", v), ("user-airing overlap", ov)]:
    q = np.quantile(x, [.1, .25, .5, .75, .9])
    print(f"{name}: n={len(x)} quantiles 10/25/50/75/90% = {q.round(0)}")
    for s in [60, 120, 300, 600, 900, 1800]:
        print(f"   share < {s:5d} s: {(x < s).mean():.3f}")
print("user-airing pairs kept at >= 300 s:", int((ov >= 300).sum()), f"({(ov >= 300).mean():.3f}); "
      f"their share of watched seconds {ov[ov >= 300].sum() / ov.sum():.3f}")
print("fraction-of-airing quantiles:", np.quantile(frac, [.1, .25, .5, .75, .9]).round(3))
sch = pd.read_pickle(f"{OUT}/schedule.pkl")
print("labelled programme (airing) duration quantiles 10/50/90%:", np.quantile(sch.duration, [.1, .5, .9]))
print("share of labelled airings shorter than 300 s:", round((sch.duration < 300).mean(), 4))

BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor="#fcfcfb")
bins = np.logspace(np.log10(30), np.log10(4e5), 60)
for ax, x, title in [(axes[0], v, "Одна запись просмотра (duraton), 5,82 млн"),
                     (axes[1], ov, "Сумма пересечений пользователя с одним показом, 3,01 млн")]:
    ax.set_facecolor("#fcfcfb")
    ax.hist(x, bins=bins, color=BLUE, edgecolor="#fcfcfb", linewidth=0.6)
    ax.set_xscale("log")
    ax.axvline(300, color=INK, lw=1.2, ls="--")
    share = (x < 300).mean()
    unit = "записей" if ax is axes[0] else "пар"
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.text(330, ax.get_ylim()[1] * 0.98, f"порог 300 с: {share:.0%} {unit} короче", color=INK, fontsize=9, va="top",
            bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1.5))
    ax.set_xticks([60, 300, 900, 3600, 14400, 86400]); ax.set_xticklabels(["1 мин", "5 мин", "15 мин", "1 ч", "4 ч", "24 ч"])
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("длительность (логарифмическая шкала)", fontsize=9, color=MUTED)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8); ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("число записей", fontsize=9, color=MUTED)
axes[1].set_ylabel("число пар «пользователь–показ»", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_durations.png", dpi=160)
print("saved", f"{OUT}/fig_durations.png")
