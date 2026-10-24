# sweet.tv TV program recommender — LLM4Rec final exam (AVITO group)

Кранина Злата (Zlata Kranina), НИУ ВШЭ, курс LLM4Rec, 2026-10-25.

## Final submission
**`submissions/submission_lr_fusion.csv` — FINAL (v1)**: logistic-regression score fusion of personal history,
Personalized PageRank, CBF and popularity. Kaggle: public **0.41170**, private 0.39020
(screenshot `figures/kaggle_submissions_v2_2026-10-25.png`).

Other files (not final):
| File | What | Kaggle public / private |
|---|---|---|
| `submissions/submission_popularity.csv` | popularity baseline | 0.28078 / 0.27750 |
| `submissions/submission_lr_fusion_v2.csv` | v2: v1 + 2 daily-regularity features | 0.40503 / 0.38357 |
| `submissions/submission_ppr.csv` | PPR alone, alpha = 0.7 | not submitted |
| `submissions/archive/submission_lr_fusion_oldrule.csv` | v1 under the old "watched" rule (>= 300 s only) | not submitted |

How the final was chosen (incl. the use of Kaggle feedback) is documented in `VERIFICATION_NOTES.md`.

## Data (not included)
The competition files are not in this repository (≈ 890 MB, owned by the competition). Download them from
https://www.kaggle.com/competitions/sweettv-tv-program-recommender (Data tab; accept the competition rules first):
`dataset11-30.csv`, `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv`, `submission.csv`.

## Environment
Python 3.13.17, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, matplotlib (figure only);
Linux, 2 CPU, 7 GB RAM. Seed: `random_state = 42` (logistic regression), `default_rng(42)` (bootstrap, random baseline).
Note: pandas 3 stores datetimes in microseconds; the code converts explicitly with `astype("datetime64[s]")`.

## Paths
The scripts were run with absolute paths: data in `/home/claude/data`, scripts in `/home/claude/work`,
intermediate files in `/home/claude/work/out`. To run elsewhere, change `DATA` and `OUT` in `src/lib.py`
and the `sys.path.insert(...)` / `runpy` paths at the top of the scripts, and pass your data folder as the argument
where a script takes one.

## Reproduce (order; each script's output is in `logs/<script>.log`)
| Step | Script | Notes |
|---|---|---|
| Data accounting | `01_explore.py <data>` | sizes, duration distribution |
| Clock / segmentation checks | `02_tzcheck.py <data>`, `03_align.py <data>` | |
| Join views x schedule | `10_join.py <data> <out>` | same channel + interval overlap; `tv_show_id = 0` dropped -> `out/interactions.pkl`, `out/schedule.pkl` |
| Duration figure | `15_durations.py <data> <out>` | `figures/fig_durations.png` |
| Baseline, methods, linear hybrid | `20_baseline.py`, `30_validate.py`, `31_explore_hist.py`, `40_hybrid.py` | **run with the OLD rule**: set `RULE = "old"` in `lib.py` to reproduce these logs |
| Checks (threshold sensitivity, exact PPR, hand AP) | `60_checks.py` | **OLD rule** (as above) |
| Rule comparison old vs new | `65_rule_compare.py` | |
| Learned fusion (v1) | `50_ltr.py` | new rule; `50_ltr_oldrule.log` = earlier old-rule run |
| **Final v1 submission** | `70_final.py` | new rule -> `submission_lr_fusion.csv`; `70_final_oldrule.log` = earlier run |
| Example users, diagnostics | `80_candidates.py`, `82_diag.py` | |
| Answer-key calibration, cross-key test, v2 | `90_keys.py`, `91_keys_se.py`, `92_crosskey.py`, `71_final_m2.py` | uses the Kaggle popularity score, see `VERIFICATION_NOTES.md` |
| All numbers of the report | `95_report_numbers.py` | single source for report tables |
| v1 vs v2 list differences | `97_v1_v2_diff.py` | run from the `out` folder's parent |

Shared code: `src/lib.py` (loading, split, candidates, MAP@5, "watched" rule), `src/methods.py` (popularity,
history, CBF, PPR, ranking). `report/build_ru.js` generates the Russian report (Node, `docx` package).

## Protocol in one paragraph
Validation: fold A = train weeks 11–24 / validation 25–30, fold B = 11–18 / 19–24. "Watched" = summed overlap with an
airing >= 300 s or >= 90 % of the airing. Candidates = labelled programmes airing in the target weeks (schedule only).
Repeats allowed for all methods. Users without history -> top-5 popularity. MAP@5 with denominator min(5, |R_u|).
Final model: trained on weeks 11–30, candidates from the week 31–42 schedule; no week 31–42 viewing exists or is used.
