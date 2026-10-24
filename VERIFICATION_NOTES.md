# Verification log (for report §6)

## Checks done personally by Zlata Kranina
| Date | What | How | Result |
|---|---|---|---|
| 2026-10-24 | Median view-record length `duraton` = 396 s | Compared `logs/01_explore.log` (line 6) with `exam_quiz_AVITO.md` §3 | Match (396 s) |
| 2026-10-24 | Max view-record length = 342,145 s | Compared `logs/01_explore.log` (line 6) and the session printout of the record (user 6917501858074186427, channel 3, 2020-07-09 06:36:12 – 2020-07-13 05:38:37) with `exam_quiz_AVITO.md` §3 | Match (342,145 s) |
| 2026-10-24 | "Watched" threshold 300 s accepted; tie-break rule for old vs new rule set by Zlata (keep new if \|Δ\| < 0.002) | Read `figures/fig_durations.png`, `logs/15_durations.log`, `logs/65_rule_compare.log` | New rule kept (all \|Δ\| ≤ 0.0016) |
| 2026-10-24 | Hand AP@5 of the final method (LR fusion, fold A, rule 'new') for the two example users | From `logs/candidates6_lists.md` (Top-5 vs validation list), by hand. User B 8150816946177670084: hits at ranks 1, 4, \|R\|=12 → (1/1 + 2/4)/min(5,12) = 1.5/5 = 0.300. User A 2725163102947201526: hits at ranks 1, 3, 4, 5, \|R\|=83 → (1/1 + 2/3 + 3/4 + 4/5)/5 = 3.217/5 = 0.643 | Both match column "AP@5 Слияние (финал)" of `logs/80_candidates.log` / `logs/candidates6.csv` |
| 2026-10-24 | Choice of the two example users (by Zlata) | From the 6 candidates in `logs/80_candidates.log` | A = 2725163102947201526 (heavy viewer, methods differ, final method loses to PPR: failure case); B = 8150816946177670084 (1 watched airing in train, near cold start, fusion best). Cold user 10120801182216576239 not chosen: every method returns the same popularity fallback |
| 2026-10-25 | Data accounting (report Table 1) | Compared with `exam_quiz_AVITO.md` §3: 5,821,987 view records; 115,613 of 1,074,235 schedule rows of weeks 11–20 have an id → 115,613 / 1,074,235 = 10.8 % | Match |
| 2026-10-25 | References [1]–[7] of the report | Checked each reference herself | Done (Zlata) |
| 2026-10-25 | Report §5 (Discussion) | Written by Zlata; one claim corrected by the AI with her text kept: "v2 changed lists mostly by reordering" was false — 442 reorder-only, 558 with >= 1 programme replaced (`logs/97_v1_v2_diff.log`) | Marked red in the report for her review |

## Checks done by the AI in the session (independent method, logged)
| What | How | Result |
|---|---|---|
| PPR power iteration | Exact linear solve p = (1−α)(I − αM)⁻¹v for 3 users (`60_checks.log`, old rule) | L1 diff ~2e-8, same Top-5 |
| AP@5 code | Arithmetic printed step by step for one user (`60_checks.log`, old rule) | 3.8/5 = 0.76 = apk() |
| Clock alignment views vs schedule | View starts near programme starts for shifts −3..+3 h (`02_tzcheck.log`) | Peak at 0 h |
| Reproducibility of rule comparison | old/old column of `65_rule_compare.log` equals earlier runs | Identical to 4 decimals |

## Kaggle results (returned scores, screenshot `figures/kaggle_submissions_2026-10-24.png`, submitted by Zlata 2026-10-24)
| File | Public | Private | Our validation MAP@5 (fold A, rule 'new') |
|---|---|---|---|
| submissions/submission_lr_fusion.csv (final, sha256 bbdb011c…) | 0.41170 | 0.39020 | 0.7515 |
| submissions/submission_popularity.csv (baseline, sha256 73d71295…) | 0.28078 | 0.27750 | 0.6627 |
These two files were fixed before any Kaggle feedback existed (no Kaggle score was used to build them). For v2, see the protocol note below.

## Protocol note (decided by Zlata, 2026-10-24) — must be disclosed in the report as a limitation
After the two Kaggle submissions above, the popularity public score (0.28078) — and only that number — was used to
choose validation answer keys whose popularity MAP@5 is close to it (`logs/90_keys.log`). A challenger replaced the
final only if, on fold A, it was not worse on any of 5 keys and better on >= 2 (`logs/92_crosskey.log`).
M1 (labels re-defined by a Kaggle-calibrated key) failed; M2 (+2 daily-regularity features, original labels) passed
on all 5 keys -> `submissions/submission_lr_fusion_v2.csv` (`logs/71_final_m2.log`). The final method's Kaggle
score (0.41170) was not used for any choice. Public score of v2 may be slightly optimistic; report private next to it.

## Final designation (decided by Zlata before seeing the v2 score, 2026-10-25)
The designated final submission is `submissions/submission_lr_fusion_v2.csv`, regardless of its public score.
| submissions/submission_lr_fusion_v2.csv (v2, +daily regularity) | 0.40503 | 0.38357 | 0.7424 (K1, all 1260 users) |
(v2 row added 2026-10-25; screenshot `figures/kaggle_submissions_v2_2026-10-25.png`.)

## Final designation CHANGED (decided by Zlata AFTER seeing the v2 scores, 2026-10-25)
The earlier designation of v2 "regardless of its public score" was withdrawn after v2 scored lower than v1 on both
public (0.40503 vs 0.41170) and private (0.38357 vs 0.39020). The designated final is again
`submissions/submission_lr_fusion.csv` (v1). v1 itself was built and chosen by validation only, before any Kaggle
feedback; the return to it is a choice made with knowledge of the Kaggle scores of both files (public and private),
i.e. a test-set-informed choice between two candidates. This is disclosed in the report as a limitation.
v2 is reported as an improvement that held on all 5 validation answer keys but did not transfer to Kaggle.
