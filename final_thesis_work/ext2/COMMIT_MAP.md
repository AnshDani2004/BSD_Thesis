# Commit IDs before and after 2026-10-14

On 2026-10-14 the commit messages on this branch were edited to remove trailer lines. Nothing else
changed: every file, author and date is the same, so each commit below has the same tree as before.
Only the commit IDs changed, from the first edited commit onward.

The records in `final_thesis_work/ext2/` (PREREG.md, RESULTS.md, STATUS.md, CPPI_SPEC.md) and
`final_thesis_work/ext2_adversarial/REPORT.md` are left exactly as written, so they cite the old
IDs. Use this table to find the current commit. Commit messages were updated to the new IDs.

| Old | New | Commit |
|---|---|---|
| 8645f12 | af2b5c3 | audit: bsd_thesis |
| dcafbf1 | 7de75da | audit: wire CPPI floor (pass wealth to RiskConstrainedKelly) and rerun experiments; not yet replicated |
| 388ebf5 | 22f38c0 | audit: independent replication of the CPPI rerun does not match (thesis text underspecifies the CPPI agent) |
| cd20b2c | c2e07a9 | ext2: CPPI agent spec from the code; optional explicit rng for Thompson and epsilon-greedy agents |
| a5ebfc0 | 91e0cdf | ext2: PREREG version 1 (before any exploratory or confirmatory outcome computation) and library code |
| 94d8cb8 | c318861 | ext2: exploratory run 1 outputs (exploratory seeds 0-999); amendments 1-3 to PREREG; tests for amended code |
| a8bd9ff | 55498a2 | FREEZE ext2: PREREG frozen (amendments 1-5), exploratory run 2 outputs and frozen thresholds; 22/22 tests pass |
| 5a50295 | 043d02f | ext2: confirmatory run (seeds 100000-101999, once, from FREEZE a8bd9ff); figures, digest; summarize.py and a figure-label fix added after the run (read outputs only) |
| eeb3fcc | 4a2ba6a | ext2: RESULTS.md (confirmatory findings, nulls, failed checks; all NOT YET REPLICATED) |
| d6d3c3d | 679b398 | ext2: STATUS.md |
| f222eb2 | 7a853ba | ext2: replication and adversarial review saved; RESULTS gains a review-outcome section superseding four overreaching statements |
| c61e190 | 441a053 | README: Extension 2 section (bounded log returns primary, disclosed deviation with both versions, CPPI agent as coded); decisions recorded |
| 11633cf | 7235979 | ext2 replication: native-threshold false alarms and delays (26.41 vs 2.14 per 1,000 pre-break steps) |
| 1114e90 | 94e9b3f | Replace the yfinance S&P 500 arrays with the download script plus checksums; derived results kept |
| ae12554 | eac8958 | Merge main (yfinance arrays replaced by download script plus checksums) into ext2-ablation |
