# Topic 2 v3: codebase audit and first algorithmic results (controlled testbed)

All numbers come from `results_v3/*.csv`, which `scripts/02-04` regenerated on 2026-10-24 from a corpus that `scripts/00_regen_testbed.sh` rebuilt with the paper's seeds.
Turns are log-free (frame-aware boundaries) and the test seeds are held out. CIs are 95% bootstrap intervals that resample connections.

## A. Audit findings (read these before rewriting the paper)
1. **The raw `results_v2/` corpus is missing.** The disk holds only per-turn caches. `cache/main_seed2_oracle.pkl` is truncated. `main_seed1_pred.pkl` matches `main_seed1_oracle.pkl` in size and in every label count, so the published "log-free" main_seed1 numbers may actually be oracle numbers. There were no `subM*` caches. I regenerated the corpus deterministically (51 chunks) into `data/raw_v3`.
2. **The labeller drops boundaries under multiplexing.** `label_boundaries` (common.py) maps every request to the next upstream record. Concurrent requests flushed in a single read land on the same record, so 19.4% of M=8 labels are lost (3,383 of 4,199). All learned segmenters in the paper were trained on these corrupted labels. See `seg_label_audit.csv`. The size-aware one-to-one labeller in `t2/seg.py` keeps 97.9% (4,113 of 4,199).
3. **The M=8 collapse is mostly a timing artefact, not a detection failure.** Client, server and observer share one asyncio loop, and the analysis rescales time by 1/TS = 10. That turns loop-queueing delay into "real-time" lag. At M=8, 38% of requests reach the observer more than 100 ms (real-time-equivalent) after `t_send` (`seg_observer_lag.csv`). Against client send time, no observer-side method can exceed about 0.62 F1@±100 ms. Against wire time, the paper rule already scores 0.885 and our frame-aware models score 0.978.
4. **Semantic-retry latency is not modelled in the main corpus.** Failed tools sleep for the same lognormal latency as successful ones (testbed2.py l.441). `testbed3.py --fail_latency instant` adds the 2–20 ms traceback regime.
5. **Tool identity is unidentifiable by construction beyond size.** Tools are drawn i.i.d. uniform: the transition χ² test gives p = 0.099. Argument length is `randint(20,300)` regardless of tool. Even the **exact** size plus observed latency reaches only 0.60 closed-set accuracy and 0.62 AUROC. The targets of 0.78 / 0.80 cannot be met on this corpus, and only a real-agent corpus can test Fix D.
6. **The context is strictly append-only, so Proposition 1 holds trivially.** `testbed3.py --ctx_policy window|summarize` adds non-monotonic contexts.
7. The baselines are re-implementations. Fix §4 (official DF / Whisper-Leak code) is **not done**. Its README links and licences have to be checked by hand.

## B. Results versus the directive's targets
| Fix | Metric | Paper / baseline | v3 | Target | Met? |
|---|---|---|---|---|---|
| A | M=8 F1@±100 ms, client clock | 0.606 (rule, reproduced) | 0.610 (frame GBM) | ≥0.80 | **No**: bounded by testbed lag (A.3) |
| A | M=8 F1@±100 ms, wire clock | 0.885 [0.879, 0.892] | **0.979 [0.968, 0.987]** | ≥0.80 | yes |
| A | Robustness to TLS write coalescing (p = 1) | — | 0.885 rule / 0.846 GBM | — | the frame rule is more robust |
| B | MedAPE <1 KB, HTTP/2 | 13.2% (42 B) | **7.4% (23 B)** | — | improved |
| B | MedAPE ≥1 KB, HTTP/2 | 3.2% | 2.8% | — | — |
| B | 90% PI coverage, HTTP/2 | 0.787 | **0.914 [0.902, 0.925]** (transport-Mondrian CQR) | 0.90 ± 0.015 | yes |
| B | Coverage, HTTP/1.1 | 0.750 | 0.888 [0.861, 0.914] | 0.90 ± 0.015 | borderline |
| B | Coverage, server-side state | 0.491 | 0.739 | 0.90 ± 0.015 | **No** (only about 150 training turns in that regime) |
| B | Sliding-window coverage (detect-and-abstain, 15% abstain) | 0.776 naive | 0.883 | — | improved |
| B | Summarisation break detection P/R | — | 0.72 / 0.52 | — | the reset-baseline estimator fails (MedAPE 0.80) |
| C | Parent accuracy, held-out main seed | 0.857 sequential | **0.955 [0.947, 0.963]** | ≥0.95 | yes |
| C | Parent accuracy, M=8 | 0.310 sequential | 0.634 | — | improved, still low |
| C | Edge macro-F1 (log-free) | 0.43 slice-only; 0.56 without context | **0.819 [0.809, 0.832]** | — | improved |
| C | Semantic-retry F1 / fork-join F1 | — | 0.62 / 0.66 | — | improved over the paper's 0.39 / 0.59, but different corpus |
| C | Effect of the latency cue on failfast | — | 0.807 vs 0.776 macro (retry F1 0.607 vs 0.606) | — | latency helps sequential/sub-agent, not retry |
| D | Closed-set accuracy / AUROC | 0.578 / 0.594 | 0.615 / 0.563 | 0.78 / 0.80 | **No**: corpus ceiling (A.5) |

## C. What this means for the manuscript
- Reframe segmentation along the directive's line. A rule suffices at M=1. Under multiplexing, the contribution is to (i) correct the labels and (ii) detect at the frame level with no NMS or silence gap. Report wire-clock F1 and name the client-clock gap as a capture-clock effect. The real-pcap captures in Kaggle stage 01 settle which effect dominates.
- Do not claim Fix D on synthetic data. Either drop open-set tool ID or make it a real-agent-only experiment.
- Server-side state needs either its own calibration set or an explicit statement that coverage holds only within the calibrated regime.

## D. Not done yet
Script `01` is a scaffold: vLLM, cloudflared and tshark conversion are written, but the LangGraph/AutoGen agent loop and the ANT adapter raise `NotImplementedError`. Still to do: the official baselines, the 6-defence sweep (def_* tags not regenerated), and a separate-process observer for the testbed.
