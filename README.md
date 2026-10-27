# What Encrypted Agent Traffic Reveals

Code, processing scripts, derived results and LaTeX source for the paper

> S. M. T. U. Salahin, "What Encrypted Agent Traffic Reveals: Log-Free Turn Segmentation, Tool-Result Size Recovery and Execution-Graph Reconstruction over TLS 1.3 and HTTP/2," preprint, 2026.

The paper studies what a passive on-path observer can learn about an LLM agent from TLS 1.3 record headers alone (length, direction, timing): where each agent turn begins, how many bytes each tool returned, and how turns depend on each other. Because a stateless model API receives the full conversation history on every call, the growth of consecutive request ciphertexts reveals tool-result sizes (Proposition 1 in the paper).

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Controlled loopback testbed: mock HTTP/2 + HTTP/1.1 API server (hyper-h2, TLS 1.3, ALPN), on-path TLS-record relay and scripted agents. `testbed3.py` adds context policies, fail-fast latency and SSE batching. |
| `t2/` | Shared library: raw-capture to parquet conversion and bootstrap (`data.py`), pcap parsing (`pcap.py`), turn-boundary labellers and frame features (`seg.py`), observer turn tables (`turns.py`). |
| `scripts/` | Numbered analysis pipeline (see below). |
| `kaggle/` | Kaggle/Colab notebooks-as-scripts used for the real-stack captures (Qwen2.5-1.5B/7B behind a Cloudflare tunnel, unmodified LangGraph and AutoGen agents) and for the official Deep Fingerprinting baseline on GPU. |
| `results/` | Derived CSV/JSON/LaTeX tables reported in the paper and run logs. |
| `figures/` | Rendered figures (PDF/PNG). |
| `paper/` | Manuscript source (LaTeX, IEEEtran class). `sec_v3_real.tex` is generated from `sec_v3_real.tmpl` by `build_v3_section.py`. |

## Pipeline

| Script | Purpose |
|---|---|
| `00_regen_testbed.sh` | Regenerate the controlled corpus deterministically from seeds 1–3 (≈15 min on 4 cores). |
| `01_real_capture_pipeline.py` | Real-stack capture scaffold (model server + cloudflared + tcpdump). |
| `02_segmentation_multiplex.py` | Log-free turn segmentation: burst rule, frame rule, frame-aware GBM, conv+transformer. |
| `03_size_recovery_cqr.py` | Tool-result size recovery (Prop. 1) and split-conformal quantile regression. |
| `04_graph_and_tool_id.py` | Parent linking (constrained MAP), edge typing, tool identification on the testbed. |
| `05_figures_tables.py` | Testbed figures and tables. |
| `06_freetier_capture.py` | Live free-tier provider capture (own accounts, `:free` models only). |
| `07_official_baselines.py` | Whisper-Leak and Deep Fingerprinting run from the authors' code; only the input loader is changed. |
| `08_ant_external.py` | External validation on the public ANT agent-traffic corpus. |
| `09_real_eval.py` | Segmentation and size recovery on real stacks. |
| `10_defences.py` | Nine-condition defence sweep (padding, gzip, server state, jitter, SSE batching, combined). |
| `11_conversation_tracker.py` | Joint conversation tracking for interleaved / per-request-connection traffic. |
| `12_verify_manuscript_numbers.py` | Checks every number in the manuscript against `results/`. |
| `13_cqr_percorpus.py` | Per-corpus Mondrian conformal intervals on real stacks. |
| `14_tool_id_real.py` | Tool identification on real agent frameworks. |
| `15_real_edges_padding.py` | Edge typing and padding on real captures. |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/00_regen_testbed.sh                     # controlled corpus -> data/
for s in 02 03 04; do python scripts/${s}_*.py; done # testbed analyses -> results/
python scripts/05_figures_tables.py                  # figures
```

Paths are controlled by environment variables `T2_RAW`, `T2_PQ`, `T2_RES`, `T2_FIG` (defaults are relative to the repository root). Provider API keys for `06_freetier_capture.py` are read from environment variables and are never stored in the repository.

## Data

Raw captures (≈0.9 GB) are not stored in git. The controlled corpus is regenerated exactly by `00_regen_testbed.sh`. Real-stack and live-provider record-header captures (TLS record metadata only, no payloads or keys) will be deposited in an archival data repository; the link will be added here. The ANT corpus must be obtained from its authors (Li et al., arXiv:2610.06514).

## Baselines

Third-party baselines are not vendored. Clone them next to this repository:

- Whisper-Leak: https://github.com/yo-yo-yo-jbo/whisper_leak (commit `9dc95fd`)
- Deep Fingerprinting: https://github.com/deep-fingerprinting/df

## Ethics

All captures were of the authors' own synthetic agent sessions on their own accounts within free-tier limits. No third-party traffic was observed.

## Citation

See `CITATION.cff`.

## License

Code is released under the MIT License (`LICENSE`). The manuscript in `paper/` is © the author.
