"""05 - Publication figures (PDF+PNG) and LaTeX tables from results_v3/*.  Requires apply_figure_style/panel_letter
(figure-style helpers) when run inside the analysis kernel; falls back to matplotlib defaults otherwise."""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if "__file__" in globals() else os.getcwd()
RES, FIG = f"{ROOT}/results_v3", f"{ROOT}/figures_v3"
try: apply_figure_style()
except NameError:
    def panel_letter(ax, l): ax.text(-0.15, 1.08, l, transform=ax.transAxes, fontweight="bold")
Mmap = {"subM1": 1, "subM2": 2, "subM4": 4, "subM8": 8}
def mx(ax):
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"]); ax.set_xlabel("Streams per connection, M")

# ---------------- Fig. segmentation
S = pd.read_csv(f"{RES}/seg_summary.csv"); LAG = pd.read_csv(f"{RES}/seg_observer_lag.csv")
meth = {"rule_paper": ("Paper rule (size + silence gap)", "#9a9a9a"), "gbm_nms_paper": ("Paper GBM + NMS", "#c9a86a"),
        "frame_gbm": ("Frame-aware GBM (ours)", "#1f5fa8"), "frame_dct": ("Frame-aware conv+transformer (ours)", "#7fb0e0")}
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.42))
for ax, clk, ttl in ((axs[0], "app", "Against client send time"), (axs[1], "wire", "Against wire time")):
    for m, (lab, c) in meth.items():
        d = S[(S.method == m) & (S.clock == clk) & (S.multiplexed) & S.tag.isin(Mmap)].copy(); d["M"] = d.tag.map(Mmap); d = d.sort_values("M")
        ax.errorbar(d.M, d["F1@100"], yerr=[d["F1@100"] - d["F1@100_lo"], d["F1@100_hi"] - d["F1@100"]], marker="o", ms=3, lw=1.6 if "frame" in m else 1, ls="--" if m in ("frame_dct","rule_paper") else "-",
                    color=c, label=lab, capsize=2)
    ax.axhline(0.80, ls=":", color="k", lw=.7); mx(ax); ax.set_title(ttl, loc="left"); ax.set_ylim(0.25, 1.04)
axs[0].set_ylabel("Boundary F1 at ±100 ms"); axs[0].text(1.05, 0.82, "target 0.80", fontsize=6)
axs[1].legend(frameon=False, loc="lower left", fontsize=5.5)
d = LAG[LAG.multiplexed & LAG.tag.isin(Mmap)].copy(); d["M"] = d.tag.map(Mmap); d = d.sort_values("M")
axs[2].plot(d.M, d.frac_gt_100ms, marker="o", color="#b2182b", ms=3); axs[2].plot(d.M, d.frac_gt_50ms, marker="s", color="#ef8a62", ms=3)
axs[2].text(8.6, d.frac_gt_100ms.iloc[-1], "> 100 ms", fontsize=6, va="center"); axs[2].text(8.6, d.frac_gt_50ms.iloc[-1], "> 50 ms", fontsize=6, va="center")
mx(axs[2]); axs[2].set_xlim(0.8, 20); axs[2].set_ylabel("Fraction of requests"); axs[2].set_title("Testbed send→observer lag", loc="left")
for a, l in zip(axs, "abc"): panel_letter(a, l)
fig.savefig(f"{FIG}/fig_seg_multiplex.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/fig_seg_multiplex.png", dpi=300, bbox_inches="tight")

# ---------------- Fig. size / CQR
R = pd.read_csv(f"{RES}/size_results.csv")
est = {"paper_lgbm_q50 (raw 90% PI)": ("Paper LightGBM quantiles", "#9a9a9a"), "ours_q50 (raw 90% PI)": ("Ours, raw quantiles", "#7fb0e0"),
       "ours_q50 + CQR": ("Ours + split CQR", "#4a7fc0"), "ours_q50 + transport-Mondrian-CQR": ("Ours + transport-Mondrian CQR", "#08306b")}
fig2, ax2 = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.3))
tags = ["main_seed3", "h1_stream", "state", "ctx_window"]; tl = {"main_seed3": "HTTP/2", "h1_stream": "HTTP/1.1", "state": "server\nstate", "ctx_window": "sliding\nwindow*"}
for i, (e, (lab, c)) in enumerate(est.items()):
    d = R[(R.estimator == e) & (R.regime == "all")].set_index("tag").reindex(tags)
    if e == "ours_q50 + transport-Mondrian-CQR":
        w = R[(R.estimator == "break-aware") & (R.regime == "all") & (R.tag == "ctx_window")].set_index("tag"); d.loc["ctx_window"] = w.iloc[0]
    x = np.arange(len(tags)) + (i - 1.5) * 0.18
    ax2[0].errorbar(x, d.PICP90, yerr=[d.PICP90 - d.PICP90_lo, d.PICP90_hi - d.PICP90], fmt="o", color=c, ms=3, capsize=2, label=lab)
ax2[0].axhspan(0.885, 0.915, color="#e3e3e3", zorder=0); ax2[0].axhline(0.9, color="k", lw=.6)
ax2[0].set_xticks(range(len(tags))); ax2[0].set_xticklabels([tl[t] for t in tags]); ax2[0].set_ylabel("Coverage of nominal 90% interval")
ax2[0].set_ylim(0.35, 1.0); ax2[0].set_title("Conformal calibration fixes HTTP/2, not server state", loc="left"); ax2[0].legend(frameon=False, fontsize=5.5, loc="lower left")
for j, (e, lab, c) in enumerate([("paper_lgbm_q50 (raw 90% PI)", "Paper LightGBM", "#9a9a9a"), ("ours_q50 + transport-Mondrian-CQR", "Ours", "#08306b")]):
    d = R[(R.estimator == e) & (R.tag == "main_seed3")].set_index("regime").reindex(["<1KB", "1-10KB", ">10KB"])
    x = np.arange(3) + (j - .5) * 0.25
    ax2[1].errorbar(x, d.MedAPE * 100, yerr=[(d.MedAPE - d.MedAPE_lo) * 100, (d.MedAPE_hi - d.MedAPE) * 100], fmt="o", color=c, ms=4, capsize=2, label=lab)
ax2[1].set_xticks(range(3)); ax2[1].set_xticklabels(["<1 KB", "1–10 KB", ">10 KB"]); ax2[1].set_ylabel("Median absolute error (%)"); ax2[1].set_xlabel("True tool-result size")
ax2[1].set_title("Sub-1 KB error roughly halved (log-free HTTP/2)", loc="left"); ax2[1].legend(frameon=False, fontsize=6); ax2[1].set_ylim(0, None); ax2[1].margins(x=0.12)
for a, l in zip(ax2, "ab"): panel_letter(a, l)
fig2.savefig(f"{FIG}/fig_size_cqr.pdf", bbox_inches="tight"); fig2.savefig(f"{FIG}/fig_size_cqr.png", dpi=300, bbox_inches="tight")

# ---------------- Fig. graph
P = pd.read_csv(f"{RES}/graph_parent.csv"); E = pd.read_csv(f"{RES}/graph_edges.csv")
lk = {"sequential": ("Sequential chain", "#9a9a9a"), "paper_ctxgrowth": ("Paper context-growth", "#c9a86a"), "hybrid_rule": ("Hybrid rule", "#7fb0e0"), "hybrid_map": ("Hybrid MAP (ours)", "#08306b")}
fig3, ax3 = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1.3]))
for m, (lab, c) in lk.items():
    d = P[(P.linker == m) & (P.conns == "all") & P.tag.isin(Mmap)].copy(); d["M"] = d.tag.map(Mmap); d = d.sort_values("M")
    ax3[0].errorbar(d.M, d.parent_acc, yerr=[d.parent_acc - d.lo, d.hi - d.parent_acc], marker="o", ms=3, color=c, label=lab, capsize=2, lw=1.6 if m == "hybrid_map" else 1)
ax3[0].axhline(0.95, ls=":", color="k", lw=.7); mx(ax3[0]); ax3[0].set_ylabel("Parent-link accuracy"); ax3[0].set_ylim(0.2, 1.02)
ax3[0].set_title("MAP linking degrades least", loc="left"); ax3[0].legend(frameon=False, fontsize=5.5, loc="lower left")
cls = ["start", "sequential", "parallel_fork_join", "transport_retry", "semantic_retry", "sub_agent"]; nm = ["start", "seq.", "fork-join", "transport\nretry", "semantic\nretry", "sub-agent"]
for j, (mod, lab, c) in enumerate([("slice_only", "Slice-only features", "#9a9a9a"), ("v3_no_context", "Ours without context growth", "#7fb0e0"), ("v3_full", "Ours, full", "#08306b")]):
    r = E[(E.tag == "main_seed3") & (E.model == mod)].iloc[0]
    ax3[1].bar(np.arange(6) + (j - 1) * 0.26, [r[f"F1_{k}"] for k in cls], 0.26, color=c, label=f"{lab} (macro {r.macroF1:.2f})")
ax3[1].set_xticks(range(6)); ax3[1].set_xticklabels(nm, fontsize=6); ax3[1].set_ylabel("Per-class F1"); ax3[1].set_ylim(0, 1.05)
ax3[1].set_title("Edge typing, log-free, held-out seed", loc="left"); ax3[1].legend(frameon=False, fontsize=5.5, loc="upper left", ncol=1)
for a, l in zip(ax3, "ab"): panel_letter(a, l)
fig3.savefig(f"{FIG}/fig_graph.pdf", bbox_inches="tight"); fig3.savefig(f"{FIG}/fig_graph.png", dpi=300, bbox_inches="tight")

# ---------------- LaTeX tables
def tex(df, path, cap, lab):
    open(path, "w").write(df.to_latex(index=False, float_format="%.3f", caption=cap, label=lab, escape=True))
s = S[S.multiplexed & S.method.isin(meth)].copy(); s["M"] = s.tag.map(Mmap).fillna(1)
s["F1@100 [95% CI]"] = s.apply(lambda r: f"{r['F1@100']:.3f} [{r['F1@100_lo']:.3f}, {r['F1@100_hi']:.3f}]", axis=1)
tex(s[["method", "clock", "tag", "n_conn", "F1@50", "F1@100 [95% CI]", "F1@250"]], f"{RES}/tab_segmentation.tex", "Boundary F1 on multiplexed connections (1,000-sample connection bootstrap).", "tab:seg")
r = R[R.regime.isin(["all", "<1KB", ">=1KB"])][["tag", "estimator", "regime", "n", "MedAE_B", "MedAPE", "PICP90"]]
tex(r, f"{RES}/tab_size.tex", "Tool-result size recovery and 90\\% interval coverage (log-free boundaries).", "tab:size")
tex(P[P.conns == "all"][["tag", "linker", "n_turns", "parent_acc", "lo", "hi"]], f"{RES}/tab_parent.tex", "Parent-link accuracy.", "tab:parent")
tex(E[["tag", "model", "macroF1", "lo", "hi"] + [f"F1_{k}" for k in cls]], f"{RES}/tab_edges.tex", "Edge-type F1 (undetected turns count as errors).", "tab:edges")
O = pd.read_csv(f"{RES}/tool_openset_reps.csv").groupby(["tag", "features"])[["closed_acc", "AUROC_msp", "AUROC_energy", "AUROC_maha"]].mean().reset_index()
tex(O, f"{RES}/tab_toolid.tex", "Open-set tool identification (mean of 5 held-out-tool splits).", "tab:tool")
print("figures + tables written")
