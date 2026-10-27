"""12 - Audit: every decimal number in paper_tex/{main,sec_v3_real}.tex must equal (at its printed precision, either as a
fraction or as a percentage) some value in results_v3/*.csv|json.  Also flags v2 leftovers and undefined refs.
Output: results_v3/manuscript_audit.csv (one row per number, matched file/column or UNMATCHED)."""
import os, re, glob, json, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); RES = f"{ROOT}/results_v3"
V = {}   # rounded-string -> source
def add(x, src):
    try: x = float(x)
    except Exception: return
    if not np.isfinite(x): return
    for v in (x, 100 * x, x / 1000, -x):
        for p in (0, 1, 2, 3):
            V.setdefault(f"{round(v, p):.{p}f}", src)
for f in glob.glob(f"{RES}/*.csv"):
    try: D = pd.read_csv(f)
    except Exception: continue
    for c in D.columns:
        for v in pd.to_numeric(D[c], errors="coerce").dropna().unique(): add(v, f"{os.path.basename(f)}:{c}")
for f in glob.glob(f"{RES}/*.json"):
    def walk(o):
        if isinstance(o, dict): [walk(v) for v in o.values()]
        elif isinstance(o, list): [walk(v) for v in o]
        else: add(o, os.path.basename(f))
    try: walk(json.load(open(f)))
    except Exception: pass
DESIGN = {"0.17", "0.08", "0.25", "1.5", "2.0", "0.80", "0.90", "0.10", "0.6", "0.0"}   # chance levels / targets / config
rows = []
for tf in ("paper_tex/main.tex", "paper_tex/sec_v3_real.tex"):
    txt = open(f"{ROOT}/{tf}").read()
    body = re.sub(r"(?m)^%.*$", "", txt); body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", body, flags=re.S)
    body = re.sub(r"\\section\{Baseline hyper-parameter parity\}.*", "", body, flags=re.S)     # config values, not results
    for mt in re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w.])", body):
        s = mt.group(1); ctx = body[max(0, mt.start() - 40): mt.end() + 15].replace("\n", " ")
        src = V.get(s) or ("design/target" if s in DESIGN else None)
        rows.append(dict(file=tf, number=s, source=src or "UNMATCHED", context=ctx))
A = pd.DataFrame(rows); A.to_csv(f"{RES}/manuscript_audit.csv", index=False)
labels = set(re.findall(r"\\label\{([^}]+)\}", open(f"{ROOT}/paper_tex/main.tex").read() + open(f"{ROOT}/paper_tex/sec_v3_real.tex").read()))
refs = set(re.findall(r"\\ref\{([^}]+)\}", open(f"{ROOT}/paper_tex/main.tex").read() + open(f"{ROOT}/paper_tex/sec_v3_real.tex").read()))
cites = set(sum([c.split(",") for c in re.findall(r"\\cite\{([^}]+)\}", open(f"{ROOT}/paper_tex/main.tex").read() + open(f"{ROOT}/paper_tex/sec_v3_real.tex").read())], []))
bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", open(f"{ROOT}/paper_tex/main.tex").read()))
v2 = [s for s in ("0.618", "0.963 [", "0.826", "0.87 for our linker", "60--76", "0.56 \\pm") if s in open(f"{ROOT}/paper_tex/main.tex").read()]
um = A[A.source == "UNMATCHED"]
# ---- value-level matching is weak (most 2-decimal numbers match some CSV cell), so key prose claims are checked
#      against their exact provenance: (printed text, file, row filter, column, scale, decimals)
def q(f, flt, col):
    D = pd.read_csv(f"{RES}/{f}")
    for k, v in flt.items(): D = D[D[k] == v]
    assert len(D) >= 1, (f, flt); return float(D[col].iloc[0])
CL = [("0.979", "seg_summary.csv", dict(method="frame_gbm", clock="wire", tag="subM8", multiplexed=True), "F1@100", 1, 3),
      ("0.885", "seg_summary.csv", dict(method="rule_paper", clock="wire", tag="subM8", multiplexed=True), "F1@100", 1, 3),
      ("0.997", "seg_summary.csv", dict(method="rule_paper", clock="wire", tag="main_seed3", multiplexed=False), "F1@100", 1, 3),
      ("7.4", "size_results.csv", dict(tag="main_seed3", estimator="ours_q50 + transport-Mondrian-CQR", regime="<1KB"), "MedAPE", 100, 1),
      ("13.2", "size_results.csv", dict(tag="main_seed3", estimator="paper_lgbm_q50 (raw 90% PI)", regime="<1KB"), "MedAPE", 100, 1),
      ("2.8", "size_results.csv", dict(tag="main_seed3", estimator="ours_q50 + transport-Mondrian-CQR", regime=">=1KB"), "MedAPE", 100, 1),
      ("0.955", "graph_parent.csv", dict(tag="main_seed3", linker="hybrid_map", conns="all"), "parent_acc", 1, 3),
      ("0.857", "graph_parent.csv", dict(tag="main_seed3", linker="sequential", conns="all"), "parent_acc", 1, 3),
      ("0.634", "graph_parent.csv", dict(tag="subM8", linker="hybrid_map", conns="all"), "parent_acc", 1, 3),
      ("0.819", "graph_edges.csv", dict(tag="main_seed3", model="v3_full"), "macroF1", 1, 3),
      ("0.558", "graph_edges.csv", dict(tag="main_seed3", model="v3_no_context"), "macroF1", 1, 3),
      ("0.426", "graph_edges.csv", dict(tag="main_seed3", model="slice_only"), "macroF1", 1, 3),
      ("0.385", "official_baselines.csv", dict(model="whisperleak_official_lgbm", tag="main_seed3"), "macroF1", 1, 3),
      ("0.259", "official_baselines.csv", dict(model="deep_fingerprinting_official", tag="main_seed3"), "macroF1", 1, 3),
      ("0.92", "tracker_link_profiled.csv", dict(corpus="ft_gemini", method="joint_assignment"), "acc", 1, 2),
      ("0.86", "tracker_link_profiled.csv", dict(corpus="fw_autogen", method="joint_assignment"), "acc", 1, 2),
      ("0.66", "tracker_link_profiled.csv", dict(corpus="real_m8v3", method="joint_assignment"), "acc", 1, 2),
      ("0.65", "tracker_link_profiled.csv", dict(corpus="fw_langgraph", method="joint_assignment"), "acc", 1, 2),
      ("0.42", "tracker_link_cross.csv", dict(corpus="ft_gemini", method="joint_assignment"), "acc", 1, 2),
      ("0.55", "tracker_link_cross.csv", dict(corpus="real_m8v3", method="joint_assignment"), "acc", 1, 2),
      ("3.0", "tracker_size_profiled.csv", dict(corpus="ft_gemini", method="joint_assignment", regime=">=1KB"), "MedAPE_per_corpus_fit_odd_eps", 100, 1),
      ("5.2", "tracker_size_profiled.csv", dict(corpus="fw_autogen", method="joint_assignment", regime=">=1KB"), "MedAPE_per_corpus_fit_odd_eps", 100, 1),
      ("7.4", "tracker_size_profiled.csv", dict(corpus="real_m4v3", method="joint_assignment", regime=">=1KB"), "MedAPE_per_corpus_fit_odd_eps", 100, 1),
      ("17", "tracker_size_profiled.csv", dict(corpus="real_m8v3", method="joint_assignment", regime=">=1KB"), "MedAPE_per_corpus_fit_odd_eps", 100, 0),
      ("12", "tracker_size_profiled.csv", dict(corpus="fw_langgraph", method="joint_assignment", regime=">=1KB"), "MedAPE_per_corpus_fit_odd_eps", 100, 0),
      ("0.886", "cqr_percorpus.csv", dict(corpus="fw_langgraph", scheme="per_corpus", regime="all"), "PICP90", 1, 3),
      ("0.955", "cqr_percorpus.csv", dict(corpus="real_m8", scheme="per_corpus", regime="all"), "PICP90", 1, 3),
      ("0.90", "tool_id_real.csv", dict(corpus="fw_autogen"), "closed_acc", 1, 2),
      ("0.69", "tool_id_real.csv", dict(corpus="fw_langgraph"), "closed_acc", 1, 2),
      ("0.52", "tool_id_real.csv", dict(corpus="real_main"), "closed_acc", 1, 2),
      ("0.43", "defences.csv", dict(defence="def_pad64k"), "size_rho", 1, 2),
      ("78", "defences.csv", dict(defence="def_pad64k"), "bw_overhead_pct", 1, 0),
      ("0.93", "defences.csv", dict(defence="def_sse_batch"), "size_rho", 1, 2),
      ("0.61", "defences.csv", dict(defence="def_combined"), "edge_macroF1", 1, 2),
      ("1.000", "real_seg.csv", dict(tag="real_7b_colab2", method="burst_rule (real)", clock="wire"), "F1@100", 1, 3),
      ("5.1", "real_size.csv", dict(tag="real_7b_colab2", estimator="struct_prop1", regime=">=1KB"), "MedAPE", 100, 1),
      ("0.898", "real_seg_mux.csv", dict(tag="real_m8v3", method="burst_rule"), "F1_100", 1, 3),
      ("0.900", "real_seg_mux.csv", dict(tag="fw_autogen", method="burst_rule"), "F1_100", 1, 3),
      ("0.707", "real_edges.csv", dict(corpus="all_real", model="ours_full"), "macroF1", 1, 3),
      ("0.476", "real_edges.csv", dict(corpus="all_real", model="whisperleak_official"), "macroF1", 1, 3),
      ("0.604", "real_edges.csv", dict(corpus="all_real", model="ours_no_context_growth"), "macroF1", 1, 3),
      ("0.989", "real_seg.csv", dict(tag="ft_groq", method="burst_rule (real)", clock="wire"), "F1@100", 1, 3),
      ("0.97", "real_padding.csv", dict(corpus="real_main", pad_B=0), "size_rho", 1, 2),
      ("0.48", "real_padding.csv", dict(corpus="real_m8v3", pad_B=4096), "size_rho", 1, 2),
      ("0.33", "real_padding.csv", dict(corpus="real_m4v3", pad_B=16384), "size_rho", 1, 2),
      ("549", "real_padding.csv", dict(corpus="real_main", pad_B=65536), "bw_overhead_pct", 1, 0),
      ("528", "noise_floor.csv", dict(corpus="fw_langgraph"), "nontool_growth_med_B", 1, 0),
      ("309", "noise_floor.csv", dict(corpus="fw_autogen"), "nontool_growth_med_B", 1, 0),
      ("0.973", "real_seg_mux.csv", dict(tag="real_m4v3", method="burst_rule"), "F1_100", 1, 3)]
TX = open(f"{ROOT}/paper_tex/main.tex").read() + open(f"{ROOT}/paper_tex/sec_v3_real.tex").read()
cr = []
for txt, f, flt, col, sc, dp in CL:
    v = q(f, flt, col) * sc; ok = f"{v:.{dp}f}" == txt
    cr.append(dict(claim=txt, value=round(v, 4), file=f, filter=json.dumps(flt), column=col, matches_source=ok, present_in_tex=txt in TX))
C = pd.DataFrame(cr); C.to_csv(f"{RES}/manuscript_claims_audit.csv", index=False)
tot = int(pd.read_csv(f"{RES}/freetier_clean.csv").clean_turns.sum()); print("free-tier clean turns", tot, "in tex", "1,913" in TX and tot == 1913)
print("claims", len(C), "match source", int(C.matches_source.sum()), "present in tex", int(C.present_in_tex.sum()))
print(C[~C.matches_source | ~C.present_in_tex][["claim", "value", "file", "column"]].to_string())
print("numbers", len(A), "matched", int((A.source != "UNMATCHED").sum()), "unmatched", len(um))
print("undefined refs", sorted(refs - labels), "undefined cites", sorted(c.strip() for c in cites if c.strip() not in bibs), "v2 leftovers", v2)
print(um[["number", "context"]].to_string(max_colwidth=70))
