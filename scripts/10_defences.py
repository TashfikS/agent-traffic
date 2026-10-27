"""10 - Defence sweep on the controlled testbed (regenerated v3 corpus, 8 conditions x 150 episodes).
Attacker is retrained per condition (chunks 0-1 train, chunk 2 test), log-free frame boundaries.
Metrics: bandwidth overhead = ciphertext bytes per GT turn vs def_none; boundary F1@100 ms (wire);
edge-type macro-F1 (undetected turns count as errors); tool-size Spearman rho and MedAPE (>=1 KB).
Outputs: results_v3/defences.csv, figures_v3/fig_defences.pdf"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled")
import sys, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, ROOT)
from t2.data import *; from t2.turns import build_turns; from t2.seg import frame_rule
from scipy.stats import spearmanr; from sklearn.metrics import f1_score
spec = importlib.util.spec_from_file_location("s02", f"{ROOT}/scripts/02_segmentation_multiplex.py"); S2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S2)
EDGE = ["start", "sequential", "parallel_fork_join", "transport_retry", "semantic_retry", "sub_agent"]
TAGS = ["def_none", "def_state", "def_gzip", "def_pad4k", "def_pad16k", "def_pad64k", "def_jitter_batch", "def_sse_batch", "def_combined"]
F = ["req", "dreq", "down", "n_down", "gap", "ttft", "n_same_t", "p_req", "p_down", "p_ndown", "dn_small_frac", "dn_med", "n_up", "pos", "first_req", "max_prev_req"]
def nx(d):
    d = d.sort_values(["conn", "t0"]).copy(); g = d.groupby("conn")
    for c in ("req", "down", "gap"): d["nx_" + c] = g[c].shift(-1)
    d["ratio"] = d.dreq / d.p_req.clip(lower=1); return d
FF = F + ["nx_req", "nx_down", "nx_gap", "ratio"]
rows = []
for tag in TAGS:
    gt, obs = load(tag); byt = obs.groupby("chunk").plen.sum() + 17 * obs.groupby("chunk").size()
    bpt = float((byt / gt.groupby("chunk").size()).mean())
    ds = [d for d in S2.prep(tag) if d["chunk"] == 2] if "chunk" in S2.prep.__code__.co_varnames else None
    T = build_turns(tag, "frame"); det = nx(T[~T.missed & ~T.spurious]); tr, te = det[det.chunk < 2], det[det.chunk == 2]
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=31, verbose=-1, random_state=0, class_weight="balanced").fit(tr[FF].fillna(-1), tr.gt_edge)
    yp = list(clf.predict(te[FF].fillna(-1))); yt = list(te.gt_edge)
    ms = T[T.missed & (T.chunk == 2)].gt_edge.tolist(); yt += ms; yp += ["missed"] * len(ms)
    mf = f1_score(yt, yp, labels=EDGE, average="macro", zero_division=0)
    tt = lambda d: d[(d.gt_n_tools_added > 0) & (d.gt_tool_bytes_added > 0) & (d.gt_edge != "transport_retry")]
    a, b = tt(tr), tt(te)
    reg = lgb.LGBMRegressor(n_estimators=400, learning_rate=.05, num_leaves=31, verbose=-1, random_state=0).fit(a[FF].fillna(-1), np.log(a.gt_tool_bytes_added))
    p = np.exp(reg.predict(b[FF].fillna(-1))); y = b.gt_tool_bytes_added.values.astype(float); big = y >= 1e3
    allds = S2.prep(tag); te_ds = [d for d in allds if d["conn"] // 100000 == 2]
    f1b = S2.f1_of(S2.evaluate("f", te_ds, frame_rule, clocks=["wire"]), 0.1)
    rows.append(dict(defence=tag, bytes_per_turn=bpt, n_test_turns=len(yt), boundary_F1_100=f1b, edge_macroF1=mf,
                     size_rho=float(spearmanr(p, y)[0]), size_MedAPE_ge1KB=float(np.median(np.abs(p[big] - y[big]) / y[big]))))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); base = R.set_index("defence").bytes_per_turn["def_none"]; R["bw_overhead_pct"] = 100 * (R.bytes_per_turn / base - 1)
R.to_csv(f"{RES}/defences.csv", index=False); print(R.round(3).to_string())
