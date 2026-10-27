"""15 - Real-stack execution-graph edge typing and padding defences (Kaggle + Cloudflare, LangGraph, AutoGen).
Edge typing: LightGBM on (a) per-turn slice features, (b) + context growth relative to the profiled joint-tracked
predecessor (scripts/11, T2_SAVE_PRED), (c) context without growth; official Whisper-Leak LightGBM on the same burst
slices.  Train: even episodes of all corpora; test: odd episodes per corpus; undetected turns count as errors.
Padding: request and response byte totals per turn padded up to multiples of B (4/16/64 KB) on the real captures; the
Prop.-1 size model is refitted per condition (even episodes) and evaluated on odd episodes (Spearman rho, MedAPE >=1KB).
Outputs: results_v3/real_edges.csv, real_padding.csv, kaggle/df_real/data/*.npz (DF inputs)"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled"); os.environ.setdefault("T2_TS", "1"); os.environ.setdefault("T2_LABEL_WIN", "1.0")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("T2_RAW", f"{ROOT}/data/raw_real"); os.environ.setdefault("T2_PQ", f"{ROOT}/data/parquet_real")
import sys, importlib.util, json, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, ROOT); from t2.data import *; from t2.turns import build_turns
from sklearn.metrics import f1_score; from sklearn.linear_model import HuberRegressor; from scipy.stats import spearmanr
spec = importlib.util.spec_from_file_location("s07", f"{ROOT}/scripts/07_official_baselines.py"); S7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S7)
EDGE = ["start", "sequential", "parallel_fork_join", "transport_retry", "semantic_retry", "sub_agent"]
CORP = ["real_main", "real_m4v3", "real_m8v3", "fw_langgraph", "fw_autogen"]
SLICE = ["req", "down", "n_down", "ttft", "dn_med", "dn_small_frac", "n_up", "conc", "gap_conn", "first_on_conn"]
CTX = ["birth", "dreq", "res", "a_p", "dt_p", "p_children", "p_is_prev", "nx_dreq", "nx_dt", "sib_same_t"]
CTX_NOGROW = ["birth", "dt_p", "p_children", "p_is_prev", "nx_dt", "sib_same_t"]
def feats(t):
    d = pd.read_parquet(f"{RES}/tracker_pred/{t}_profiled.parquet").sort_values("t0").reset_index(drop=True)
    d["corpus"] = t; pj = d.pred_joint.values; n = len(d); i = np.arange(n)
    pc = np.full(n, -1); last = {}
    for j, c in enumerate(d.conn.values): pc[j] = last.get(c, -1); last[c] = j
    d["gap_conn"] = np.where(pc >= 0, d.t0.values - d.t_last.values[np.maximum(pc, 0)], -1); d["first_on_conn"] = (pc < 0).astype(int)
    ok = pj >= 0; P = np.maximum(pj, 0)
    d["birth"] = (~ok).astype(int); d["dreq"] = np.where(ok, d.req.values - d.req.values[P], np.nan); d["a_p"] = np.where(ok, d.a.values[P], np.nan)
    d["res"] = d.dreq - 0.009 * d.a_p; d["dt_p"] = np.where(ok, d.t0.values - d.t_last.values[P], np.nan); d["p_is_prev"] = (pj == pc).astype(int)
    cnt = np.zeros(n); seen = {}
    for j in range(n):
        if ok[j]: seen[pj[j]] = seen.get(pj[j], 0) + 1; cnt[j] = seen[pj[j]]
    d["p_children"] = cnt
    succ = np.full(n, -1); succ[pj[ok]] = i[ok]; s_ok = succ >= 0; S_ = np.maximum(succ, 0)
    d["nx_dreq"] = np.where(s_ok, d.req.values[S_] - d.req.values, np.nan); d["nx_dt"] = np.where(s_ok, d.t0.values[S_] - d.t_last.values, np.nan)
    d["sib_same_t"] = d.groupby(np.round(d.t0.values, 2)).t0.transform("size").values
    return d
def lgbm(): return lgb.LGBMClassifier(n_estimators=500, learning_rate=.05, num_leaves=31, class_weight="balanced", verbose=-1, random_state=0)
def mf(yt, yp): return f1_score(yt, yp, labels=EDGE, average="macro", zero_division=0), f1_score(yt, yp, labels=EDGE, average=None, zero_division=0)
if __name__ == "__main__":
    F = {t: feats(t) for t in CORP}; MISS = {t: (lambda T: T[T.missed & (T.gt_ep % 2 == 1)].gt_edge.values)(build_turns(t, "burst")) for t in CORP}
    tr = pd.concat([F[t][F[t].gt_ep % 2 == 0] for t in CORP]); rows = []
    for name, cols in (("slice_only", SLICE), ("ours_no_context_growth", SLICE + CTX_NOGROW), ("ours_full", SLICE + CTX)):
        m = lgbm().fit(tr[cols].fillna(-1), tr.gt_edge)
        for t in CORP:
            te = F[t][F[t].gt_ep % 2 == 1]; yp = list(m.predict(te[cols].fillna(-1))) + ["missed"] * len(MISS[t]); yt = list(te.gt_edge) + list(MISS[t])
            a, per = mf(yt, yp); rows.append(dict(corpus=t, model=name, n=len(yt), macroF1=a, **{f"F1_{e}": v for e, v in zip(EDGE, per)}))
        te = pd.concat([F[t][F[t].gt_ep % 2 == 1] for t in CORP]); ms = np.concatenate([MISS[t] for t in CORP])
        a, per = mf(list(te.gt_edge) + list(ms), list(m.predict(te[cols].fillna(-1))) + ["missed"] * len(ms)); rows.append(dict(corpus="all_real", model=name, n=len(te) + len(ms), macroF1=a, **{f"F1_{e}": v for e, v in zip(EDGE, per)}))
    # official Whisper-Leak LightGBM on burst slices
    sys.path.insert(0, S7.WL); from core.classifiers.lightgbm_classifier import LightGBMClassifier; import yaml
    cfg = yaml.safe_load(open(f"{S7.WL}/configs/lgbm.yaml"))
    SL = {}
    for t in CORP:
        s = S7.slices(t, "burst"); T = build_turns(t, "burst"); ep = T.set_index("rec").gt_ep
        s = s.copy(); s["ep"] = [np.nan] * len(s); SL[t] = s
    BT = {t: build_turns(t, "burst") for t in CORP}
    for t in CORP:   # attach episode by order (slices and detected turns share the same per-connection record order)
        T = BT[t][~BT[t].missed].sort_values(["conn", "rec"]); s = SL[t]
        s["ep"] = T.gt_ep.values[:len(s)] if len(T) == len(s) else np.nan
    W = pd.concat([SL[t] for t in CORP], ignore_index=True); W = W[W.ep.notna()]
    for c in ("time_diffs", "data_lengths"): W[c] = W[c].apply(list)
    trw = W[(W.ep % 2 == 0) & ~W.spurious].copy(); trw["target"] = trw.edge.map({e: i for i, e in enumerate(EDGE)}).astype(int)
    rng = np.random.default_rng(0); cs = trw.conn.unique(); vc = set(rng.choice(cs, max(1, int(.1 * len(cs))), replace=False))
    max_len = int(np.percentile(trw.data_lengths.apply(len), 95))
    clf = LightGBMClassifier({"max_len": max_len}, objective="multiclass", metric="multi_logloss", num_class=len(EDGE), **cfg["model"]["params"])
    clf.fit(trw[~trw.conn.isin(vc)], trw[trw.conn.isin(vc)], patience=cfg["training"]["patience"])
    allt, allp = [], []
    for t in CORP:
        te = W[(W.tag == t) & (W.ep % 2 == 1)]; yhat = list(np.array(EDGE)[clf.model.predict_proba(clf._prepare_features(te)).argmax(1)]) + ["missed"] * len(MISS[t])
        yt = list(te.edge.fillna("spurious")) + list(MISS[t]); allt += yt; allp += yhat
        a, per = mf(yt, yhat); rows.append(dict(corpus=t, model="whisperleak_official", n=len(yt), macroF1=a, **{f"F1_{e}": v for e, v in zip(EDGE, per)}))
    a, per = mf(allt, allp); rows.append(dict(corpus="all_real", model="whisperleak_official", n=len(allt), macroF1=a, **{f"F1_{e}": v for e, v in zip(EDGE, per)}))
    # DF inputs (run on Kaggle CPU)
    out = f"{ROOT}/kaggle/df_real/data"; os.makedirs(out, exist_ok=True)
    for name, df in (("train", W[(W.ep % 2 == 0) & ~W.spurious]), *[(t, W[(W.tag == t) & (W.ep % 2 == 1)]) for t in CORP]):
        X = np.zeros((len(df), 5000), np.int8)
        for j, L in enumerate(df.data_lengths.values): L = np.sign(np.asarray(L))[:5000]; X[j, :len(L)] = L
        np.savez_compressed(f"{out}/{name}.npz", X=X, y=df.edge.fillna("spurious").values, missed=MISS.get(name, np.array([])))
    E = pd.DataFrame(rows); E.to_csv(f"{RES}/real_edges.csv", index=False); print(E.pivot(index="corpus", columns="model", values="macroF1").round(3).to_string(), flush=True)
    # padding on real captures
    pr = []
    for t in ["real_main", "real_m4v3", "real_m8v3", "fw_autogen"]:
        d = F[t]; tool = (d.gt_n_tools_added > 0) & (d.gt_tool_bytes_added > 0) & (d.gt_edge != "transport_retry") & (d.pred_joint >= 0)
        for B in (0, 4096, 16384, 65536):
            pad = (lambda x: np.ceil(np.maximum(x, 1) / B) * B) if B else (lambda x: x)
            req, dn = pad(d.req.values), pad(d.down.values); P = np.maximum(d.pred_joint.values, 0)
            X = np.c_[req - req[P], dn[P] - 9 * d.n_down.values[P], d.n_down.values[P]][tool.values] / 1e3
            y = d.gt_tool_bytes_added.values[tool.values].astype(float); ev = (d.gt_ep.values[tool.values] % 2 == 0)
            H = HuberRegressor(max_iter=3000).fit(X[ev], y[ev] / 1e3); yh = np.maximum(H.predict(X[~ev]) * 1e3, 1); yt = y[~ev]; big = yt >= 1e3
            bw = (req.sum() + dn.sum()) / (d.req.values.sum() + d.down.values.sum()) - 1
            pr.append(dict(corpus=t, pad_B=B, n_test=int((~ev).sum()), bw_overhead_pct=100 * bw, size_rho=float(spearmanr(yh, yt)[0]), MedAPE_ge1KB=float(np.median(np.abs(yh[big] - yt[big]) / yt[big]))))
    PR = pd.DataFrame(pr); PR.to_csv(f"{RES}/real_padding.csv", index=False); print(PR.round(3).to_string())
