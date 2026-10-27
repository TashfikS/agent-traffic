"""Steps 1.4-1.6: size calibration, execution-graph reconstruction vs re-implemented baselines, defences.
usage: python analyze_v2.py build|size|graph|defense"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
sys.path.insert(0, os.path.dirname(__file__)); from common import *; from pipeline import *
torch.manual_seed(0); np.random.seed(0)
CACHE = os.environ.get("T2CACHE", f"{OUT}/cache"); os.makedirs(CACHE, exist_ok=True)
def feats(tag, mode):
    p = f"{CACHE}/{tag}_{mode}.pkl"
    if os.path.exists(p): return pd.read_pickle(p)
    df = build(tag, mode); df.to_pickle(p); return df

# ------------------------------------------------------------------ size calibration (1.4)
SZ = ["dreq", "prev_down", "prev_ndown", "req", "dreq_ratio"]
def size_eval(tag, mode="oracle"):
    df = feats(tag, mode); d = df[(~df.missed) & (df.gt_turn > 0) & (df.n_tools > 0) & (df.tool_bytes > 0) & (df.edge != "transport_retry")].copy()
    d = d[(d.pred_parent == d.gt_parent)] if mode == "oracle" else d
    d = d.dropna(subset=SZ + ["tool_bytes"]); tr = (d.chunk % 2 == 0).values
    if tr.sum() < 30 or (~tr).sum() < 30: return None, None
    X = d[SZ].values; y = d.tool_bytes.values.astype(float)
    lin = np.linalg.lstsq(np.c_[X[tr], np.ones(tr.sum())], y[tr], rcond=None)[0]; pl = np.c_[X[~tr], np.ones((~tr).sum())] @ lin
    Xl = np.c_[np.sign(X)*np.log1p(np.abs(X))]; yl = np.log(y)
    q = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=400, learning_rate=.05, num_leaves=31, min_child_samples=10, verbose=-1, random_state=0).fit(Xl[tr], yl[tr]) for a in (.05, .5, .95)}
    lo, md, hi = [np.exp(q[a].predict(Xl[~tr])) for a in (.05, .5, .95)]
    te = d[~tr].copy(); te["pred_lin"] = pl; te["pred_q50"] = md; te["q05"] = lo; te["q95"] = hi
    rows = []
    for reg, m in [("all", np.ones(len(te), bool)), ("<1KB", te.tool_bytes < 1e3), ("1-10KB", (te.tool_bytes >= 1e3) & (te.tool_bytes < 1e4)), (">10KB", te.tool_bytes >= 1e4), (">=1KB", te.tool_bytes >= 1e3)]:
        g = te[m]
        if len(g) < 10: continue
        for est in ("pred_lin", "pred_q50"):
            ape = np.abs(g[est] - g.tool_bytes)/g.tool_bytes
            ci = boot_ci(ape.values, f=np.median)
            rows.append(dict(tag=tag, boundaries=mode, regime=reg, estimator=est, n=len(g), median_APE=ci[0], median_APE_lo=ci[1], median_APE_hi=ci[2],
                             p90_APE=float(np.quantile(ape, .9)), MAE_bytes=float(np.abs(g[est]-g.tool_bytes).mean()),
                             spearman=float(spearmanr(g[est], g.tool_bytes).correlation),
                             PICP90=float(((g.tool_bytes >= g.q05) & (g.tool_bytes <= g.q95)).mean()) if est == "pred_q50" else np.nan))
    return rows, te

# ------------------------------------------------------------------ graph reconstruction (1.5)
class DF(torch.nn.Module):   # Deep-Fingerprinting-style 1D CNN on (signed log size, log dt) sequences
    def __init__(s, nc):
        super().__init__()
        s.f = torch.nn.Sequential(torch.nn.Conv1d(2, 32, 5, padding=2), torch.nn.ELU(), torch.nn.MaxPool1d(2), torch.nn.Conv1d(32, 64, 5, padding=2), torch.nn.ReLU(),
                                  torch.nn.MaxPool1d(2), torch.nn.Conv1d(64, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveMaxPool1d(1))
        s.h = torch.nn.Linear(64, nc)
    def forward(s, x): return s.h(s.f(x).squeeze(-1))
def seqmat(df):
    M = np.zeros((len(df), 2, 64), np.float32)
    for i, s in enumerate(df.seq.values):
        if isinstance(s, np.ndarray) and len(s): M[i, :, :len(s)] = s.T
    return M
def histmat(df): return np.vstack([h if isinstance(h, np.ndarray) else np.zeros(12) for h in df.hist.values])

def ece(P, y, nb=10):
    conf = P.max(1); pred = P.argmax(1); bins = np.linspace(0, 1, nb+1); e = 0
    for a, b in zip(bins[:-1], bins[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean()*abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e

def fit_predict(method, tr, te):
    lab = {e: i for i, e in enumerate(EDGE)}
    ytr = tr.edge.map(lab).values
    if method == "ours":
        cols = SLICE + CTX
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=31, verbose=-1, random_state=0).fit(tr[cols].fillna(0), ytr)
        P = m.predict_proba(te[cols].fillna(0))
    elif method == "agentprint_volume":
        Xtr = np.c_[histmat(tr), tr[["req", "down", "n_up", "n_down"]].fillna(0)]; Xte = np.c_[histmat(te), te[["req", "down", "n_up", "n_down"]].fillna(0)]
        m = RandomForestClassifier(300, random_state=0, n_jobs=1).fit(Xtr, ytr); P = m.predict_proba(Xte)
    elif method == "whisperleak_lgbm":
        Xtr = seqmat(tr).reshape(len(tr), -1); Xte = seqmat(te).reshape(len(te), -1)
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, verbose=-1, random_state=0).fit(Xtr, ytr); P = m.predict_proba(Xte)
    elif method == "deep_fingerprinting":
        net = DF(len(EDGE)); opt = torch.optim.Adam(net.parameters(), 1e-3); X = torch.tensor(seqmat(tr)); Y = torch.tensor(ytr)
        w = torch.tensor(1/np.maximum(np.bincount(ytr, minlength=len(EDGE)), 1), dtype=torch.float32); w = w/w.sum()*len(EDGE)
        for ep in range(15):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 256):
                b = perm[i:i+256]; loss = torch.nn.functional.cross_entropy(net(X[b]), Y[b], weight=w); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): P = torch.softmax(net(torch.tensor(seqmat(te))), 1).numpy()
    full = np.zeros((len(te), len(EDGE))); cls = np.unique(ytr)
    full[:, cls if P.shape[1] == len(cls) else np.arange(P.shape[1])] = P
    return full

def graph_eval(te, P, method):
    """Per-port normalised GED: missed/extra nodes + wrong parent + wrong edge type, / (|V|+|E|)."""
    te = te.copy(); te["pred_edge"] = [EDGE[i] for i in P.argmax(1)]
    rows = []
    for port, g in te.groupby("port"):
        g = g.sort_values("t0", na_position="last"); det = g[(~g.missed) & (g.gt_turn > 0)]
        V = (g.gt_turn > 0).sum(); E = V - (g.edge == "start").sum()
        miss = g.missed.sum(); extra = (g.gt_turn < 0).sum()
        if method == "ours":   # context-growth parent; spawned/fresh turns attach to latest completed turn on the port
            pp = det.pred_parent.values.copy(); prev = np.r_[-1, det.gt_turn.values[:-1]]
            fresh = det.fresh.values.astype(bool); pp[fresh] = prev[fresh]
        else: pp = np.r_[-1, det.gt_turn.values[:-1]]         # slice-only baselines: assume a sequential chain
        gp = det.gt_parent.fillna(-1).values
        gp_on_port = np.where(np.isin(gp, g.gt_turn.values), gp, -1)
        wrong_parent = (pp != gp_on_port).sum(); wrong_type = (det.pred_edge.values != det.edge.values).sum()
        rows.append(dict(method=method, port=port, V=V, E=E, ged=(miss + extra + wrong_parent + wrong_type)/max(V + E, 1),
                         parent_acc=float((pp == gp_on_port).mean()) if len(det) else np.nan))
    return pd.DataFrame(rows)

def score(te, P, method, setting):
    y = te.edge.values; pred = np.array([EDGE[i] for i in P.argmax(1)])
    pred = np.where(te.missed.values, "missed", pred)   # undetected turns count as errors
    keep = te.gt_turn.values > 0
    yk, pk, Pk = y[keep], pred[keep], P[keep]
    lab = {e: i for i, e in enumerate(EDGE)}; yi = np.array([lab[v] for v in yk])
    onehot = np.eye(len(EDGE))[yi]
    G = graph_eval(te, P, method)
    f1s = []
    r = np.random.default_rng(0); ports = te.port.unique()
    for _ in range(300):
        s = r.choice(ports, len(ports)); m = np.concatenate([np.where(te.port.values[keep] == p)[0] for p in s])
        f1s.append(f1_score(yk[m], pk[m], labels=EDGE, average="macro", zero_division=0))
    det = ~te.missed.values[keep]
    out = dict(method=method, setting=setting, n=int(keep.sum()), macroF1=f1_score(yk, pk, labels=EDGE, average="macro", zero_division=0),
               macroF1_lo=float(np.percentile(f1s, 2.5)), macroF1_hi=float(np.percentile(f1s, 97.5)),
               brier=float(((Pk[det] - onehot[det])**2).sum(1).mean()), ece=float(ece(Pk[det], yi[det])),
               GED=G.ged.mean(), GED_lo=boot_ci(G.ged.values)[1], GED_hi=boot_ci(G.ged.values)[2], parent_acc=G.parent_acc.mean())
    for e in EDGE: out[f"F1_{e}"] = f1_score(yk, pk, labels=[e], average="macro", zero_division=0)
    return out, confusion_matrix(yk, pk, labels=EDGE + ["missed"])

if __name__ == "__main__":
    step = sys.argv[1]
    if step == "build":
        for tag in sys.argv[2:]:
            for mode in ("oracle", "pred"): feats(tag, mode); print("built", tag, mode, flush=True)
    if step == "size":
        rows = []
        for tag in ["main_seed3", "main_seed1", "h1_stream", "state", "gzip", "nostream", "sweep_seq", "fixed5k"]:
            if not chunks_of(tag): continue
            for mode in ("oracle", "pred"):
                rr, te = size_eval(tag, mode)
                if rr: rows += rr; te.drop(columns=["seq", "hist"], errors="ignore").to_csv(f"{OUT}/size_pred_{tag}_{mode}.csv", index=False)
        pd.DataFrame(rows).to_csv(f"{OUT}/size_calibration.csv", index=False); print(pd.DataFrame(rows).query("regime in ['all','>=1KB','<1KB']").round(3).to_string())
    if step == "graph":
        res, cms = [], {}
        tr_o = pd.concat([feats("main_seed1", "oracle"), feats("main_seed2", "oracle")]); te_o = feats("main_seed3", "oracle")
        tr_p = pd.concat([feats("main_seed1", "pred"), feats("main_seed2", "pred")]); te_p = feats("main_seed3", "pred")
        trf = lambda d: d[(~d.missed) & (d.gt_turn > 0)]
        BASE = ["agentprint_volume", "whisperleak_lgbm", "deep_fingerprinting"]
        for meth in BASE:
            P = fit_predict(meth, trf(tr_o), te_o); s, cm = score(te_o, P, meth, "1_baseline_oracle"); res.append(s)
            P = fit_predict(meth, trf(tr_p), te_p); s, cm = score(te_p, P, meth, "2_baseline_logfree"); res.append(s); print(meth, "done", flush=True)
        P = fit_predict("ours", trf(tr_p), te_p); s, cm = score(te_p, P, "ours", "3_ours_logfree"); res.append(s); cms["ours_logfree"] = cm
        P = fit_predict("ours", trf(tr_o), te_o); s, cm = score(te_o, P, "ours", "4_ours_oracle"); res.append(s); cms["ours_oracle"] = cm
        # ablation: ours without context-growth features (slice-only LightGBM)
        cols_bak = list(CTX); CTX.clear()
        P = fit_predict("ours", trf(tr_p), te_p); s, _ = score(te_p, P, "ours", "ablation_no_context_logfree"); s["method"] = "ours_slice_only"; res.append(s)
        CTX.extend(cols_bak)
        # cross-framework: leave one architecture out
        allp = pd.concat([tr_p, te_p])
        for a in ["chat", "messages", "responses"]:
            for meth in ["ours", "whisperleak_lgbm"]:
                P = fit_predict(meth, trf(allp[allp.arch != a]), allp[allp.arch == a]); s, _ = score(allp[allp.arch == a], P, meth, f"crossframework_test_{a}"); res.append(s)
        # other transports / modes (log-free, models trained on main)
        for tag in ["h1_stream", "state", "gzip", "nostream", "subM1", "subM2", "subM4", "subM8"]:
            if not chunks_of(tag): continue
            d = feats(tag, "pred"); P = fit_predict("ours", trf(tr_p), d); s, _ = score(d, P, "ours", f"transfer_{tag}"); res.append(s)
        R_ = pd.DataFrame(res); R_.to_csv(f"{OUT}/graph_results.csv", index=False)
        for k, cm in cms.items(): pd.DataFrame(cm, index=EDGE + ["missed"], columns=EDGE + ["missed"]).to_csv(f"{OUT}/confusion_{k}.csv")
        # open-set tool recognition: single-tool turns, 3 of 12 tools held out
        d = pd.concat([tr_o, te_o]); d = d[(~d.missed) & (d.n_tools == 1) & (d.edge.isin(["sequential", "semantic_retry"])) & (d.pred_parent == d.gt_parent)].copy()
        d["tool"] = d.tools.astype(str)
        tools = sorted(d.tool.unique()); rr = np.random.default_rng(0); out = []
        for rep in range(5):
            unseen = list(rr.choice([t for t in tools if t != "subagent"], 3, replace=False)); known = d[~d.tool.isin(unseen)]
            trm = known.chunk % 2 == 0; Xc = ["dreq", "gap", "prev_down", "prev_ndown", "ttft"]
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, verbose=-1, random_state=0).fit(known[trm][Xc].fillna(0), known[trm].tool)
            test = pd.concat([known[~trm], d[d.tool.isin(unseen)]]); P = m.predict_proba(test[Xc].fillna(0))
            is_unk = test.tool.isin(unseen).values; score_unk = 1 - P.max(1)
            acc_known = (m.classes_[P.argmax(1)][~is_unk] == test.tool.values[~is_unk]).mean()
            out.append(dict(rep=rep, unseen="|".join(unseen), closed_set_acc=acc_known, open_set_AUROC=roc_auc_score(is_unk, score_unk)))
        pd.DataFrame(out).to_csv(f"{OUT}/openset_tools.csv", index=False)
        print(R_[["method", "setting", "macroF1", "macroF1_lo", "macroF1_hi", "GED", "parent_acc", "brier", "ece"]].round(3).to_string()); print(pd.DataFrame(out).round(3))
    if step == "defense":
        from segmenter import nms
        tr_p = pd.concat([feats("main_seed1", "pred"), feats("main_seed2", "pred")]); rows = []
        trf = lambda d: d[(~d.missed) & (d.gt_turn > 0)]
        def overhead(tag):
            gt, obs, _ = load(tag); return obs.plen.sum()/gt.req_bytes.add(gt.resp_bytes).sum(), ((gt.t_end - gt.t_send)/TS).mean()
        base_bw, base_lat = overhead("main_seed3")
        for tag in ["main_seed3", "state", "gzip", "def_pad4k", "def_pad16k", "def_pad64k", "def_jitter_batch", "def_combined"]:
            if not chunks_of(tag): continue
            d = feats(tag, "pred"); tr = d[d.chunk % 2 == 0]; te = d[d.chunk % 2 == 1]
            P = fit_predict("ours", trf(tr if tag != "main_seed3" else tr_p), te); s, _ = score(te, P, "ours", tag)
            rr, _ = size_eval(tag, "pred"); sp = [r for r in (rr or []) if r["regime"] == "all" and r["estimator"] == "pred_q50"]
            bw, lat = overhead(tag)
            rows.append(dict(defense=tag, GED=s["GED"], GED_lo=s["GED_lo"], GED_hi=s["GED_hi"], macroF1=s["macroF1"], size_spearman=sp[0]["spearman"] if sp else np.nan,
                             size_medAPE=sp[0]["median_APE"] if sp else np.nan, bytes_per_plain=bw, bandwidth_overhead_pct=100*(bw/base_bw - 1),
                             turn_latency_s=lat, latency_overhead_ms=1e3*(lat - base_lat)))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/defenses.csv", index=False)
