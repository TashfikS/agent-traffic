"""03 - Tool-result size recovery: generalized (non-monotonic) Proposition 1, explicit framing subtraction,
split-conformal quantile regression (CQR), sub-1 KB vs >=1 KB reporting.

Generalized Prop. 1 (per parent-linked turn k on one connection):
    b_k - b_{k-1} = phi(a_{k-1}) + sum_j psi(r_{k,j}) + c_m - delta_k,    delta_k >= 0, delta_k = 0 for append-only turns
phi(a) is estimated from the observable downstream stream with explicit framing removal:
    a_hat = D_{k-1} - n_{k-1} * (9 + beta_arch)        (9-B HTTP/2 DATA header per record; TLS 17 B already removed)
Break detector: one-sided CUSUM on z_k = -(y1_k)/sigma (y1 = structural tool-byte estimate; append-only => y1 >= 0),
threshold h calibrated to 1% false-alarm rate on append-only calibration turns.  On an alarmed turn the estimator
switches to the reset baseline  y = b_k - b_first - phi(a_{k-1}) - c_summ  (summarize), or abstains (sliding window).
Splits: train main_seed1 | calibration main_seed2 + chunk0 of h1_stream/state | test main_seed3, h1_stream c1, state c1;
non-monotonic: ctx_summ / ctx_window chunks 0-1 (calibrate c_summ, h) and chunks 2-3 (test).
Outputs: results_v3/size_results.csv, size_coverage.csv, size_breaks.csv, size_test_predictions.parquet, figures_v3/fig_size_*.pdf
"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled")
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from t2.data import *; from t2.turns import build_turns
MODE = os.environ.get("T2_BOUND", "frame")        # frame (log-free) | oracle
ALPHA = 0.10
PAPER = ["dreq_naive", "p_down", "p_ndown", "req", "dreq_ratio"]
FEAT = ["grow_frac", "shrink_frac", "shrink", "switched", "dreq", "a_hat", "p_ndown", "p_req", "req", "dreq_ratio", "p_dn_med", "p_dn_small", "gap", "ttft", "n_up", "y_struct", "y_level", "y_regime"]

def prep(tag, chunks=None, require_link=True):
    df = build_turns(tag, MODE)
    df = df[(~df.missed) & (~df.spurious)].copy()
    for conn, g in df.groupby("conn"):
        df.loc[g.index, "p_dn_med"] = g.dn_med.shift(1).values; df.loc[g.index, "p_dn_small"] = g.dn_small_frac.shift(1).values
        # hybrid baseline: previous turn on the connection unless the request shrank, then the context-growth
        # candidate (largest earlier request <= req_k + 64 B with a completed response); observer-only
        req, dn, nd, tl, t0 = g.req.values, g.down.values, g.n_down.values, g.t_last.values, g.t0.values
        br, bd, bn, sw = np.full(len(g), np.nan), np.full(len(g), np.nan), np.full(len(g), np.nan), np.zeros(len(g))
        for k in range(1, len(g)):
            j = k - 1
            if req[k] < req[j] - 64:
                ok = [i for i in range(max(0, k - 40), k) if req[i] <= req[k] + 64 and tl[i] <= t0[k] + 1e-4]
                if ok: j = max(ok, key=lambda i: req[i]); sw[k] = 1
                else: sw[k] = 2
            br[k], bd[k], bn[k] = req[j], dn[j], nd[j]
        df.loc[g.index, "b_req"] = br; df.loc[g.index, "b_down"] = bd; df.loc[g.index, "b_ndown"] = bn; df.loc[g.index, "switched"] = sw
        df.loc[g.index, "grow_frac"] = (g.req.diff() > 0).astype(float).expanding().mean().shift(1).values
    df["dreq_ratio"] = df.dreq / df.p_req.clip(lower=1)
    df["shrink"] = (df.req < df.p_req - 64).astype(float)                     # observable non-monotonic event
    df["shrink_frac"] = df.groupby("conn").shrink.transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
    df["dreq_naive"] = df.dreq                                                  # paper: previous-on-connection
    df["dreq"] = df.req - df.b_req; df["p_down"] = df.b_down; df["p_ndown"] = df.b_ndown
    df["a_hat"] = df.p_down - 9 * df.p_ndown
    if chunks is not None: df = df[df.chunk.isin(chunks)]
    df = df[(~df.mux) & df.p_req.notna() & df.b_req.notna()]
    df["tool"] = (df.gt_n_tools_added > 0) & (df.gt_tool_bytes_added > 0) & (df.gt_edge != "transport_retry")
    df["linked"] = df.p_turn == df.gt_parent
    if require_link and MODE == "oracle": df = df[df.linked]
    df["brk"] = df.get("gt_ctx_event", pd.Series("none", index=df.index)).fillna("none") != "none"
    df["conv_switch"] = df.p_turn != df.gt_parent
    return df

def fit_struct(d):
    X = d[["dreq", "a_hat", "p_ndown"]].values; y = d.gt_tool_bytes_added.values.astype(float)
    return HuberRegressor(epsilon=1.35, alpha=1e-6, max_iter=2000).fit(X / 1e3, y / 1e3)
def struct_pred(m, d): return m.predict(d[["dreq", "a_hat", "p_ndown"]].values / 1e3) * 1e3

def slog(x): x = np.asarray(x, float); return np.sign(x) * np.log1p(np.abs(x))
def Xof(d): return slog(d[FEAT].fillna(0).values)
def qmodel(a, X, y): return lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=500, learning_rate=.04, num_leaves=31, min_child_samples=15,
                                              subsample=.8, subsample_freq=1, verbose=-1, random_state=0).fit(X, y)
REG = [("all", lambda y: np.ones(len(y), bool)), ("<1KB", lambda y: y < 1e3), (">=1KB", lambda y: y >= 1e3), ("1-10KB", lambda y: (y >= 1e3) & (y < 1e4)), (">10KB", lambda y: y >= 1e4)]
def bins_of(p): return np.digitize(p, [1e3, 1e4])

def metrics(tag, est, y, p, lo=None, hi=None, extra=None):
    out = []
    for reg, f in REG:
        m = f(y)
        if m.sum() < 10: continue
        ae = np.abs(p[m] - y[m]); ape = ae / y[m]; ci = boot_ci(ape); cia = boot_ci(ae)
        r = dict(tag=tag, boundaries=MODE, estimator=est, regime=reg, n=int(m.sum()), MedAE_B=cia[0], MedAE_B_lo=cia[1], MedAE_B_hi=cia[2],
                 MedAPE=ci[0], MedAPE_lo=ci[1], MedAPE_hi=ci[2], p90_APE=float(np.quantile(ape, .9)), spearman=float(spearmanr(p[m], y[m]).correlation))
        if lo is not None:
            cov = (y[m] >= lo[m]) & (y[m] <= hi[m]); r.update(PICP90=float(cov.mean()), PICP90_lo=boot_ci(cov.astype(float), f=np.mean)[1],
                                                               PICP90_hi=boot_ci(cov.astype(float), f=np.mean)[2], width_med_ratio=float(np.median(hi[m] / lo[m])))
        if extra: r.update(extra)
        out.append(r)
    return out

if __name__ == "__main__":
    h0, s0 = prep("h1_stream", [0]), prep("state", [0])
    tr = pd.concat([prep("main_seed1"), h0[h0.gt_ep % 2 == 0], s0[s0.gt_ep % 2 == 0]]); tr = tr[tr.tool]
    cal = pd.concat([prep("main_seed2"), h0[h0.gt_ep % 2 == 1], s0[s0.gt_ep % 2 == 1]]); cal = cal[cal.tool]
    tests = {"main_seed3": prep("main_seed3"), "h1_stream": prep("h1_stream", [1]), "state": prep("state", [1])}
    print("n train/cal", len(tr), len(cal), {k: int(v.tool.sum()) for k, v in tests.items()}, flush=True)
    S = fit_struct(tr[tr.tag != "state"]); json.dump(dict(coef_dreq=S.coef_[0], coef_ahat=S.coef_[1], coef_ndown=S.coef_[2] * 1e3, intercept=S.intercept_ * 1e3),
                                  open(f"{RES}/size_struct_coef.json", "w"), indent=1)
    # delta-upload regime (server-side state: only new items are uploaded): level model y ~ req, a_hat, n_down
    GROW = float(np.median(pd.concat([tr, cal]).groupby("conn").grow_frac.last().dropna()))   # informational
    lvl_tr = tr[tr.gt_ep.notna() & tr.tag.eq("state")] if "tag" in tr else tr.iloc[:0]
    S2 = HuberRegressor(max_iter=2000).fit(np.c_[lvl_tr.req, lvl_tr.p_down - 9 * lvl_tr.p_ndown, lvl_tr.p_ndown] / 1e3, lvl_tr.gt_tool_bytes_added / 1e3)
    THR = 0.25   # observable regime switch: fraction of earlier turns on the connection whose request shrank
    def add_struct(d):
        d["y_struct"] = struct_pred(S, d); d["y_level"] = S2.predict(np.c_[d.req, d.a_hat, d.p_ndown] / 1e3) * 1e3
        d["y_regime"] = np.where(d.shrink_frac < THR, d.y_struct, d.y_level)
    for d in [tr, cal] + list(tests.values()): add_struct(d)
    y_tr = np.log(tr.gt_tool_bytes_added.values.astype(float)); X_tr = Xof(tr)
    # paper baselines (same 5 features, linear + raw LightGBM quantiles)
    lin = np.linalg.lstsq(np.c_[tr[PAPER].values, np.ones(len(tr))], tr.gt_tool_bytes_added.values, rcond=None)[0]
    pq = {a: qmodel(a, slog(tr[PAPER].values), y_tr) for a in (.05, .5, .95)}
    # per-regime quantile models (append-growth vs delta-upload); regime and ALPN are observer-visible
    def regime(d): return (d.shrink_frac.values >= THR).astype(int)
    def gkey(d): return regime(d) * 2 + (d.tag.values == "h1_stream").astype(int)
    rtr = regime(tr); QR = {g: {a: qmodel(a, X_tr[rtr == g], y_tr[rtr == g]) for a in (ALPHA / 2, .5, 1 - ALPHA / 2)} for g in (0, 1)}
    def qpred(d, a):
        X = Xof(d); rg = regime(d); out = np.zeros(len(d))
        for g in (0, 1):
            if (rg == g).any(): out[rg == g] = QR[g][a].predict(X[rg == g])
        return out
    Xc_ = cal; yc = np.log(cal.gt_tool_bytes_added.values.astype(float))
    lo_c, hi_c, md_c = qpred(cal, ALPHA / 2), qpred(cal, 1 - ALPHA / 2), qpred(cal, .5)
    E = np.maximum(lo_c - yc, yc - hi_c); n = len(E)
    def cq(e): return np.quantile(e, min(1, np.ceil((len(e) + 1) * (1 - ALPHA)) / len(e)))
    qhat = cq(E); gc = gkey(cal); qg = {g: (cq(E[gc == g]) if (gc == g).sum() > 30 else qhat) for g in range(4)}
    bc = bins_of(np.exp(md_c)); qb = {b_: (cq(E[bc == b_]) if (bc == b_).sum() > 30 else qhat) for b_ in range(3)}
    json.dump(dict(qhat=qhat, qhat_bins={int(k): float(v) for k, v in qb.items()}, qhat_groups={int(k): float(v) for k, v in qg.items()},
                   n_cal=n, n_cal_groups={int(g): int((gc == g).sum()) for g in range(4)}), open(f"{RES}/size_cqr_params.json", "w"), indent=1)
    def predict_all(d):
        md = np.exp(qpred(d, .5)); lo, hi = qpred(d, ALPHA / 2), qpred(d, 1 - ALPHA / 2)
        qv = np.array([qb[i] for i in bins_of(md)]); qgv = np.array([qg[i] for i in gkey(d)])
        return dict(paper_lin=np.c_[d[PAPER].values, np.ones(len(d))] @ lin, paper_q50=np.exp(pq[.5].predict(slog(d[PAPER].values))),
                    paper_lo=np.exp(pq[.05].predict(slog(d[PAPER].values))), paper_hi=np.exp(pq[.95].predict(slog(d[PAPER].values))),
                    struct=np.maximum(d.y_regime.values, 1.0), ours=md, raw_lo=np.exp(lo), raw_hi=np.exp(hi),
                    cqr_lo=np.exp(lo - qhat), cqr_hi=np.exp(hi + qhat), mcqr_lo=np.exp(lo - qv), mcqr_hi=np.exp(hi + qv),
                    tcqr_lo=np.exp(lo - qgv), tcqr_hi=np.exp(hi + qgv))
    rows, preds = [], []
    for tag, d0 in tests.items():
        d = d0[d0.tool].copy(); P = predict_all(d); y = d.gt_tool_bytes_added.values.astype(float)
        rows += metrics(tag, "paper_linear", y, P["paper_lin"])
        rows += metrics(tag, "paper_lgbm_q50 (raw 90% PI)", y, P["paper_q50"], P["paper_lo"], P["paper_hi"])
        rows += metrics(tag, "struct_prop1_framing", y, P["struct"])
        rows += metrics(tag, "ours_q50 (raw 90% PI)", y, P["ours"], P["raw_lo"], P["raw_hi"])
        rows += metrics(tag, "ours_q50 + CQR", y, P["ours"], P["cqr_lo"], P["cqr_hi"])
        rows += metrics(tag, "ours_q50 + Mondrian-CQR", y, P["ours"], P["mcqr_lo"], P["mcqr_hi"])
        rows += metrics(tag, "ours_q50 + transport-Mondrian-CQR", y, P["ours"], P["tcqr_lo"], P["tcqr_hi"])
        for k, v in P.items(): d[k] = v
        preds.append(d[["tag", "conn", "chunk", "gt_turn", "gt_arch", "gt_tool_bytes_added", "gt_n_tools_added", "gt_n_err_added"] + list(P)])
    # ---------------- non-monotonic contexts
    brk_rows = []
    sig = np.median(np.abs(cal.y_struct - cal.gt_tool_bytes_added)) * 1.4826 + 1.0
    def cusum(d, h, nu=0.0):
        """one-sided CUSUM on z = -(y_struct)/sigma with reset; an observable shrink (req_k < req_{k-1}-64) is an immediate alarm"""
        z = -(d.y_struct.values) / sig; g = 0.0; out = np.zeros(len(z), bool); conn = d.conn.values; sh = d.shrink.values > 0
        for i in range(len(z)):
            if i and conn[i] != conn[i - 1]: g = 0.0
            g = max(0.0, g + z[i] - nu)
            if g > h or sh[i]: out[i] = True; g = 0.0
        return out
    # null = parent-linked append-only turns (calibration labels); conversation switches are true resets, not false alarms
    calall = prep("main_seed2", require_link=False); add_struct(calall); null = calall[~calall.conv_switch]
    hs = np.linspace(0, 40, 801); far = np.array([cusum(null, h).mean() for h in hs]); h_star = float(hs[np.argmax(far <= 0.01)]) if (far <= 0.01).any() else float(hs[np.argmin(far)])
    for tag in ("ctx_summ", "ctx_window"):
        if not chunks_of(tag): continue
        a = prep(tag, [0, 1], require_link=False); b = prep(tag, [2, 3], require_link=False)
        for d in (a, b): add_struct(d)
        a["alarm"] = cusum(a, h_star); b["alarm"] = cusum(b, h_star)
        def reset_raw(d): return struct_pred(S, d.assign(dreq=d.req - d.first_req))
        m_a = (a.brk & a.tool).values
        c_summ = float(np.median(reset_raw(a)[m_a] - a.gt_tool_bytes_added.values[m_a])) if m_a.any() else 0.0
        ev = b.brk | b.conv_switch
        tp = int((b.alarm & ev).sum()); fp = int((b.alarm & ~ev).sum()); fn = int((~b.alarm & ev).sum()); rc_ctx = float((b.alarm & b.brk).sum() / max(b.brk.sum(), 1))
        bt = b[b.tool].copy(); y = bt.gt_tool_bytes_added.values.astype(float); P = predict_all(bt)
        reset = np.maximum(reset_raw(bt) - c_summ, 1.0)
        aware = P["ours"].copy(); lo, hi = P["mcqr_lo"].copy(), P["mcqr_hi"].copy(); al = bt.alarm.values
        if tag == "ctx_summ": aware[al] = reset[al]; lo[al] = np.maximum(reset[al] - 1500, 1); hi[al] = reset[al] + 1500
        keep = ~al if tag == "ctx_window" else np.ones(len(bt), bool)
        brk_rows.append(dict(tag=tag, h=h_star, c_summ=c_summ, n_turns=len(b), n_breaks=int(b.brk.sum()), det_precision=tp / max(tp + fp, 1), det_recall=tp / max(tp + fn, 1), det_recall_ctx_events=rc_ctx, n_conv_switch=int(b.conv_switch.sum()),
                             far_on_appendonly=float(far[np.searchsorted(hs, h_star)]), n_tool_turns=len(bt), n_tool_break_turns=int(bt.brk.sum()), abstain_rate=float((~keep).mean())))
        rows += metrics(tag, "naive (append-only model)", y, P["ours"], P["mcqr_lo"], P["mcqr_hi"])
        rows += metrics(tag, "break-aware", y[keep], aware[keep], lo[keep], hi[keep], dict(abstain=float((~keep).mean())))
        for nm, m in (("break turns only - naive", bt.brk.values), ("non-break turns - naive", ~bt.brk.values)):
            if m.sum() >= 10: rows += metrics(tag, nm, y[m], P["ours"][m], P["mcqr_lo"][m], P["mcqr_hi"][m])
        if tag == "ctx_summ" and (bt.brk.values & al).sum() >= 10:
            m = bt.brk.values & al; rows += metrics(tag, "break turns only - reset baseline", y[m], aware[m], lo[m], hi[m])
    R = pd.DataFrame(rows); R.to_csv(f"{RES}/size_results.csv", index=False); pd.DataFrame(brk_rows).to_csv(f"{RES}/size_breaks.csv", index=False)
    pd.concat(preds).to_parquet(f"{RES}/size_test_predictions.parquet")
    show = R[R.regime.isin(["all", "<1KB", ">=1KB"])][["tag", "estimator", "regime", "n", "MedAE_B", "MedAPE", "PICP90", "width_med_ratio"]]
    print(show.round(3).to_string()); print(pd.DataFrame(brk_rows).round(3).to_string())
