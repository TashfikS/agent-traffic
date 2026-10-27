"""09 - Real-stack evaluation (T2_TS=1, wall-clock seconds).
Corpora (data/raw_real): real_* = vLLM-free HF Qwen2.5-1.5B on Kaggle T4 behind a Cloudflare quick tunnel (TLS 1.3 + h2
terminated at the Cloudflare edge, client on the public internet path); ft_* = live free-tier providers (Gemini, Groq,
Cloudflare Workers AI, OpenRouter :free) observed by an on-path relay.  GT = client request log.
A. boundaries: testbed frame rule vs real-stack burst rule (params fitted on real_m4 v1 only), F1 at +-50/100/250 ms.
B. tool-result size (log-free burst boundaries): structural Prop.-1 Huber + LightGBM q50, split CQR 90% intervals;
   train/calibrate on real_main (even/odd episodes), test on all other corpora (incl. live providers).
C. parent link: sequential vs hybrid context-growth rule.
Outputs: results_v3/real_seg.csv, real_size.csv, real_parent.csv"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled"); os.environ.setdefault("T2_TS", "1"); os.environ.setdefault("T2_LABEL_WIN", "1.0")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("T2_RAW", f"{ROOT}/data/raw_real"); os.environ.setdefault("T2_PQ", f"{ROOT}/data/parquet_real")
import sys, glob, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, ROOT); from t2.data import *; from t2.seg import frame_rule, burst_rule; from t2.turns import build_turns
from sklearn.linear_model import HuberRegressor
spec = importlib.util.spec_from_file_location("s02", f"{ROOT}/scripts/02_segmentation_multiplex.py"); S2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S2)
TAGS = sorted({os.path.basename(f)[3:-10] for f in glob.glob(f"{RAW}/gt_*_c0000.csv")})

def seg():
    rows = []
    for tag in TAGS:
        ds = S2.prep(tag)
        for name, fn in (("frame_rule (testbed)", frame_rule), ("burst_rule (real)", burst_rule)):
            for clk in ("app", "wire"):
                R = pd.DataFrame(S2.evaluate(name, ds, fn, clocks=[clk])); r = dict(tag=tag, method=name, clock=clk, n_conn=len(R), n_req=int(R.n_true.sum()),
                                                                                mux_conns=int(R.mux.sum()))
                for tol in S2.TOLS:
                    f = boot_ratio_ci(2 * R[f"tp{tol}"], 2 * R[f"tp{tol}"] + R[f"fp{tol}"] + R[f"fn{tol}"]); ms = int(tol * 1e3)
                    r.update({f"F1@{ms}": f[0], f"F1@{ms}_lo": f[1], f"F1@{ms}_hi": f[2]})
                rows.append(r)
        lag = np.concatenate([d["lag"] for d in ds]) * 1e3
        rows[-1].update(lag_p50_ms=float(np.percentile(lag, 50)), lag_p95_ms=float(np.percentile(lag, 95)))
    D = pd.DataFrame(rows); D.to_csv(f"{RES}/real_seg.csv", index=False); return D

def turns(tag):
    d = build_turns(tag, "burst"); d = d[~d.missed & ~d.spurious].copy()
    d["a_hat"] = d.p_down - 9 * d.p_ndown; d["ratio"] = d.dreq / d.p_req.clip(lower=1)
    d["tool"] = (d.gt_n_tools_added > 0) & (d.gt_tool_bytes_added > 0) & (d.gt_edge != "transport_retry")
    d["linked"] = d.p_turn == d.gt_parent
    # hybrid parent (observer-only): previous unless request shrank, then latest earlier request <= req+64 B
    d["h_dreq"] = d.dreq; d["h_turn"] = d.p_turn; d["h_down"] = d.p_down; d["h_ndown"] = d.p_ndown
    for _, g in d.groupby("conn"):
        rq, tt = g.req.values, g.gt_turn.values; dn, nd = g.down.values, g.n_down.values; ix = g.index
        for k in range(1, len(g)):
            if rq[k] < rq[k - 1] - 64:
                ok = [i for i in range(k - 1, -1, -1) if rq[i] <= rq[k] + 64]
                if ok: j = ok[0]; d.loc[ix[k], ["h_dreq", "h_turn", "h_down", "h_ndown"]] = [rq[k] - rq[j], tt[j], dn[j], nd[j]]
    if tag.startswith("ft_"):
        # live providers: each request rode a fresh tunnelled connection, so per-connection linking is undefined.
        # Observer-side cross-connection context-growth linker over the client's whole capture (time order):
        # parent = the most recent request whose response completed before t0 and whose size is <= req + 64 B and
        # maximal among the last 12 such candidates (largest consistent prefix).
        d = d.sort_values("t0"); t0, tl, rq, dn, nd, tt = (d[c].values for c in ("t0", "t_last", "req", "down", "n_down", "gt_turn"))
        hd, hdn, hnd, ht = (np.full(len(d), np.nan) for _ in range(4))
        for k in range(len(d)):
            cand = [i for i in range(k - 1, max(-1, k - 200), -1) if tl[i] <= t0[k] and rq[i] <= rq[k] + 64][:12]
            if cand: j = max(cand, key=lambda i: rq[i]); hd[k], hdn[k], hnd[k], ht[k] = rq[k] - rq[j], dn[j], nd[j], tt[j]
        d["h_dreq"], d["h_down"], d["h_ndown"], d["h_turn"] = hd, hdn, hnd, ht
        for c in ("p_req", "gap"): d[c] = d[c].fillna(0)
        if os.environ.get("T2_FT_ORACLE_LINK") == "1":   # upper bound: parent from the GT log, sizes still from ciphertext
            ix = d.set_index("gt_turn"); par = d.gt_parent.values
            ok = np.isin(par, ix.index); pr = ix.reindex(par)
            d["h_dreq"] = np.where(ok, d.req.values - pr.req.values, np.nan); d["h_down"] = pr.down.values; d["h_ndown"] = pr.n_down.values; d["h_turn"] = par
    if not tag.startswith("ft_") and os.environ.get("T2_LINK", "growth") == "growth":
        # multiplexed connections: the previous request on the connection is often another conversation's.  Link each
        # turn to the completed earlier request on the same connection that is the largest prefix-consistent one
        # (req_p <= req_k + 64 B), among the last 16 such candidates; dedicated connections reduce to the hybrid rule.
        for _, g in d.groupby("conn"):
            if not g.mux.iloc[0]: continue
            g = g.sort_values("t0"); t0, tl, rq, dn, nd, tt, ix = g.t0.values, g.t_last.values, g.req.values, g.down.values, g.n_down.values, g.gt_turn.values, g.index
            for k in range(len(g)):
                cand = [i for i in range(k - 1, -1, -1) if tl[i] <= t0[k] and rq[i] <= rq[k] + 64][:16]
                if LINKV == "recent" and cand: cand = [max(cand, key=lambda i: tl[i])]
                if cand:
                    # prefer the most recent candidate whose implied tool bytes are non-negative and smallest
                    res = [(rq[k] - rq[i]) - (dn[i] - 9 * nd[i]) * 0.0 for i in cand]
                    j = cand[int(np.argmax([rq[i] for i in cand]))]
                    d.loc[ix[k], ["h_dreq", "h_turn", "h_down", "h_ndown"]] = [rq[k] - rq[j], tt[j], dn[j], nd[j]]
                else: d.loc[ix[k], ["h_dreq", "h_turn", "h_down", "h_ndown"]] = [np.nan, np.nan, np.nan, np.nan]
    d["h_a_hat"] = d.h_down - 9 * d.h_ndown
    return d

LINKV = os.environ.get("T2_LINKV", "recent")
FEAT = ["h_dreq", "h_a_hat", "h_ndown", "req", "p_req", "ratio", "gap", "ttft", "n_up", "dn_med"]
def slog(x): x = np.asarray(x, float); return np.sign(x) * np.log1p(np.abs(x))

def size_and_parent():
    base = turns("real_main"); tr = base[(base.gt_ep % 2 == 0) & base.tool & base.h_dreq.notna()]; ca = base[(base.gt_ep % 2 == 1) & base.tool & base.h_dreq.notna()]
    H = HuberRegressor(max_iter=3000).fit(np.c_[tr.h_dreq, tr.h_a_hat, tr.h_ndown] / 1e3, tr.gt_tool_bytes_added / 1e3)
    X = lambda d: slog(np.c_[d[FEAT].fillna(0).values, H.predict(np.c_[d.h_dreq, d.h_a_hat, d.h_ndown] / 1e3) * 1e3])
    y = lambda d: np.log(d.gt_tool_bytes_added.values.astype(float))
    Q = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=400, learning_rate=.04, num_leaves=15, min_child_samples=10, verbose=-1, random_state=0).fit(X(tr), y(tr))
         for a in (.05, .5, .95)}
    E = np.maximum(Q[.05].predict(X(ca)) - y(ca), y(ca) - Q[.95].predict(X(ca))); qh = np.quantile(E, min(1, np.ceil((len(E) + 1) * .9) / len(E)))
    rows, prow = [], []
    for tag in TAGS:
        d = base[base.gt_ep % 2 == 1] if tag == "real_main" else turns(tag)
        prow.append(dict(tag=tag, n=len(d), sequential=float(d.linked.mean()), hybrid=float((d.h_turn == d.gt_parent).mean()),
                         mux_turns=int(d.mux.sum()), seq_mux=float(d[d.mux].linked.mean()) if d.mux.any() else np.nan, hybrid_mux=float((d[d.mux].h_turn == d[d.mux].gt_parent).mean()) if d.mux.any() else np.nan,
                         note="cross-connection linker" if tag.startswith("ft_") else "per-connection"))
        t = d[d.tool & d.h_dreq.notna()]
        if len(t) < 20: continue
        yt = t.gt_tool_bytes_added.values.astype(float); st = np.maximum(H.predict(np.c_[t.h_dreq, t.h_a_hat, t.h_ndown] / 1e3) * 1e3, 1)
        md = np.exp(Q[.5].predict(X(t))); lo = np.exp(Q[.05].predict(X(t)) - qh); hi = np.exp(Q[.95].predict(X(t)) + qh)
        rlo, rhi = np.exp(Q[.05].predict(X(t))), np.exp(Q[.95].predict(X(t)))
        for reg, m in (("all", yt > 0), ("<1KB", yt < 1e3), (">=1KB", yt >= 1e3)):
            if m.sum() < 10: continue
            for est, p, a, b in (("struct_prop1", st, None, None), ("lgbm_q50 raw PI", md, rlo, rhi), ("lgbm_q50 + CQR", md, lo, hi)):
                ape = np.abs(p[m] - yt[m]) / yt[m]; c = boot_ci(ape)
                rows.append(dict(tag=tag, estimator=est, regime=reg, n=int(m.sum()), MedAE_B=float(np.median(np.abs(p[m] - yt[m]))), MedAPE=c[0], MedAPE_lo=c[1], MedAPE_hi=c[2],
                                 PICP90=float(((yt[m] >= a[m]) & (yt[m] <= b[m])).mean()) if a is not None else np.nan))
    R = pd.DataFrame(rows); R.to_csv(f"{RES}/real_size{os.environ.get('T2_OUT_SUFFIX', '')}.csv", index=False); P = pd.DataFrame(prow); P.to_csv(f"{RES}/real_parent.csv", index=False); return R, P

if __name__ == "__main__":
    print(TAGS)
    D = seg(); print(D[["tag", "method", "clock", "n_conn", "n_req", "F1@50", "F1@100", "F1@100_lo", "F1@100_hi", "F1@250"]].round(3).to_string(), flush=True)
    R, P = size_and_parent(); print(R.round(3).to_string()); print(P.round(3).to_string())
