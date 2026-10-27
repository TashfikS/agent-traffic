"""11 - Joint conversation tracking from ciphertext (Aim 1).
Target of linking: the conversation predecessor of turn k (previous request of the same agent conversation), or BIRTH
(k starts a conversation).  This is the link Prop. 1 needs for size recovery.
  1. Candidates p for k: earlier detected turns in the scope (same connection, or the whole capture when the corpus
     opens a connection per request), last W=64 by time, with t_end(p) <= t0(k) (causality).
  2. Prop.-1 pruning: residual r = (b_k - b_p) - a_p  (a_p = downstream bytes minus 9 B/frame); candidates with
     r < -eps are infeasible unless flagged as a compaction transition (heavy penalty instead of removal).
  3. Pairwise LightGBM P(p = pred(k)) on size/Prop.-1/timing/transport features, leave-one-corpus-out training.
     Birth score from the same model family on per-turn features.
  4. Joint decoding: min-cost bipartite assignment (scipy linear_sum_assignment) - each k gets one predecessor or BIRTH,
     each p is used by at most one successor (conversations are chains), cost = -log P + lambda*violation.
Baselines: previous-on-connection, greedy recency, greedy argmax of the same pairwise model (no joint constraint).
Outputs: results_v3/tracker_link.csv, tracker_size.csv"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled"); os.environ.setdefault("T2_TS", "1"); os.environ.setdefault("T2_LABEL_WIN", "1.0")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("T2_RAW", f"{ROOT}/data/raw_real"); os.environ.setdefault("T2_PQ", f"{ROOT}/data/parquet_real")
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, ROOT); from t2.data import *; from t2.turns import build_turns
from scipy.optimize import linear_sum_assignment; from sklearn.linear_model import HuberRegressor
MODE = os.environ.get("T2_TRACK_MODE", "cross")    # cross: leave-one-corpus-out | profiled: same stack, even/odd episodes
W, EPS, LAM = 64, 256.0, 4.0
BETA = [0.0]
KAPPA = [0.0, 0.0]          # phi_hat(a) = KAPPA[0] * a + KAPPA[1]; set per test corpus from other corpora
CORPORA = [t for t in os.environ.get("T2_TRACK", "real_main,real_m4v3,real_m8v3,fw_langgraph,fw_autogen,ft_gemini,ft_groq,ft_cloudflare").split(",") if chunks_of(t)]
PER_REQ_CONN = lambda tag: tag.startswith("ft_")

def turns(tag):
    d = build_turns(tag, "burst"); d = d[~d.missed & ~d.spurious].sort_values("t0").reset_index(drop=True).copy()
    d["tag"] = tag; d["a"] = d.down - 9 * d.n_down
    # GT conversation predecessor
    d["pred"] = -1; last = {}
    for i, (cv, tt) in enumerate(zip(d.gt_conv.values, d.gt_turn.values)):
        d.at[i, "pred"] = last.get(cv, -1); last[cv] = i
    # active concurrency at t0
    s, e = d.t0.values, d.t_last.values; d["conc"] = [int(((s < x) & (e > x)).sum()) for x in s]
    return d

PF = ["dreq", "res", "res_rel", "growth_ratio", "dt", "p_ndown", "same_conn", "conc", "rank_t", "rank_res", "n_cand", "b_k", "b_p", "a_p", "p_ttft", "k_ttft", "infeasible", "p_has_succ_before"]
def pairs(d, scope_all):
    t0, tl, b, a, nd, cn, tf = (d[c].values for c in ("t0", "t_last", "req", "a", "n_down", "conn", "ttft"))
    rows = []
    for k in range(len(d)):
        if scope_all: idx = np.where(tl[:k] <= t0[k] + 1e-4)[0]
        else: idx = np.where((cn[:k] == cn[k]) & (tl[:k] <= t0[k] + 1e-4))[0]
        idx = idx[np.argsort(-tl[idx])][:W]
        if not len(idx): continue
        res = (b[k] - b[idx]) - a[idx]; order_res = np.argsort(np.argsort(np.where(res >= -EPS, np.abs(res), 1e12)))
        for j, p in enumerate(idx):
            rows.append((k, p, b[k] - b[p], res[j], res[j] / max(b[p], 1), b[k] / max(b[p] + a[p], 1), t0[k] - tl[p], nd[p], float(cn[p] == cn[k]), d.conc.values[k],
                         j, order_res[j], len(idx), b[k], b[p], a[p], tf[p], tf[k], float(res[j] < -EPS), 0.0))
    P = pd.DataFrame(rows, columns=["k", "p"] + PF); P["y"] = (d.pred.values[P.k.values] == P.p.values).astype(int); return P

BF = ["b_k", "k_ttft", "conc", "n_cand", "min_res_abs", "n_feasible", "rel_first"]
GROUP = lambda tag: "ft" if tag.startswith("ft_") else "kaggle"
def fit_kappa(D, corpora):
    """dreq_true - tool_bytes = kappa * a_pred + c  on GT predecessor pairs of the given (training) corpora (Huber)."""
    X, Y = [], []
    for u in corpora:
        d = D[u]; k = np.where(d.pred.values >= 0)[0]; p = d.pred.values[k]
        X.append(d.a.values[p]); Y.append(d.req.values[k] - d.req.values[p] - d.gt_tool_bytes_added.fillna(0).values[k])
    X, Y = np.concatenate(X), np.concatenate(Y); h = HuberRegressor(max_iter=3000).fit(X[:, None] / 1e3, Y / 1e3); return float(h.coef_[0]), float(h.intercept_ * 1e3)
def set_res(P):
    P["res"] = P.dreq - (KAPPA[0] * P.a_p + KAPPA[1]); P["res_rel"] = P.res / P.b_p.clip(lower=1); P["infeasible"] = (P.res < -EPS).astype(float)
    P["rank_res"] = P.assign(r=np.where(P.infeasible == 0, P.res.abs(), 1e12)).groupby("k").r.rank(method="first")
    return P

def birth_feats(d, P):
    g = P.assign(ra=np.where(P.infeasible == 0, P.res.abs(), np.nan)).groupby("k")
    f = pd.DataFrame(dict(n_cand=g.size(), min_res_abs=g.ra.min(), n_feasible=g.infeasible.apply(lambda x: (x == 0).sum()))).reindex(range(len(d)))
    f["b_k"] = d.req.values; f["k_ttft"] = d.ttft.values; f["conc"] = d.conc.values; f["n_cand"] = f.n_cand.fillna(0)
    f["rel_first"] = d.req.values / np.maximum(np.minimum.accumulate(d.req.values), 1)
    f["y"] = (d.pred.values == -1).astype(int); return f

def decode(d, P, pp, pb, joint=True):
    n = len(d); pred = np.full(n, -1)
    if not joint:
        best = P.assign(s=pp).sort_values("s", ascending=False).drop_duplicates("k")
        for k, p, s in zip(best.k, best.p, best.s):
            if s > pb[k]: pred[k] = p
        return pred
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    cost = -np.log(np.clip(pp, 1e-6, 1)) + LAM * P.infeasible.values + 1.0
    bc = -np.log(np.clip(pb, 1e-6, 1)) + BETA[0] + 1.0
    rows = np.r_[P.k.values, np.arange(n)]; cols = np.r_[P.p.values, n + np.arange(n)]; w = np.r_[cost, bc]
    M = coo_matrix((w, (rows, cols)), shape=(n, 2 * n)).tocsr()
    r_, c_ = np.arange(n), min_weight_full_bipartite_matching(M)[1]
    for k, j in zip(r_, c_):
        if j < n: pred[k] = j
    return pred

def lgbm(): return lgb.LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=63, min_child_samples=20, subsample=.8, subsample_freq=1, verbose=-1, random_state=0)

if __name__ == "__main__":
    D = {t: turns(t) for t in CORPORA}; PR = {t: pairs(D[t], PER_REQ_CONN(t)) for t in CORPORA}; BR = {t: birth_feats(D[t], PR[t]) for t in CORPORA}
    for t in CORPORA: print(t, len(D[t]), "pairs", len(PR[t]), "pred-in-cands", round(PR[t].groupby("k").y.max().sum() / max((D[t].pred >= 0).sum(), 1), 3), flush=True)
    # structural size model (Prop. 1) fitted on real_main even episodes with GT predecessors (no tracker involvement)
    m0 = D["real_main"]; tr = m0[(m0.gt_ep % 2 == 0) & (m0.pred >= 0) & (m0.gt_n_tools_added > 0) & (m0.gt_tool_bytes_added > 0)]
    pv = m0.loc[tr.pred.values]; Hs = HuberRegressor(max_iter=3000).fit(np.c_[tr.req.values - pv.req.values, pv.a.values, pv.n_down.values] / 1e3, tr.gt_tool_bytes_added / 1e3)
    link_rows, size_rows = [], []
    kap_rows = []
    for t in CORPORA:
        oth = [u for u in CORPORA if u != t]; grp = [u for u in oth if GROUP(u) == GROUP(t)] or oth
        KAPPA[:] = fit_kappa(D, grp); kap_rows.append(dict(corpus=t, kappa=KAPPA[0], c=KAPPA[1], fitted_on="|".join(grp)))
        tres = np.concatenate([(lambda d: (lambda k: (d.req.values[k] - d.req.values[d.pred.values[k]]) - KAPPA[0] * d.a.values[d.pred.values[k]] - KAPPA[1] - 0 * k)(np.where(d.pred.values >= 0)[0]))(D[u]) for u in grp])
        EPS = float(np.clip(-np.percentile(tres, 5), 64.0, 2048.0))   # robust: merged-burst outliers excluded
        for u in CORPORA: set_res(PR[u]); BR[u] = birth_feats(D[u], PR[u])
        Ptr = pd.concat([PR[u] for u in oth]); Btr = pd.concat([BR[u] for u in oth])
        if MODE == "profiled":
            # attacker profiles the same stack: train on pairs whose child is in an even episode, test on odd-episode children
            ev = (D[t].gt_ep.values % 2 == 0); Ptr = pd.concat([Ptr, PR[t][ev[PR[t].k.values]]] * 1); Btr = pd.concat([Btr, BR[t][ev]])
            Ptr = PR[t][ev[PR[t].k.values]]; Btr = BR[t][ev]
            ka, kc = fit_kappa({t: D[t].assign(pred=np.where(ev, D[t].pred.values, -1))}, [t]); KAPPA[:] = [ka, kc]; set_res(PR[t]); BR[t] = birth_feats(D[t], PR[t])
            Ptr = PR[t][ev[PR[t].k.values]]; Btr = BR[t][ev]
        mp = lgbm().fit(Ptr[PF], Ptr.y); mb = lgbm().fit(Btr[BF].fillna(-1), Btr.y)
        best = (-1, 0.0)
        for beta in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
            BETA[0] = beta; acc = []
            for u in ([t] if MODE == "profiled" else oth[:3]):
                pr = decode(D[u], PR[u], mp.predict_proba(PR[u][PF])[:, 1], mb.predict_proba(BR[u][BF].fillna(-1))[:, 1])
                msk = (D[u].gt_ep.values % 2 == 0) if MODE == "profiled" else np.ones(len(pr), bool); acc.append((pr == D[u].pred.values)[msk].mean())
            if np.mean(acc) > best[0]: best = (np.mean(acc), beta)
        BETA[0] = best[1]; kap_rows[-1].update(beta=best[1], eps=EPS)
        d, P = D[t], PR[t]; pp = mp.predict_proba(P[PF])[:, 1]; pb = mb.predict_proba(BR[t][BF].fillna(-1))[:, 1]
        cand = {}
        cand["prev_on_conn"] = np.array([-1] + [0] * 0) if False else None
        cn = d.conn.values; prevc = np.full(len(d), -1); lastc = {}
        for i, c in enumerate(cn): prevc[i] = lastc.get(c, -1); lastc[c] = i
        rec = np.full(len(d), -1); g = P[P.infeasible == 0].sort_values(["k", "rank_t"]).drop_duplicates("k"); rec[g.k.values] = g.p.values
        meth = {"prev_on_connection": prevc, "greedy_recent_feasible": rec, "greedy_pairwise": decode(d, P, pp, pb, joint=False), "joint_assignment": decode(d, P, pp, pb, joint=True)}
        if os.environ.get("T2_SAVE_PRED"):
            os.makedirs(f"{RES}/tracker_pred", exist_ok=True)
            d.assign(pred_joint=meth["joint_assignment"], pred_greedy=meth["greedy_recent_feasible"], pred_prev=meth["prev_on_connection"]).to_parquet(f"{RES}/tracker_pred/{t}_{MODE}.parquet")
        for name, pred in meth.items():
            ev_mask = (d.gt_ep.values % 2 == 1) if MODE == "profiled" else np.ones(len(d), bool)
            ok = (pred == d.pred.values)[ev_mask]; nonb = (d.pred.values >= 0)[ev_mask]
            link_rows.append(dict(corpus=t, method=name, n=len(d), acc=float(ok.mean()), acc_nonbirth=float(ok[nonb].mean()), birth_recall=float(ok[~nonb].mean()) if (~nonb).any() else np.nan))
            tt = (d.gt_n_tools_added > 0) & (d.gt_tool_bytes_added > 0) & (pred >= 0) & (d.gt_edge != "transport_retry")
            if tt.sum() < 20: continue
            x = d[tt]; pv = d.iloc[pred[tt.values]]; y = x.gt_tool_bytes_added.values.astype(float)
            X = np.c_[x.req.values - pv.req.values, pv.a.values, pv.n_down.values] / 1e3; yh = Hs.predict(X) * 1e3
            ev = (x.gt_ep % 2 == 1).values; Hc = HuberRegressor(max_iter=3000).fit(X[~ev], y[~ev] / 1e3) if (~ev).sum() > 30 else None
            for reg, mk in (("all", y > 0), ("<1KB", y < 1e3), (">=1KB", y >= 1e3)):
                if mk.sum() < 10: continue
                mk_ = mk & ev if MODE == "profiled" else mk
                r = dict(corpus=t, method=name, regime=reg, n=int(mk_.sum()), MedAPE_kaggle_fit=float(np.median(np.abs(yh[mk_] - y[mk_]) / y[mk_])))
                if Hc is not None and (mk & ev).sum() >= 10:
                    yc = Hc.predict(X[mk & ev]) * 1e3; r["MedAPE_per_corpus_fit_odd_eps"] = float(np.median(np.abs(yc - y[mk & ev]) / y[mk & ev])); r["n_odd"] = int((mk & ev).sum())
                size_rows.append(r)
        print(pd.DataFrame(link_rows[-4:]).round(3).to_string(), flush=True)
    pd.DataFrame(kap_rows).to_csv(f"{RES}/tracker_kappa_{MODE}.csv", index=False); print(pd.DataFrame(kap_rows).round(4).to_string())
    L = pd.DataFrame(link_rows); S = pd.DataFrame(size_rows); L.to_csv(f"{RES}/tracker_link_{MODE}.csv", index=False); S.to_csv(f"{RES}/tracker_size_{MODE}.csv", index=False)
    print(S[S.regime == ">=1KB"].round(3).to_string())
