"""04 - Execution-graph reconstruction (parent linking + edge typing) and open-set tool identification.

Parent linkers (per connection, detected log-free turns, time order):
  sequential      : parent = previous detected turn on the connection (baseline that scored 0.92)
  paper_ctxgrowth : paper linker - earlier turn with the largest request <= req_k + 64 B and completed response
  hybrid_rule     : keep k-1 unless its conservation residual is inconsistent (req_k < req_{k-1} + phi(a_{k-1}) - m);
                    then the completed candidate with the smallest non-negative residual; fresh contexts (no consistent
                    candidate) attach to the most recently completed turn (spawn point)
  hybrid_map      : constrained MAP assignment - pairwise LightGBM likelihood over (candidate, child) features
                    [recency/sequential prior, conservation residual, request ratio, gap, concurrency, out-degree so far,
                    fresh-context flag], greedy time-ordered decode with the DAG constraint (parent completed before child)
Edge typing: LightGBM over slice + MAP-parent context-growth + bidirectional (next-turn) features; ablation without
context features; failfast corpus (tracebacks return in 2-20 ms) tests the execution-latency retry cue.
Tool ID: single-tool, non-failed turns; closed-set acc + unknown-tool AUROC (3 of 12 held out, 5 reps) with
max-softmax, energy and class-conditional Mahalanobis scores; GT-feature upper bound quantifies the corpus ceiling.
Outputs: results_v3/graph_parent.csv, graph_edges.csv, graph_confusion_*.csv, tool_openset.csv, tool_corpus_ceiling.csv
"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled")
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from t2.data import *; from t2.turns import build_turns
EDGE = ["start", "sequential", "parallel_fork_join", "transport_retry", "semantic_retry", "sub_agent"]
MODE = os.environ.get("T2_BOUND", "frame")

def load_turns(tag, chunks=None):
    d = build_turns(tag, MODE)
    if chunks is not None: d = d[d.chunk.isin(chunks)]
    d = d.copy(); d["is_det"] = ~d.missed; return d

# ------------------------------------------------------------ structural phi (fitted once on main_seed1 dedicated turns)
def fit_phi(d):
    d = d[d.is_det & ~d.spurious & ~d.mux & d.p_req.notna() & (d.gt_n_tools_added > 0) & (d.gt_edge != "transport_retry") & (d.p_turn == d.gt_parent)]
    a = d.p_down - 9 * d.p_ndown
    return HuberRegressor(max_iter=2000).fit(np.c_[d.dreq, a, d.p_ndown] / 1e3, d.gt_tool_bytes_added / 1e3)
def resid(S, req_k, req_p, down_p, ndown_p):
    """structural tool-byte estimate if p were the parent (negative => k cannot extend p)."""
    X = np.c_[req_k - req_p, down_p - 9 * ndown_p, ndown_p] / 1e3; return S.predict(X) * 1e3

# ------------------------------------------------------------ pair features
PF = ["rank", "is_prev", "res", "log_ratio", "gap", "p_children", "p_childless", "fresh", "conc", "n_same_t", "p_down_small", "dt_t0", "same_req", "k_req", "p_req_"]
def conn_pairs(g, S, max_back=40):
    """g: detected turns of one connection sorted by t0. Yields per child k the candidate feature matrix (children counts filled at decode)."""
    t0 = g.t0.values; tl = g.t_last.values; req = g.req.values; dn = g.down.values; nd = g.n_down.values; ns = g.n_same_t.values
    out = []
    for k in range(len(g)):
        cand = [p for p in range(max(0, k - max_back), k) if tl[p] <= t0[k] + 1e-4]
        if not cand: out.append((np.array([], int), None)); continue
        cand = np.array(cand); r = resid(S, req[k], req[cand], dn[cand], nd[cand])
        fresh = float((r < -256).all()); conc = float(((t0 < t0[k]) & (tl > t0[k])).sum())
        F = np.c_[np.arange(len(cand))[::-1], cand == k - 1, np.sign(r) * np.log1p(np.abs(r)), np.log(req[k] / np.maximum(req[cand], 1)),
                  np.log1p(np.maximum(t0[k] - tl[cand], 0) * 1e4), np.zeros(len(cand)), np.zeros(len(cand)), np.full(len(cand), fresh),
                  np.full(len(cand), conc), np.full(len(cand), ns[k]), dn[cand] < 300, np.log1p((t0[k] - t0[cand]) * 1e4),
                  np.abs(req[k] - req[cand]) < 32, np.full(len(cand), np.log1p(req[k])), np.log1p(req[cand])]
        out.append((cand, F.astype(np.float32)))
    return out

def gt_parent_pos(g):
    pos = {t: i for i, t in enumerate(g.gt_turn.values)}; return np.array([pos.get(p, -1) for p in g.gt_parent.values])

def decode(g, pairs, mode, model=None, S=None, margin=256, tau_none=0.05):
    n = len(g); par = np.full(n, -1); ch = np.zeros(n); req = g.req.values; tl = g.t_last.values
    for k, (cand, F) in enumerate(pairs):
        if not len(cand): continue
        if mode == "sequential": par[k] = k - 1; ch[k - 1] += 1; continue
        if mode == "paper_ctxgrowth":
            ok = cand[req[cand] <= req[k] + 64]; par[k] = ok[np.argmax(req[ok])] if len(ok) else -1
            if par[k] >= 0: ch[par[k]] += 1
            continue
        F = F.copy(); F[:, PF.index("p_children")] = ch[cand]; F[:, PF.index("p_childless")] = ch[cand] == 0
        r = np.sign(F[:, PF.index("res")]) * np.expm1(np.abs(F[:, PF.index("res")]))
        if mode == "hybrid_rule":
            if cand[-1] == k - 1 and r[-1] >= -margin: p = k - 1
            else:
                okm = r >= -margin
                p = cand[okm][np.argmin(r[okm])] if okm.any() else cand[np.argmax(tl[cand])]
        else:
            pr = model.predict_proba(F)[:, 1]; p = cand[np.argmax(pr)] if pr.max() >= tau_none else -1
        par[k] = p
        if p >= 0: ch[p] += 1
    return par

def train_pair_model(G, S):
    X, Y = [], []
    for g in G:
        gp = gt_parent_pos(g); pairs = conn_pairs(g, S); ch = np.zeros(len(g))
        for k, (cand, F) in enumerate(pairs):          # teacher forcing: children counts from GT
            if not len(cand): 
                if gp[k] >= 0: ch[gp[k]] += 1
                continue
            F = F.copy(); F[:, PF.index("p_children")] = ch[cand]; F[:, PF.index("p_childless")] = ch[cand] == 0
            X.append(F); Y.append(cand == gp[k])
            if gp[k] >= 0: ch[gp[k]] += 1
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=31, verbose=-1, random_state=0).fit(np.vstack(X), np.concatenate(Y))

def conns(d): return [g.sort_values("t0").reset_index(drop=True) for _, g in d[d.is_det & ~d.spurious].groupby("conn") if len(g)]
def conns_all(d): return [g.sort_values("t0").reset_index(drop=True) for _, g in d[d.is_det].groupby("conn") if len(g)]

def link_eval(tag, G, S, model):
    rows = []; out = {}
    for meth in ("sequential", "paper_ctxgrowth", "hybrid_rule", "hybrid_map"):
        acc_n, acc_d, mux = [], [], []
        for gi, g in enumerate(G):
            par = decode(g, conn_pairs(g, S), meth, model, S); gp = gt_parent_pos(g)
            acc_n.append((par == gp).sum()); acc_d.append(len(g)); mux.append(bool(g.mux.iloc[0]))
            out.setdefault(meth, []).append(par)
        acc_n, acc_d, mux = map(np.array, (acc_n, acc_d, mux))
        for sel, nm in ((np.ones(len(mux), bool), "all"), (mux, "multiplexed"), (~mux, "dedicated")):
            if sel.sum(): a = boot_ratio_ci(acc_n[sel], acc_d[sel]); rows.append(dict(tag=tag, linker=meth, conns=nm, n_turns=int(acc_d[sel].sum()), parent_acc=a[0], lo=a[1], hi=a[2]))
    return rows, out

# ------------------------------------------------------------ edge typing
SLICE = ["req", "n_up", "down", "n_down", "ttft", "dn_small_frac", "dn_med", "n_same_t"]
CTXF = ["m_dreq", "m_res", "m_logratio", "m_gap", "m_pdown", "m_pndown", "m_pdown_small", "m_child_rank", "m_outdeg", "m_fresh", "conc", "first_on_conn",
        "m_same_req", "nx_gap", "nx_res", "nx_dreq", "seq_gap", "seq_dreq", "p_outdeg_sib"]
def edge_features(g, par, S):
    g = g.copy(); n = len(g); req, dn, nd, t0, tl = g.req.values, g.down.values, g.n_down.values, g.t0.values, g.t_last.values
    p = par; has = p >= 0; pp = np.where(has, p, 0)
    r = np.where(has, resid(S, req, req[pp], dn[pp], nd[pp]), 0)
    outdeg = np.bincount(p[has], minlength=n) if has.any() else np.zeros(n)
    rank = np.zeros(n)
    seen = {}
    for k in range(n):
        if has[k]: seen[p[k]] = seen.get(p[k], 0) + 1; rank[k] = seen[p[k]]
    child = {}
    for k in range(n):
        if has[k] and p[k] not in child: child[p[k]] = k
    nx = np.array([child.get(k, -1) for k in range(n)])
    g["m_dreq"] = np.where(has, req - req[pp], 0); g["m_res"] = r; g["m_logratio"] = np.where(has, np.log(req / np.maximum(req[pp], 1)), 0)
    g["m_gap"] = np.where(has, np.log1p(np.maximum(t0 - tl[pp], 0) * 1e4), -1); g["m_pdown"] = np.where(has, dn[pp], -1); g["m_pndown"] = np.where(has, nd[pp], -1)
    g["m_pdown_small"] = np.where(has, dn[pp] < 300, 0); g["m_child_rank"] = rank; g["m_outdeg"] = np.where(has, outdeg[pp], 0); g["p_outdeg_sib"] = outdeg
    g["m_fresh"] = (~has) | (r < -256); g["conc"] = [((t0 < t0[k]) & (tl > t0[k])).sum() for k in range(n)]; g["first_on_conn"] = np.arange(n) == 0
    g["m_same_req"] = np.where(has, np.abs(req - req[pp]) < 32, 0)
    nxx = np.where(nx >= 0, nx, 0); g["nx_gap"] = np.where(nx >= 0, np.log1p(np.maximum(t0[nxx] - tl, 0) * 1e4), -1)
    g["nx_res"] = np.where(nx >= 0, resid(S, req[nxx], req, dn, nd), 0); g["nx_dreq"] = np.where(nx >= 0, req[nxx] - req, 0)
    g["seq_gap"] = np.r_[-1, np.log1p(np.maximum(t0[1:] - tl[:-1], 0) * 1e4)]; g["seq_dreq"] = np.r_[0, np.diff(req)]
    return g

def edge_table(d, S, model):
    rows = []
    for g in conns_all(d):
        det = g[~g.spurious].reset_index(drop=True) if False else g
        par = decode(det, conn_pairs(det, S), "hybrid_map", model, S); rows.append(edge_features(det, par, S))
    return pd.concat(rows, ignore_index=True)

def edge_eval(name, tr, te, cols, tag):
    lab = {e: i for i, e in enumerate(EDGE)}; trk = tr[~tr.spurious & tr.gt_edge.isin(EDGE)]
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=.05, num_leaves=31, class_weight="balanced", verbose=-1, random_state=0)
    m.fit(trk[cols].astype(float).fillna(0), trk.gt_edge.map(lab))
    pred = np.array(EDGE)[m.predict(te[cols].astype(float).fillna(0))]
    y = te.gt_edge.fillna("spurious").values; ok = te.gt_edge.isin(EDGE).values
    # undetected GT turns count as errors (paper protocol)
    yk = np.r_[y[ok], te_missed_labels(tag)]; pk = np.r_[pred[ok], np.array(["missed"] * len(te_missed_labels(tag)))]
    rr = np.random.default_rng(0); conn_k = np.r_[te.conn.values[ok], MISSED_CONN.get(tag, np.zeros(0))]; uc = np.unique(conn_k)
    idx = {c: np.where(conn_k == c)[0] for c in uc}; bs = []
    for _ in range(300):
        s = np.concatenate([idx[c] for c in rr.choice(uc, len(uc))]); bs.append(f1_score(yk[s], pk[s], labels=EDGE, average="macro", zero_division=0))
    r = dict(tag=tag, model=name, n=len(yk), macroF1=f1_score(yk, pk, labels=EDGE, average="macro", zero_division=0), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))
    for e in EDGE: r[f"F1_{e}"] = f1_score(yk, pk, labels=[e], average="macro", zero_division=0)
    return r, confusion_matrix(yk, pk, labels=EDGE + ["missed"]), m
MISSED = {}; MISSED_CONN = {}
def te_missed_labels(tag): return MISSED.get(tag, np.array([], dtype=object))

# ------------------------------------------------------------ open-set tool ID
TF = ["y_hat", "gap", "a_prev", "ttft", "p_ndown", "dreq", "dn_med", "prev_y_hat", "prev_gap"]
def tool_table(d, S):
    d = d[d.is_det & ~d.spurious & ~d.mux & d.p_req.notna()].copy()
    d["y_hat"] = np.log1p(np.maximum(resid(S, d.req.values, d.p_req.values, d.p_down.values, d.p_ndown.values), 1))
    d["gap"] = np.log1p(np.maximum(d.gap, 0) * 1e4); d["a_prev"] = np.log1p(np.maximum(d.p_down - 9 * d.p_ndown, 0))
    for c in ("y_hat", "gap"): d[f"prev_{c}"] = d.groupby("conn")[c].shift(1)
    d = d[(d.gt_n_tools_added == 1) & (d.gt_n_err_added == 0) & (d.gt_tools_added != "subagent") & (d.gt_edge.isin(["sequential", "semantic_retry"]))]
    d["tool"] = d.gt_tools_added.astype(str); d["prev_tool"] = d.groupby("conn").tool.shift(1).fillna("none"); return d

def openset(d, cols, tag, reps=5, markov=False):
    tools = sorted(d.tool.unique()); rr = np.random.default_rng(0); out = []
    for rep in range(reps):
        unseen = list(rr.choice(tools, 3, replace=False)); known = d[~d.tool.isin(unseen)]; trm = (known.chunk % 2 == 0).values
        Xtr = known[trm][cols].fillna(-1).values; ytr = known[trm].tool.values
        test = pd.concat([known[~trm], d[d.tool.isin(unseen)]]); Xte = test[cols].fillna(-1).values; is_unk = test.tool.isin(unseen).values
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15, verbose=-1, random_state=0).fit(Xtr, ytr)
        P = m.predict_proba(Xte); Z = m.predict_proba(Xte, raw_score=True)
        if markov:   # transition prior P(tool_k | tool_{k-1}) estimated on training turns (Laplace)
            cls = list(m.classes_); T = pd.crosstab(known[trm].prev_tool, known[trm].tool).reindex(columns=cls, fill_value=0) + 1
            T = T.div(T.sum(1), axis=0); prior = np.vstack([T.loc[p].values if p in T.index else np.full(len(cls), 1 / len(cls)) for p in test.prev_tool])
            P = P * prior; P = P / P.sum(1, keepdims=True); Z = np.log(np.maximum(P, 1e-12))
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xs, Xts = (Xtr - mu) / sd, (Xte - mu) / sd
        cls = np.unique(ytr); means = np.vstack([Xs[ytr == c].mean(0) for c in cls]); cov = np.cov((Xs - means[np.searchsorted(cls, ytr)]).T) + 1e-3 * np.eye(Xs.shape[1])
        ic = np.linalg.inv(cov); maha = np.min([np.einsum("ij,jk,ik->i", Xts - mu_c, ic, Xts - mu_c) for mu_c in means], axis=0)
        energy = -np.log(np.exp(Z - Z.max(1, keepdims=True)).sum(1)) - Z.max(1)
        acc = (m.classes_[P.argmax(1)][~is_unk] == test.tool.values[~is_unk]).mean()
        out.append(dict(tag=tag, features="+".join(cols) + ("+markov" if markov else ""), rep=rep, unseen="|".join(unseen), closed_acc=acc, n_known_test=int((~is_unk).sum()),
                        AUROC_msp=roc_auc_score(is_unk, 1 - P.max(1)), AUROC_energy=roc_auc_score(is_unk, energy), AUROC_maha=roc_auc_score(is_unk, maha)))
    return out

if __name__ == "__main__":
    tr1 = load_turns("main_seed1"); tr2 = load_turns("main_seed2"); te3 = load_turns("main_seed3")
    S = fit_phi(tr1)
    mux_tr = [load_turns(t) for t in ("subM2b", "subM4b", "subM8b") if chunks_of(t)]
    G_tr = conns(pd.concat([tr1, tr2] + mux_tr)); model = train_pair_model(G_tr, S); print("pair model trained on", len(G_tr), "conns", flush=True)
    rows = []
    tests = {"main_seed3": te3, **{t: load_turns(t) for t in ("subM1", "subM2", "subM4", "subM8") if chunks_of(t)}}
    for tag, d in tests.items(): r, _ = link_eval(tag, conns(d), S, model); rows += r
    P = pd.DataFrame(rows); P.to_csv(f"{RES}/graph_parent.csv", index=False); print(P.round(3).to_string(), flush=True)
    # edge typing
    ff = [load_turns("failfast", [0, 1]), load_turns("failfast", [2, 3])] if chunks_of("failfast") else None
    trE = edge_table(pd.concat([tr1, tr2] + mux_tr + ([ff[0]] if ff else [])), S, model)
    res = []; cms = {}
    evals = {"main_seed3": te3, **({"failfast": ff[1]} if ff else {}), **{t: tests[t] for t in ("subM8",) if t in tests}}
    for tag, d in evals.items():
        MISSED[tag] = d[d.missed].gt_edge.values; MISSED_CONN[tag] = d[d.missed].conn.values
        teE = edge_table(d, S, model)
        for name, cols in (("slice_only", SLICE), ("v3_full", SLICE + CTXF), ("v3_no_context", SLICE + ["conc", "first_on_conn", "seq_gap"]),
                           ("v3_no_latency", [c for c in SLICE + CTXF if c not in ("m_gap", "nx_gap", "seq_gap")])):
            r, cm, _ = edge_eval(name, trE, teE, cols, tag); res.append(r)
            if name == "v3_full": pd.DataFrame(cm, index=EDGE + ["missed"], columns=EDGE + ["missed"]).to_csv(f"{RES}/graph_confusion_{tag}.csv")
    E = pd.DataFrame(res); E.to_csv(f"{RES}/graph_edges.csv", index=False); print(E.round(3).to_string(), flush=True)
    # open-set tool ID
    T = tool_table(pd.concat([tr1, tr2, te3], ignore_index=True), S).reset_index(drop=True); out = []
    out += openset(T, ["y_hat", "gap"], "main")                               # paper-equivalent size+latency
    out += openset(T, TF, "main"); out += openset(T, TF, "main", markov=True)
    Tg = T.assign(true_logsize=np.log1p(T.gt_tool_bytes_added.astype(float)))
    out += openset(Tg, ["true_logsize", "gap"], "main_GT_ceiling")            # corpus ceiling: exact size + observed latency
    O = pd.DataFrame(out); O.to_csv(f"{RES}/tool_openset_reps.csv", index=False)
    Os = O.groupby(["tag", "features"])[["closed_acc", "AUROC_msp", "AUROC_energy", "AUROC_maha"]].agg(["mean", "std"]); Os.to_csv(f"{RES}/tool_openset.csv"); print(Os.round(3).to_string())
    # corpus diagnostics: are args-length / transition priors informative by construction?
    diag = dict(n=len(T), MI_proxy_argslen_vs_tool=float(T.groupby("tool").a_prev.median().std()), transition_chi2_p=None)
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(T[T.prev_tool != "none"].prev_tool, T[T.prev_tool != "none"].tool); diag["transition_chi2_p"] = float(chi2_contingency(ct)[1])
    json.dump(diag, open(f"{RES}/tool_corpus_ceiling.json", "w"), indent=1); print(diag)
