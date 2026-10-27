"""08 - External validity on ANT (Li et al., arXiv:2610.06514; HF benzenenene/ANT, fixed train/val/test splits).
ANT releases per-episode pcaps of real coding/research agents (Claude Code with DeepSeek / MiMo / relay back-ends,
Qwen Code, Codex) plus coarse step annotations, but no per-request ground truth.  We therefore test (i) whether the
observer pipeline finds requests and how many relative to annotated assistant steps, (ii) Prop. 1's append-only
assumption in the wild (request shrink / compaction rate, context growth per turn), (iii) multiplexing levels that
real agents produce, and (iv) whether the recovered structure supports ANT's own episode-level tasks (risk and
scenario) on the fixed splits, against a volume-histogram baseline on the same flows.
Main LLM host per engine is the endpoint carrying the API traffic (codex: model traffic not visible as TLS/443 SNI in
the capture -> excluded; cc_xiaoai: relay host not captured by the ingest filter -> api.anthropic.com only).
Outputs: results_v3/ant_requests.parquet, ant_engine_summary.csv, ant_tasks.csv, figures_v3/fig_ant.pdf"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled")
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, ROOT)
from t2.data import RES, FIG, boot_ci
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, accuracy_score
A = f"{ROOT}/data/ant"
HOST = {"cc_dpsk": "api.deepseek.com", "USA_cc_dpsk": "api.deepseek.com", "qwen": "dashscope.aliyuncs.com",
        "cc_mimo": "token-plan-cn.xiaomimimo.com", "cc_xiaoai": "api.anthropic.com"}
SMIN = 300                               # minimum request bytes (fitted on our real Cloudflare captures)
SCEN = json.load(open(f"{ROOT}/baselines/ant/scenario.json")) if os.path.exists(f"{ROOT}/baselines/ant/scenario.json") else None

def requests(R):
    """burst rule per connection; request size = burst bytes (TLS overhead removed), response = downstream bytes until next."""
    out = []
    for (ep, srv, cp), o in R.groupby(["ep", "server", "cport"], sort=False):
        o = o[o.ctype == 23].sort_values("t"); t = o.t.values; up = o.up.values; L = (o.len.values - 17).astype(float)
        # significant records (>64 B plaintext); an upstream run = consecutive significant upstream records with no
        # significant downstream record in between (WAN uploads of 100s of KB span many ms, so a pure time gap splits them)
        c = np.where(L > 64)[0]; B = []; i = 0
        while i < len(c):
            if not up[c[i]]: i += 1; continue
            j = i; tot = L[c[i]]
            while j + 1 < len(c) and up[c[j + 1]]: j += 1; tot += L[c[j]]
            if tot >= SMIN: B.append((c[i], c[j], tot))
            i = j + 1
        for k, (b, e, tot) in enumerate(B):
            nx = B[k + 1][0] if k + 1 < len(B) else len(t); dn = (~up[e + 1:nx]); Ld = L[e + 1:nx][dn]; td = t[e + 1:nx][dn]
            out.append(dict(ep=ep, conn=f"{srv}:{cp}", t0=t[b], req=tot, down=Ld.sum(), n_down=len(Ld), t_last=td[-1] if len(td) else t[b],
                            ttft=(td[0] - t[b]) if len(td) else np.nan))
    Q = pd.DataFrame(out).sort_values(["ep", "conn", "t0"]).reset_index(drop=True)
    g = Q.groupby(["ep", "conn"])
    Q["dreq"] = g.req.diff(); Q["shrink"] = Q.dreq < -64; Q["gap"] = Q.t0 - g.t_last.shift(1)
    Q["a_hat"] = g.down.shift(1) - 9 * g.n_down.shift(1); Q["tool_hat"] = Q.dreq - Q.a_hat     # Prop. 1 residual = tool bytes + c_m
    # context-growth parent (hybrid rule of 04): previous request unless it shrank, then the most recent earlier request
    # with req <= req_k + 64 B on the same connection (interleaved side conversations, e.g. title / summary calls)
    Q["lnk_dreq"] = np.nan; Q["lnk_tool_hat"] = np.nan; Q["fresh"] = False
    for _, gg in Q.groupby(["ep", "conn"]):
        rq, dn, nd = gg.req.values, gg.down.values, gg.n_down.values; a = np.full(len(gg), np.nan); b = np.full(len(gg), np.nan); fr = np.zeros(len(gg), bool)
        for k in range(1, len(gg)):
            j = k - 1
            if rq[k] < rq[j] - 64:
                ok = [i for i in range(k - 1, -1, -1) if rq[i] <= rq[k] + 64]
                if not ok: fr[k] = True; continue
                j = ok[0]
            a[k] = rq[k] - rq[j]; b[k] = a[k] - (dn[j] - 9 * nd[j])
        Q.loc[gg.index, "lnk_dreq"] = a; Q.loc[gg.index, "lnk_tool_hat"] = b; Q.loc[gg.index, "fresh"] = fr
    # concurrency on the connection: requests started while an earlier response is still streaming
    Q["conc"] = 1
    for _, gg in Q.groupby(["ep", "conn"]):
        s, e = gg.t0.values, gg.t_last.values; Q.loc[gg.index, "conc"] = [int(((s <= x) & (e > x)).sum()) for x in s]
    return Q

def ep_features(Q):
    f = Q.groupby("ep").agg(n_req=("req", "size"), n_conn=("conn", "nunique"), req_med=("req", "median"), req_max=("req", "max"),
                            down_sum=("down", "sum"), down_med=("down", "median"), shrink=("shrink", "sum"), conc_max=("conc", "max"),
                            gap_med=("gap", "median"), gap_p90=("gap", lambda x: np.nanpercentile(x, 90) if x.notna().any() else np.nan),
                            tool_med=("tool_hat", "median"), tool_p90=("tool_hat", lambda x: np.nanpercentile(x, 90) if x.notna().any() else np.nan),
                            tool_sum=("tool_hat", lambda x: x[x > 0].sum()), ttft_med=("ttft", "median"),
                            tool_small=("tool_hat", lambda x: float(((x > 0) & (x < 1000)).mean())), tool_big=("tool_hat", lambda x: float((x > 20000).mean())))
    f["shrink_rate"] = f.shrink / f.n_req; return f

if __name__ == "__main__":
    R = pd.concat([pd.read_parquet(f"{A}/ant_llm_records_{s}.parquet") for s in ("train", "test_val")], ignore_index=True)
    R["engine"] = R.ep.str.split("/").str[1]; R = R[R.engine.map(HOST).notna() & (R.sni == R.engine.map(HOST))]
    Q = requests(R); Q["split"] = Q.ep.str.split("/").str[0]; Q["engine"] = Q.ep.str.split("/").str[1]; Q["family"] = Q.ep.str.split("/").str[2]
    Q.to_parquet(f"{RES}/ant_requests.parquet")
    M = pd.concat([pd.read_parquet(f"{A}/ant_messages_{s}.parquet") for s in ("train", "test_val")])
    nasst = M[M.role.isin(["assistant", "model"])].groupby("ep").size()
    F = ep_features(Q); F["n_asst"] = nasst.reindex(F.index)
    F["split"] = F.index.str.split("/").str[0]; F["engine"] = F.index.str.split("/").str[1]; F["family"] = F.index.str.split("/").str[2]
    F["risk"] = F.family.isin(["attack_code_generation", "attack_execution"]).astype(int)
    F["task"] = F.index.str.split("/").str[3].str[1:3]
    rows = []
    for e, g in F.groupby("engine"):
        q = Q[Q.engine == e]; v = q.dropna(subset=["dreq"])
        rho = spearmanr(g.n_req, g.n_asst, nan_policy="omit")[0]
        rows.append(dict(engine=e, host=HOST[e], episodes=len(g), requests=len(q), req_per_ep_med=g.n_req.median(), asst_steps_med=g.n_asst.median(),
                         spearman_nreq_vs_asst=rho, req_kB_med=q.req.median() / 1e3, req_kB_p90=q.req.quantile(.9) / 1e3,
                         grow_frac=float((v.dreq > 64).mean()), shrink_frac=float(v.shrink.mean()), eps_with_shrink=float((g.shrink > 0).mean()),
                         tool_hat_med_B=float(v.tool_hat[v.dreq > 0].median()),
                         fresh_frac=float(q.fresh.mean()), linked_resid_nonneg=float((q.lnk_tool_hat.dropna() >= -256).mean()),
                         linked_tool_hat_med_B=float(q.lnk_tool_hat.dropna().median()), mux_frac=float((q.conc > 1).mean()), conc_max=int(q.conc.max()),
                         conns_per_ep_med=g.n_conn.median()))
    E = pd.DataFrame(rows); E.to_csv(f"{RES}/ant_engine_summary.csv", index=False); print(E.round(3).to_string(), flush=True)
    # ---- ANT tasks on fixed splits (train -> test), episode level
    feat = [c for c in F.columns if c not in ("n_asst", "split", "engine", "family", "risk", "task")]
    vol = R.groupby("ep").apply(lambda o: pd.Series(np.histogram(np.log2(o.len.clip(1)), bins=16, range=(0, 16), weights=o.up.map({True: 1, False: -1}))[0]))
    vol.columns = [f"vh{i}" for i in vol.columns]; F = F.join(vol)
    vf = list(vol.columns); out = []
    for target in ("family", "task"):   # risk: ANT test split of these engines holds benign families only
        tr, te = F[F.split == "train"], F[F.split == "test"]
        for name, cols in (("volume_hist (AgentPrint-style)", vf), ("ours: recovered structure", feat), ("ours + volume", feat + vf)):
            y = tr[target].astype(str); clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=15, min_child_samples=5, verbose=-1, random_state=0)
            clf.fit(tr[cols].fillna(-1), y); p = clf.predict(te[cols].fillna(-1)); yt = te[target].astype(str)
            out.append(dict(target=target, features=name, n_train=len(tr), n_test=len(te), acc=accuracy_score(yt, p), macroF1=f1_score(yt, p, average="macro")))
    T = pd.DataFrame(out); T["test_class_counts"] = [json.dumps(F[F.split == "test"][r].astype(str).value_counts().to_dict()) for r in T.target]; T.to_csv(f"{RES}/ant_tasks.csv", index=False); print(T.round(3).to_string())
    F.to_parquet(f"{RES}/ant_episode_features.parquet")
