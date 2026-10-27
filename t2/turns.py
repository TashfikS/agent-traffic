"""Observer-side turn table: one row per detected request boundary, observer features only;
ground truth is attached by record identity (size-aware assignment) for evaluation."""
import os, numpy as np, pandas as pd
from t2.data import *
from t2.seg import frame_candidates, frame_rule, assign_sizeaware, FRAME_COLS, burst_rule
GTC = ["turn", "parent", "edge", "tool_bytes_added", "n_tools_added", "n_err_added", "tools_added", "arch", "ep", "depth", "conv",
       "status", "req_bytes", "resp_bytes", "t_send", "t_end"]

def conn_turns(o, g, mode):
    c, Fc = frame_candidates(o); asg = assign_sizeaware(o, g)
    cont = Fc[:, FRAME_COLS.index("cont")] if len(c) else np.zeros(0)
    if mode == "oracle": bset = np.array(sorted(asg), int)
    elif mode == "burst":
        bset = burst_rule(dict(o=o)); tt_, up_ = o.t.values, o.up.values; remap = {}
        for bi in bset:           # a GT request matches the burst that contains its size-matched record
            jj = bi
            while jj + 1 < len(tt_) and up_[jj + 1] and tt_[jj + 1] - tt_[jj] < 0.005: jj += 1
            for ri in [r_ for r_ in asg if bi <= r_ <= jj]: remap[bi] = asg[ri]; break
        asg = remap
    else: bset = c[cont == 0]
    t = o.t.values; up = o.up.values; L = o.plen.values.astype(float); cpos = {ri: k for k, ri in enumerate(c)}
    rows = []; ends = np.r_[bset[1:], len(t)]
    for k, (b, e) in enumerate(zip(bset, ends)):
        j = cpos.get(b, None); req = L[b]; nrec = 1
        if mode == "burst":       # request = the whole upstream write burst (records < 5 ms apart, no downstream data between)
            jj = b + 1
            while jj < e and up[jj] and t[jj] - t[jj - 1] < 0.005: req += L[jj] if L[jj] > 64 else 0; nrec += 1; jj += 1
            j = None
        if j is not None:
            jj = j + 1
            while jj < len(c) and c[jj] < e and cont[jj] == 1: req += L[c[jj]]; nrec += 1; jj += 1
        seg = slice(b + 1, e); dn = ~up[seg]; Ld = L[seg][dn]; td = t[seg][dn]
        r = dict(rec=int(b), k=k, t0=t[b], req=req, n_up=nrec, down=Ld.sum(), n_down=int(len(Ld)),
                 ttft=(td[0] - t[b]) if len(td) else np.nan, t_last=(td[-1] if len(td) else t[b]),
                 n_same_t=Fc[cpos[b], FRAME_COLS.index("n_same_t")] if b in cpos else 1,
                 dn_small_frac=float((Ld < 200).mean()) if len(Ld) else 0.0,
                 dn_med=float(np.median(Ld)) if len(Ld) else 0.0)
        if b in asg:
            gr = g.iloc[asg[b]]; r.update({f"gt_{x}": gr[x] for x in GTC if x in gr}); 
            for x in ("ctx_event", "ctx_dropped"):
                if x in gr: r[f"gt_{x}"] = gr[x]
            r["spurious"] = False
        else: r["spurious"] = True
        rows.append(r)
    missed = set(range(len(g))) - {asg[b] for b in bset if b in asg}
    for pos in missed:
        gr = g.iloc[pos]; r = dict(rec=-1, k=-1, t0=gr.t_send, missed=True, spurious=False); r.update({f"gt_{x}": gr[x] for x in GTC if x in gr}); rows.append(r)
    return rows

def build_turns(tag, mode="oracle", refresh=False):
    p = f"{PQ}/turns_{tag}_{mode}.parquet"
    if os.path.exists(p) and not refresh: return pd.read_parquet(p)
    gt, obs = load(tag); rows = []
    for conn, o, g in conn_groups(gt, obs):
        iv = g[["t_send", "t_end"]].values; iv = iv[np.argsort(iv[:, 0])]
        conc = int(max(((iv[:, 0] <= x) & (iv[:, 1] > x)).sum() for x in iv[:, 0])) if len(iv) else 0
        for r in conn_turns(o, g, mode): r.update(conn=conn, chunk=int(g.chunk.iloc[0]), mux=conc > 1, tag=tag); rows.append(r)
    df = pd.DataFrame(rows); df["missed"] = df.get("missed", False); df["missed"] = df["missed"].fillna(False).astype(bool)
    df = df.sort_values(["conn", "t0"]).reset_index(drop=True)
    # sequential (previous-on-connection) context features over detected turns
    det = ~df.missed
    for col in ("p_req", "p_down", "p_ndown", "p_tlast", "p_turn", "first_req"): df[col] = np.nan
    for conn, g in df[det].groupby("conn"):
        i = g.index
        df.loc[i, "p_req"] = g.req.shift(1).values; df.loc[i, "p_down"] = g.down.shift(1).values; df.loc[i, "p_ndown"] = g.n_down.shift(1).values
        df.loc[i, "p_tlast"] = g.t_last.shift(1).values; df.loc[i, "p_turn"] = g.gt_turn.shift(1).values if "gt_turn" in g else np.nan
        df.loc[i, "first_req"] = g.req.iloc[0]; df.loc[i, "max_prev_req"] = g.req.cummax().shift(1).values
        df.loc[i, "pos"] = np.arange(len(g))
    df["dreq"] = df.req - df.p_req; df["gap"] = df.t0 - df.p_tlast
    df.to_parquet(p); return df
