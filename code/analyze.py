"""Observer-side analysis of the Topic 2 testbed captures (ciphertext record sizes + timing only)."""
import pandas as pd, numpy as np, os, json, sys
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
R = sys.argv[1]
OVH = 17  # TLS1.3 AES-GCM: 16-byte tag + 1 inner content type

def bursts(obs):
    out = []
    for cid, g in obs[obs.ctype == 23].groupby("conn"):
        g = g.sort_values("t"); cur = None
        for r in g.itertuples():
            if cur and cur["dir"] == r.dir: cur["bytes"] += r.len - OVH; cur["t1"] = r.t
            else:
                if cur: out.append(cur)
                cur = dict(conn=cid, dir=r.dir, t0=r.t, t1=r.t, bytes=r.len - OVH)
        out.append(cur)
    b = pd.DataFrame(out); return b

summary = {}
for cond in [c for c in ["full", "server_state", "gzip", "padding", "stream"] if os.path.exists(f"{R}/gt_{c}.csv")]:
    gt = pd.read_csv(f"{R}/gt_{cond}.csv"); obs = pd.read_csv(f"{R}/obs_{cond}.csv")
    b = bursts(obs)
    req = b[(b.dir == "c2s") & (b.bytes > 200)].copy()        # drop TLS Finished-only flights
    resp = b[b.dir == "s2c"].copy()
    # match observed requests to ground truth by send time (evaluation only)
    gt = gt.sort_values("t_send").reset_index(drop=True)
    req = req.sort_values("t0").reset_index(drop=True)
    idx = np.searchsorted(req.t0.values, gt.t_send.values)
    best = []
    for i, t in zip(idx, gt.t_send.values):
        c = [j for j in (i - 1, i, i + 1) if 0 <= j < len(req)]
        j = min(c, key=lambda j: abs(req.t0[j] - t)); best.append(j if abs(req.t0[j] - t) < 0.05 else -1)
    gt["oreq"] = best
    seg_recall = (gt.oreq >= 0).mean(); seg_prec = gt.oreq[gt.oreq >= 0].nunique()/len(req)
    gt = gt[gt.oreq >= 0].copy()
    gt["obs_req"] = req.bytes.values[gt.oreq]; gt["conn"] = req.conn.values[gt.oreq]; gt["t_obs"] = req.t0.values[gt.oreq]
    # response following each request on the same connection
    rs = []
    for r in gt.itertuples():
        c = resp[(resp.conn == r.conn) & (resp.t0 > r.t_obs)]
        rs.append(c.bytes.iloc[0] if len(c) else np.nan)
    gt["obs_resp"] = rs
    gt = gt.sort_values(["conv", "t_send"])
    gt["prev_req"] = gt.groupby("conv").obs_req.shift(); gt["prev_resp"] = gt.groupby("conv").obs_resp.shift()
    gt["prev_trecv"] = gt.groupby("conv").t_recv.shift()
    gt["gap"] = gt.t_send - gt.prev_trecv; gt["dreq"] = gt.obs_req - gt.prev_req
    # ---- (a) tool-result size estimation ----
    s = gt[(gt.tool_bytes_added > 0) & ~gt.retry & gt.prev_req.notna()].copy()
    X = np.c_[s.obs_req if cond == "server_state" else s.dreq, s.prev_resp, np.ones(len(s))]
    tr = (s.ep % 2 == 0).values
    coef, *_ = np.linalg.lstsq(X[tr], s.tool_bytes_added.values[tr], rcond=None)
    pred = X[~tr] @ coef; y = s.tool_bytes_added.values[~tr]
    ape = np.abs(pred - y)/y
    rho = spearmanr(pred, y).correlation
    s.loc[~tr, "pred"] = pred
    s.to_csv(f"{R}/size_est_{cond}.csv", index=False)
    # ---- (c) step-type classification ----
    g = gt[gt.prev_req.notna() & (gt.depth == 0)].copy()
    subs = gt[gt.depth == 1].groupby("ep").t_send.apply(list).to_dict()
    def lab(r):
        if r.retry: return "retry"
        if any(r.prev_trecv < t < r.t_send for t in subs.get(r.ep, [])): return "subagent"
        return "parallel" if r.n_calls_added > 1 else "sequential"
    g["label"] = g.apply(lab, axis=1)
    conns = gt.groupby("conn").t_obs.min()
    g["new_conn_in_gap"] = [((conns > a) & (conns < bb)).sum() for a, bb in zip(g.prev_trecv, g.t_send)]
    g["dreq_ratio"] = g.dreq/g.prev_req
    feats = ["dreq", "prev_resp", "gap", "new_conn_in_gap", "dreq_ratio", "obs_req"]
    tr = (g.ep % 2 == 0).values
    clf = RandomForestClassifier(300, random_state=0).fit(g[feats][tr].fillna(0), g.label[tr])
    p = clf.predict(g[feats][~tr].fillna(0)); yt = g.label[~tr]
    labels = ["sequential", "parallel", "retry", "subagent"]
    cm = confusion_matrix(yt, p, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(f"{R}/confusion_{cond}.csv")
    rep = classification_report(yt, p, labels=labels, output_dict=True, zero_division=0)
    summary[cond] = dict(n_requests=len(gt), request_boundary_recall=round(seg_recall, 4), request_boundary_precision=round(seg_prec, 4),
                         size_n_test=int((~tr).sum()) if False else len(y), size_median_ape_pct=round(100*np.median(ape), 2),
                         size_p90_ape_pct=round(100*np.quantile(ape, .9), 2), size_spearman=round(rho, 4),
                         size_coef=[round(c, 4) for c in coef],
                         steptype_macroF1=round(f1_score(yt, p, average="macro"), 3),
                         steptype_F1={k: round(rep[k]["f1-score"], 3) for k in labels})
json.dump(summary, open(f"{R}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
