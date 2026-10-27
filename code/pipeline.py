"""Observer-side turn extraction, context-growth linking and feature construction (Track 1)."""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, joblib, torch, sys
sys.path.insert(0, os.path.dirname(__file__)); from common import *
EDGE = ["start", "sequential", "parallel_fork_join", "transport_retry", "semantic_retry", "sub_agent"]

_SEG = None
def segment(o, F):
    """Log-free boundaries with the trained GBM segmenter (best-performing on held-out data; see segmenter.py)."""
    global _SEG
    if _SEG is None: _SEG = joblib.load(f"{OUT}/segmenter_models.pkl")
    from segmenter import nms
    p = _SEG["gbm"].predict_proba(F)[:, 1]; return nms(o.t.values, p, _SEG["thr"])

def turns_from_boundaries(o, bidx, port):
    t = o.t.values; up = o.dir.values == "c2s"; L = o.plen.values; rows = []
    bidx = np.sort(np.asarray(bidx, int)); ends = np.r_[bidx[1:], len(t)]
    for k, (b, e) in enumerate(zip(bidx, ends)):
        # upstream burst: contiguous upstream records (allow WINDOW_UPDATE-size downstream gaps) within 5 ms
        j = b; last = t[b]; req = 0; nrec_up = 0
        while j < e:
            if up[j] and L[j] > 40:
                if t[j] - last > 0.005: break
                req += L[j]; nrec_up += 1; last = t[j]
            j += 1
        dn = (~up[b:e]); Ld = L[b:e][dn]; td = t[b:e][dn]
        rows.append(dict(port=port, k=k, t0=t[b], req=req, n_up=nrec_up, down=Ld.sum(), n_down=len(Ld),
                         ttft=(td[0] - t[b]) if len(td) else np.nan, t_last=(td[-1] if len(td) else t[b]),
                         down_small_frac=float((Ld < 200).mean()) if len(Ld) else 0.0,
                         hist=np.histogram(np.log1p(L[b:e]), bins=12, range=(0, 10))[0] * np.where(True, 1, 1),
                         seq=np.c_[np.where(up[b:e], 1, -1)*np.log1p(L[b:e]), np.log1p(np.diff(np.r_[t[b], t[b:e]])*1e4)][:64]))
    return rows

def link_and_features(T):
    """Context-growth linking: predecessor = earlier turn on the port with the largest request size that is
    <= this request (+64 B) and whose response ended before this request started."""
    T = sorted(T, key=lambda r: r["t0"]); out = []
    for i, r in enumerate(T):
        best = None
        for j in range(i):
            q = T[j]
            if q["t_last"] <= r["t0"] + 1e-4 and q["req"] <= r["req"] + 64 and (best is None or q["req"] > T[best]["req"]): best = j
        fresh = best is None or (r["req"] < 0.5*max([q["req"] for q in T[:i]] + [1]) and r["req"] - T[best]["req"] < 0)
        q = T[best] if best is not None else None
        conc = sum(1 for q2 in T[:i] if q2["t_last"] > r["t0"])
        f = dict(r)
        f.update(pred_idx=best, dreq=(r["req"] - q["req"]) if q else 0.0, dreq_ratio=((r["req"] - q["req"])/max(q["req"], 1)) if q else 0.0,
                 prev_down=q["down"] if q else 0.0, prev_ndown=q["n_down"] if q else 0, gap=(r["t0"] - q["t_last"]) if q else 0.0,
                 prev_small=float(q["down"] < 400) if q else 0.0, first_on_port=float(i == 0), fresh=float(fresh), conc=conc,
                 since_prev_any=(r["t0"] - T[i-1]["t_last"]) if i else 0.0)
        out.append(f)
    return out

SLICE = ["req", "n_up", "down", "n_down", "ttft", "down_small_frac"]
CTX = ["dreq", "dreq_ratio", "prev_down", "prev_ndown", "gap", "prev_small", "first_on_port", "fresh", "conc", "since_prev_any"]

def build(tag, boundaries="oracle"):
    """Return per-turn feature DataFrame for a dataset, with ground-truth labels attached by time matching."""
    gt, obs, meta = load(tag); rows = []
    gt = gt.sort_values("t_send")
    for port, o in obs.groupby("port"):
        g = gt[gt.port == port]
        if len(g) == 0: continue
        o = o.reset_index(drop=True); F = record_features(o)
        if boundaries == "oracle":
            y = label_boundaries(o, g); bidx = np.where(y == 1)[0]
        else: bidx = segment(o, F)
        T = link_and_features(turns_from_boundaries(o, bidx, port))
        tt = g.t_send.values; used = set()
        for r in T:
            k = int(np.argmin(np.abs(tt - r["t0"]))) if len(tt) else -1
            if k >= 0 and abs(tt[k] - r["t0"]) < 0.025 and k not in used:
                used.add(k); gr = g.iloc[k]
                r.update(gt_turn=gr.turn, edge=gr.edge, gt_parent=gr.parent, tool_bytes=gr.tool_bytes_added, n_tools=gr.n_tools_added,
                         n_err=gr.n_err_added, tools=gr.tools_added, arch=gr.arch, ep=gr.ep, chunk=gr.chunk, depth=gr.depth, status=gr.status,
                         req_plain=gr.req_bytes, resp_plain=gr.resp_bytes)
            else: r.update(gt_turn=-1, edge="spurious", arch=g.arch.iloc[0], ep=g.ep.iloc[0], chunk=g.chunk.iloc[0])
            rows.append(r)
        # ground-truth turns that were not detected
        for k in set(range(len(g))) - used:
            gr = g.iloc[k]; rows.append(dict(port=port, gt_turn=gr.turn, edge=gr.edge, gt_parent=gr.parent, missed=True, arch=gr.arch, ep=gr.ep, chunk=gr.chunk,
                                             tool_bytes=gr.tool_bytes_added, n_tools=gr.n_tools_added, depth=gr.depth))
    df = pd.DataFrame(rows); df["missed"] = df.get("missed", False); df["missed"] = df["missed"].fillna(False).astype(bool)
    # map predicted predecessor index -> gt turn id
    df["pred_parent"] = -1
    for port, g in df[~df.missed].groupby("port"):
        g = g.sort_values("t0"); ids = g.gt_turn.values; pi = g.pred_idx.values
        df.loc[g.index, "pred_parent"] = [ids[int(p)] if p == p and p is not None else -1 for p in pi]
    df["tag"] = tag; return df
