"""Shared loading / observer-feature utilities for Track 1 analysis (observer sees only TLS record headers)."""
import pandas as pd, numpy as np, json, os
OVH = 17           # TLS 1.3 AEAD tag (16) + inner content type (1)
R = os.path.join(os.path.dirname(__file__), "..", "results_v2")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
TS = 0.1           # testbed time compression; real-time-equivalent = compressed / TS

import glob
def chunks_of(tag): return sorted(glob.glob(f"{R}/gt_{tag}_c*.csv"))
def load(tag):
    G, O = [], []
    for k, fg in enumerate(chunks_of(tag)):
        fo = fg.replace("/gt_", "/obs_"); g = pd.read_csv(fg); o = pd.read_csv(fo)
        g["port"] = g.port + 100000*k; o["port"] = o.port + 100000*k; g["chunk"] = k; G.append(g); O.append(o)
    gt = pd.concat(G, ignore_index=True); obs = pd.concat(O, ignore_index=True)
    obs = obs[obs.ctype == 23].copy(); obs["plen"] = obs.len - OVH
    meta = json.load(open(chunks_of(tag)[0].replace("/gt_", "/meta_").replace(".csv", ".json")))
    return gt, obs.sort_values(["port", "t"]).reset_index(drop=True), meta

def record_features(o):
    """Per-record features on one connection (sorted by t)."""
    t = o.t.values; up = (o.dir.values == "c2s").astype(float); L = o.plen.values.astype(float)
    dtp = np.r_[1.0, np.diff(t)]; dtn = np.r_[np.diff(t), 1.0]
    # time since last upstream record with payload > 64 B, and since last downstream record
    last_up = np.full(len(t), np.nan); last_dn = np.full(len(t), np.nan); lu = ld = -1e9
    for i in range(len(t)):
        last_up[i] = t[i] - lu; last_dn[i] = t[i] - ld
        if up[i] and L[i] > 64: lu = t[i]
        if not up[i]: ld = t[i]
    # upstream bytes in the next 5 ms (compressed) burst
    j = np.searchsorted(t, t + 0.005); cs = np.r_[0, np.cumsum(L*up)]
    burst_up = cs[j] - cs[np.arange(len(t))]
    F = np.c_[up, np.log1p(L), np.log1p(dtp*1e3), np.log1p(dtn*1e3), np.log1p(np.minimum(last_up, 10)*1e3),
              np.log1p(np.minimum(last_dn, 10)*1e3), np.log1p(burst_up)]
    return F

def label_boundaries(o, gt_port):
    """Record index of the first upstream record of each ground-truth request on this port."""
    t = o.t.values; up = o.dir.values == "c2s"; y = np.zeros(len(t), int)
    for ts in gt_port.t_send.values:
        k = np.searchsorted(t, ts - 1e-4)
        while k < len(t) and not (up[k] and o.plen.values[k] > 64): k += 1
        if k < len(t): y[k] = 1
    return y

def match_boundaries(pred_t, true_t, tol):
    """Greedy one-to-one matching within tol seconds. Returns tp, fp, fn."""
    pred_t = np.sort(pred_t); true_t = np.sort(true_t); used = np.zeros(len(pred_t), bool); tp = 0
    for tt in true_t:
        k = np.searchsorted(pred_t, tt)
        best = None
        for j in (k-1, k, k+1):
            if 0 <= j < len(pred_t) and not used[j] and abs(pred_t[j]-tt) <= tol and (best is None or abs(pred_t[j]-tt) < abs(pred_t[best]-tt)): best = j
        if best is not None: used[best] = True; tp += 1
    return tp, len(pred_t) - tp, len(true_t) - tp

def seg_iou(pred_t, true_t, t_end):
    """Mean over true segments of best IoU with a predicted segment (segments = boundary to next boundary)."""
    def segs(b):
        b = np.sort(b); return list(zip(b, np.r_[b[1:], t_end]))
    P, T = segs(pred_t), segs(true_t)
    if not T: return np.nan
    out = []
    for a, b in T:
        best = 0
        for c, d in P:
            inter = max(0, min(b, d) - max(a, c)); uni = max(b, d) - min(a, c)
            if uni > 0: best = max(best, inter/uni)
        out.append(best)
    return float(np.mean(out))

def boot_ci(vals, B=1000, f=np.mean, seed=0):
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
    if len(vals) == 0: return (np.nan, np.nan, np.nan)
    r = np.random.default_rng(seed); bs = [f(vals[r.integers(0, len(vals), len(vals))]) for _ in range(B)]
    return float(f(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def boot_ratio_ci(num, den, B=1000, seed=0):
    """CI of sum(num)/sum(den) resampling units (e.g., connections)."""
    num, den = np.asarray(num, float), np.asarray(den, float); r = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = r.integers(0, len(num), len(num)); bs.append(num[i].sum() / max(den[i].sum(), 1e-9))
    return float(num.sum()/max(den.sum(), 1e-9)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
