"""v3 data layer: raw testbed chunks -> parquet; observer sees only (conn, dir, t, ctype, len) of TLS records.
Ground truth (gt_*.csv) is used ONLY for labels/evaluation."""
import os as _o; _o.environ.setdefault("KMP_AFFINITY", "disabled"); _o.environ.setdefault("OMP_NUM_THREADS", "4")
import os, glob, json, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.environ.get("T2_RAW", f"{ROOT}/data/raw_v3"); PQ = os.environ.get("T2_PQ", f"{ROOT}/data/parquet")
RES = os.environ.get("T2_RES", f"{ROOT}/results_v3"); FIG = os.environ.get("T2_FIG", f"{ROOT}/figures_v3")
for d in (PQ, RES, FIG): os.makedirs(d, exist_ok=True)
OVH = 17            # TLS 1.3 AEAD tag (16) + inner content type (1)
H2_FRAME = 9        # HTTP/2 frame header
MAXREC = 16384      # TLS max plaintext / default h2 max frame
TS = float(os.environ.get("T2_TS", "0.1"))   # testbed time compression (real captures: T2_TS=1)

def chunks_of(tag): return sorted(glob.glob(f"{RAW}/gt_{tag}_c*.csv"))

def load(tag, refresh=False):
    pg, po = f"{PQ}/gt_{tag}.parquet", f"{PQ}/obs_{tag}.parquet"
    if os.path.exists(pg) and os.path.exists(po) and not refresh:
        return pd.read_parquet(pg), pd.read_parquet(po)
    G, O = [], []
    for k, fg in enumerate(chunks_of(tag)):
        g = pd.read_csv(fg); o = pd.read_csv(fg.replace("/gt_", "/obs_"))
        g["conn"] = g.port + 100000 * k; o["conn"] = o.port + 100000 * k; g["chunk"] = k; o["chunk"] = k
        G.append(g); O.append(o)
    if not G: raise FileNotFoundError(tag)
    gt = pd.concat(G, ignore_index=True); obs = pd.concat(O, ignore_index=True)
    obs = obs[obs.ctype == 23].copy(); obs["plen"] = (obs.len - OVH).astype(np.int32); obs["up"] = (obs.dir == "c2s")
    obs = obs[["conn", "chunk", "t", "up", "plen"]].sort_values(["conn", "t"], kind="stable").reset_index(drop=True)
    gt["tag"] = tag; gt.to_parquet(pg); obs.to_parquet(po); return gt, obs

def conn_groups(gt, obs):
    gi = {c: g.sort_values("t_send") for c, g in gt.groupby("conn")}
    for c, o in obs.groupby("conn", sort=False):
        if c in gi: yield c, o.reset_index(drop=True), gi[c]

def boot_ratio_ci(num, den, B=1000, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float); r = np.random.default_rng(seed)
    i = r.integers(0, len(num), (B, len(num))); bs = num[i].sum(1) / np.maximum(den[i].sum(1), 1e-9)
    return float(num.sum() / max(den.sum(), 1e-9)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def boot_ci(vals, f=np.median, B=1000, seed=0):
    v = np.asarray(vals, float); v = v[~np.isnan(v)]; r = np.random.default_rng(seed)
    if not len(v): return (np.nan,) * 3
    bs = [f(v[r.integers(0, len(v), len(v))]) for _ in range(B)]
    return float(f(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
