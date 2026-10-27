"""Shared frame-aware segmentation primitives (labellers, candidate features, frame rule)."""
import numpy as np
from t2.data import MAXREC
import os
LABEL_WIN = float(os.environ.get("T2_LABEL_WIN", "0.06"))   # GT->record assignment window (s); real WAN captures: 1.0
PAPER_OVH = 86                    # observed HEADERS(+HPACK)+DATA frame overhead on a non-first h2 request

def labels_paper(o, g):
    """Paper labeller: first upstream >64 B record after each t_send. Concurrent requests flushed in one read
    collapse onto one record (y[k]=1 written twice) -> label loss under multiplexing (audited below)."""
    t = o.t.values; c = o.up.values & (o.plen.values > 64); y = np.zeros(len(t), int)
    for ts in g.t_send.values:
        k = np.searchsorted(t, ts - 1e-4)
        while k < len(t) and not c[k]: k += 1
        if k < len(t): y[k] = 1
    return y

def labels_sizeaware(o, g, win=None):
    """Corrected labeller: one-to-one assignment of each GT request to the unassigned upstream record within
    [t_send, t_send+win] whose size best matches min(16384, req_bytes+overhead). Uses GT only for labels."""
    win = win or LABEL_WIN; t = o.t.values; L = o.plen.values; c = np.where(o.up.values & (L > 64))[0]; y = np.zeros(len(t), int); used = set()
    tc = t[c]
    for ts, rb in zip(g.t_send.values, g.req_bytes.values):
        lo, hi = np.searchsorted(tc, ts - 1e-4), np.searchsorted(tc, ts + win)
        best, bs = None, 1e9
        for j in range(lo, hi):
            i = c[j]
            if i in used: continue
            s = abs(np.log(L[i]) - np.log(min(MAXREC, rb + PAPER_OVH))) + 2.0 * (t[i] - ts) / win
            if s < bs: best, bs = i, s
        if best is not None: used.add(best); y[best] = 1
    return y

# ---------------------------------------------------------------- features
def paper_record_features(o):
    t = o.t.values; up = o.up.values.astype(float); L = o.plen.values.astype(float)
    dtp = np.r_[1.0, np.diff(t)]; dtn = np.r_[np.diff(t), 1.0]
    big_up = (up > 0) & (L > 64)
    iu = np.where(big_up, np.arange(len(t)), -1); iu = np.maximum.accumulate(iu)   # last big-up index <= i
    prev_u = np.r_[-1, iu[:-1]]; last_up = np.where(prev_u >= 0, t - t[np.maximum(prev_u, 0)], 1e9)
    idn = np.where(up == 0, np.arange(len(t)), -1); idn = np.maximum.accumulate(idn); prev_d = np.r_[-1, idn[:-1]]
    last_dn = np.where(prev_d >= 0, t - t[np.maximum(prev_d, 0)], 1e9)
    j = np.searchsorted(t, t + 0.005); cs = np.r_[0, np.cumsum(L * up)]; burst_up = cs[j] - cs[np.arange(len(t))]
    return np.c_[up, np.log1p(L), np.log1p(dtp*1e3), np.log1p(dtn*1e3), np.log1p(np.minimum(last_up, 10)*1e3),
                 np.log1p(np.minimum(last_dn, 10)*1e3), np.log1p(burst_up)]

FRAME_COLS = ["logL", "is_full", "prev_full", "cont", "dt_prev_c", "dt_prev_dn", "n_same_t", "rank_same_t", "n_dn_between",
              "logdn_between", "dt_next_dn", "n_dn_next5", "res_ovh", "first_on_conn", "log_cum_up"]
def frame_candidates(o):
    """Candidate = upstream application record with payload > 64 B (excludes WINDOW_UPDATE/SETTINGS/PING)."""
    t = o.t.values; up = o.up.values; L = o.plen.values.astype(float); n = len(t)
    c = np.where(up & (L > 64))[0]
    if not len(c): return c, np.zeros((0, len(FRAME_COLS)))
    tc, Lc = t[c], L[c]
    prev_full = np.r_[0, (Lc[:-1] >= MAXREC)].astype(float)
    dtp = np.r_[1.0, np.diff(tc)]
    cont = ((prev_full > 0) & (dtp < 0.002)).astype(float)
    dn = ~up; cdn = np.r_[0, np.cumsum(dn)]; cdb = np.r_[0, np.cumsum(np.where(dn, L, 0))]
    prev_c = np.r_[0, c[:-1]]
    n_dn_b = cdn[c] - cdn[prev_c]; dnb = cdb[c] - cdb[prev_c]; n_dn_b[0] = cdn[c[0]]; dnb[0] = cdb[c[0]]
    idn = np.where(dn, np.arange(n), -1); idn = np.maximum.accumulate(idn)
    pd_ = np.r_[-1, idn[:-1]][c]; dt_prev_dn = np.where(pd_ >= 0, tc - t[np.maximum(pd_, 0)], 10.0)
    nxt = np.searchsorted(np.where(dn)[0], c); dpos = np.where(dn)[0]
    dt_next_dn = np.where(nxt < len(dpos), t[dpos[np.minimum(nxt, len(dpos)-1)]] - tc, 10.0)
    j5 = np.searchsorted(t, tc + 0.005); n_dn5 = cdn[j5] - cdn[c]
    # same-timestamp groups (one relay read): size and rank
    _, inv, cnt = np.unique(tc, return_inverse=True, return_counts=True); n_same = cnt[inv].astype(float)
    rank = np.zeros(len(c)); 
    for k in range(1, len(c)): rank[k] = rank[k-1] + 1 if tc[k] == tc[k-1] else 0
    res = np.mod(Lc - PAPER_OVH, MAXREC) / MAXREC
    F = np.c_[np.log1p(Lc), (Lc >= MAXREC), prev_full, cont, np.log1p(np.minimum(dtp, 10)*1e3), np.log1p(np.minimum(dt_prev_dn, 10)*1e3),
              n_same, rank, np.log1p(n_dn_b), np.log1p(dnb), np.log1p(np.minimum(dt_next_dn, 10)*1e3), np.log1p(n_dn5), res,
              (np.arange(len(c)) == 0), np.log1p(np.cumsum(Lc))]
    return c, F.astype(np.float32)

def frame_rule(d): return d["c"][d["Fc"][:, FRAME_COLS.index("cont")] == 0]

def assign_sizeaware(o, g, win=None):
    """Same as labels_sizeaware but returns {record_index: gt_row_position}."""
    win = win or LABEL_WIN; t = o.t.values; L = o.plen.values; c = np.where(o.up.values & (L > 64))[0]; tc = t[c]; used = {}
    for pos, (ts, rb) in enumerate(zip(g.t_send.values, g.req_bytes.values)):
        lo, hi = np.searchsorted(tc, ts - 1e-4), np.searchsorted(tc, ts + win); best, bs = None, 1e9
        for j in range(lo, hi):
            i = c[j]
            if i in used: continue
            sc = abs(np.log(L[i]) - np.log(min(MAXREC, rb + PAPER_OVH))) + 2.0 * (t[i] - ts) / win
            if sc < bs: best, bs = i, sc
        if best is not None: used[best] = pos
    return used

def burst_rule(d, g=0.005, smin=300):
    """Real-stack boundary rule: upstream application records (>64 B plaintext) closer than g seconds form one write
    burst (h2 HEADERS + DATA + control frames flushed together by the client / CDN edge); a burst carrying >= smin bytes
    starts a request, at the time of its first record.  (g, smin) fitted on real_m4 v1: (5 ms, 300 B)."""
    o = d["o"]; t = o.t.values; L = o.plen.values; c = np.where(o.up.values & (L > 64))[0]; out = []; i = 0
    while i < len(c):
        j = i; tot = L[c[i]]
        while j + 1 < len(c) and t[c[j + 1]] - t[c[j]] < g: j += 1; tot += L[c[j]]
        if tot >= smin: out.append(c[i])
        i = j + 1
    return np.array(out, int)
