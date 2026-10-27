"""Step 1.3: log-free turn-boundary detection from TLS record series.
Segmenters: (1) rule (direction + size + silence gap), (2) HSMM-style semi-Markov decoder over
GBM emission log-odds with an empirical duration prior, (3) bidirectional TCN.
Train: main_seed1 + main_seed2; test: main_seed3 (unseen episodes/seed), h1_stream, subM{1,2,4,8}.
"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, json, sys, os, torch, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__)); from common import *
torch.manual_seed(0); np.random.seed(0)

def prep(tag):
    gt, obs, meta = load(tag); data = []
    for port, o in obs.groupby("port"):
        g = gt[gt.port == port]
        if len(g) == 0: continue
        o = o.reset_index(drop=True)
        iv = g[["t_send", "t_end"]].values; iv = iv[np.argsort(iv[:, 0])]
        mux = bool(np.any(iv[1:, 0] < np.maximum.accumulate(iv[:-1, 1]))) if len(iv) > 1 else False
        data.append(dict(tag=tag, port=port, o=o, F=record_features(o), y=label_boundaries(o, g), true_t=np.sort(g.t_send.values), mux=mux))
    return data

def nms(t, p, thr, win=0.004):
    idx = np.where(p >= thr)[0]; keep = []
    for i in idx[np.argsort(-p[idx])]:
        if all(abs(t[i] - t[j]) > win for j in keep): keep.append(i)
    return np.sort(np.array(keep, int))

# ---------------- (1) rule
def rule_pred(d, th_len, th_gap):
    F = d["F"]; up = F[:, 0] == 1; L = np.expm1(F[:, 1]); gap = np.expm1(F[:, 4]) / 1e3
    return np.where(up & (L > th_len) & (gap > th_gap))[0]

# ---------------- (2) HSMM-style decoder
def hsmm_decode(t, p, dur_logpdf, cand_thr=0.02, max_back=60):
    c = np.where(p >= cand_thr)[0]
    if len(c) == 0: return c
    lo = np.log(np.clip(p[c], 1e-6, 1-1e-6)) - np.log(np.clip(1-p[c], 1e-6, 1))
    best = np.full(len(c), -np.inf); prev = np.full(len(c), -1)
    for i in range(len(c)):
        best[i] = lo[i]  # start a fresh chain
        for j in range(max(0, i-max_back), i):
            v = best[j] + lo[i] + dur_logpdf(t[c[i]] - t[c[j]])
            if v > best[i]: best[i], prev[i] = v, j
    # allow ending anywhere: take best, then backtrack; drop negative-gain boundaries
    i = int(np.argmax(best)); out = []
    while i >= 0: out.append(c[i]); i = prev[i]
    out = np.array(sorted(out)); return out[p[out] > 0.15] if len(out) else out

# ---------------- (3) BiTCN
class TCN(torch.nn.Module):
    def __init__(s, nin, h=48):
        super().__init__(); L = []; c = nin
        for d in (1, 2, 4, 8, 16, 1):
            L += [torch.nn.Conv1d(c, h, 5, padding=2*d, dilation=d), torch.nn.GELU()]; c = h
        s.net = torch.nn.Sequential(*L); s.out = torch.nn.Conv1d(h, 1, 1)
    def forward(s, x): return s.out(s.net(x)).squeeze(1)
def chunks(d, n=1024):
    F, y = d["F"], d["y"]
    for i in range(0, len(F), n): yield F[i:i+n], y[i:i+n]

def evaluate(name, dsets, pred_fn, tols=(0.05, 0.1, 0.25)):
    rows = []
    for d in dsets:
        idx = pred_fn(d); pt = d["o"].t.values[idx] if len(idx) else np.array([])
        r = dict(method=name, tag=d["tag"], port=d["port"], mux=d["mux"], n_true=len(d["true_t"]), n_pred=len(pt),
                 iou=seg_iou(pt, d["true_t"], d["o"].t.values[-1]))
        for tol in tols:
            tp, fp, fn = match_boundaries(pt, d["true_t"], tol*TS); r.update({f"tp{tol}": tp, f"fp{tol}": fp, f"fn{tol}": fn})
        rows.append(r)
    return rows

if __name__ == "__main__":
    tr = prep("main_seed1") + prep("main_seed2")
    tests = {t: prep(t) for t in ["main_seed3", "h1_stream", "subM1", "subM2", "subM4", "subM8", "nostream", "state", "gzip"]
             if chunks_of(t)}
    print("loaded", len(tr), {k: len(v) for k, v in tests.items()}, flush=True)
    # rule: grid search on train
    best = None
    for th_len in (64, 128, 256, 512, 1024):
        for th_gap in (0.001, 0.002, 0.004, 0.008, 0.016):
            rr = evaluate("rule", tr[::3], lambda d: rule_pred(d, th_len, th_gap))
            tp = sum(r["tp0.1"] for r in rr); fp = sum(r["fp0.1"] for r in rr); fn = sum(r["fn0.1"] for r in rr)
            f1 = 2*tp/(2*tp+fp+fn)
            if best is None or f1 > best[0]: best = (f1, th_len, th_gap)
    print("rule params", best, flush=True)
    # GBM emissions
    Xtr = np.vstack([d["F"] for d in tr]); ytr = np.concatenate([d["y"] for d in tr])
    gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=63, verbose=-1, random_state=0).fit(Xtr, ytr)
    for d in tr[::4]: d["p"] = gbm.predict_proba(d["F"])[:, 1]
    # duration prior from train ground truth
    durs = np.concatenate([np.diff(d["true_t"]) for d in tr if len(d["true_t"]) > 1]); durs = durs[durs > 0]
    hist, edges = np.histogram(np.log(durs), bins=60, density=True)
    def dur_logpdf(x):
        if x <= 0: return -20.0
        k = np.clip(np.searchsorted(edges, np.log(x)) - 1, 0, len(hist)-1); return float(np.log(hist[k] + 1e-3))
    # GBM threshold for NMS on train subset
    bthr = max(((thr, sum(r["tp0.1"] for r in evaluate("g", tr[::4], lambda d: nms(d["o"].t.values, d["p"], thr))) -
                 0.5*sum(r["fp0.1"]+r["fn0.1"] for r in evaluate("g", tr[::4], lambda d: nms(d["o"].t.values, d["p"], thr)))) for thr in (.2, .35, .5, .65)), key=lambda x: x[1])[0]
    # TCN
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    net = TCN(Xtr.shape[1]); opt = torch.optim.Adam(net.parameters(), 2e-3); pw = torch.tensor(float((ytr == 0).sum() / max(ytr.sum(), 1)) ** .5)
    ch = [c for d in tr for c in chunks(d)]
    for ep in range(6):
        np.random.shuffle(ch); tot = 0
        for F, y in ch:
            x = torch.tensor(((F - mu)/sd).T[None], dtype=torch.float32); yy = torch.tensor(y[None], dtype=torch.float32)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), yy, pos_weight=pw); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print("tcn epoch", ep, round(tot/len(ch), 4), flush=True)
    def tcn_prob(d):
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(net(torch.tensor(((F-mu)/sd).T[None], dtype=torch.float32)))[0].numpy() for F, _ in chunks(d)])
    rows = []
    for tag, ds in tests.items():
        for d in ds: d["p"] = gbm.predict_proba(d["F"])[:, 1]; d["q"] = tcn_prob(d)
        rows += evaluate("rule", ds, lambda d: rule_pred(d, best[1], best[2]))
        rows += evaluate("gbm_nms", ds, lambda d: nms(d["o"].t.values, d["p"], bthr))
        rows += evaluate("hsmm", ds, lambda d: hsmm_decode(d["o"].t.values, d["p"], dur_logpdf))
        rows += evaluate("bitcn", ds, lambda d: nms(d["o"].t.values, d["q"], .5))
        print("evaluated", tag, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/seg_per_port.csv", index=False)
    summ = []
    for (m, tag, mux), g in df.groupby(["method", "tag", "mux"]):
        r = dict(method=m, tag=tag, multiplexed=mux, n_ports=len(g))
        for tol in (0.05, 0.1, 0.25):
            f1 = boot_ratio_ci(2*g[f"tp{tol}"], 2*g[f"tp{tol}"]+g[f"fp{tol}"]+g[f"fn{tol}"])
            pr = boot_ratio_ci(g[f"tp{tol}"], g[f"tp{tol}"]+g[f"fp{tol}"]); rc = boot_ratio_ci(g[f"tp{tol}"], g[f"tp{tol}"]+g[f"fn{tol}"])
            r.update({f"F1@{int(tol*1e3)}ms": f1[0], f"F1@{int(tol*1e3)}ms_lo": f1[1], f"F1@{int(tol*1e3)}ms_hi": f1[2],
                      f"P@{int(tol*1e3)}ms": pr[0], f"R@{int(tol*1e3)}ms": rc[0]})
        r["IoU"], r["IoU_lo"], r["IoU_hi"] = boot_ci(g.iou)
        summ.append(r)
    pd.DataFrame(summ).to_csv(f"{OUT}/segmentation_summary.csv", index=False)
    json.dump(dict(rule=best, gbm_thr=bthr), open(f"{OUT}/segmentation_params.json", "w"))
    import joblib; joblib.dump(dict(gbm=gbm, rule=best, thr=bthr), f"{OUT}/segmenter_models.pkl"); torch.save(dict(net=net.state_dict(), mu=mu, sd=sd), f"{OUT}/bitcn.pt")
    print(pd.DataFrame(summ)[["method", "tag", "multiplexed", "n_ports", "F1@50ms", "F1@100ms", "F1@250ms", "IoU"]].round(3).to_string())
