"""02 - Log-free turn segmentation on dedicated (M=1) and multiplexed HTTP/2 connections (M in {2,4,8}).

Methods
  rule_paper     : paper baseline - upstream record > tau_L after upstream silence > tau_g (grid-searched on train)
  gbm_nms_paper  : paper GBM over the 7 per-record features + 4 ms NMS, trained with the paper's labeller
  frame_rule     : HTTP/2/TLS frame-aware rule - every upstream application record > 64 B that is NOT the
                   continuation of a full-size (16384 B) record of the same burst starts a request.
                   No silence gap, no NMS -> concurrent requests flushed in the same read are all kept.
  frame_gbm      : LightGBM over frame-aware candidate features (size, full-record continuation, same-timestamp
                   group size/rank, downstream activity between candidates, H2+TLS overhead residues), corrected labels
  frame_dct      : dilated-conv + transformer encoder over the candidate sequence (same features), corrected labels
Train: main_seed1 + main_seed2 + subM{2,4,8}b ; Test: main_seed3, subM1, subM2, subM4, subM8 (disjoint seeds).
Outputs: results_v3/seg_summary.csv, seg_per_conn.parquet, seg_coalescing_stress.csv, seg_label_audit.csv,
         figures_v3/fig_seg_multiplex.pdf
"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled")
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from t2.data import *
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
TRAIN = os.environ.get("T2_TRAIN", "main_seed1,main_seed2,subM2b,subM4b,subM8b").split(",")
TEST = os.environ.get("T2_TEST", "main_seed3,subM1,subM2,subM4,subM8").split(",")
TOLS = (0.05, 0.1, 0.25)          # real-time-equivalent seconds; compared on compressed clock as tol*TS
from t2.seg import *

def prep(tag):
    gt, obs = load(tag); out = []
    for conn, o, g in conn_groups(gt, obs):
        iv = g[["t_send", "t_end"]].values; iv = iv[np.argsort(iv[:, 0])]
        conc = int(max(((iv[:, 0] <= x) & (iv[:, 1] > x)).sum() for x in iv[:, 0])) if len(iv) else 0
        c, Fc = frame_candidates(o)
        asg = assign_sizeaware(o, g); ts = g.t_send.values.copy(); wire = ts.copy()
        for ri, pos in asg.items(): wire[pos] = o.t.values[ri]
        out.append(dict(wire_t=np.sort(wire), lag=wire[list(asg.values())] - ts[list(asg.values())], tag=tag, conn=conn, chunk=int(g.chunk.iloc[0]), o=o, true_t=np.sort(g.t_send.values), mux=conc > 1, maxconc=conc,
                        y_paper=labels_paper(o, g), y_sa=labels_sizeaware(o, g), c=c, Fc=Fc))
    return out

# ---------------------------------------------------------------- decoders / eval
def nms(t, p, thr, win=0.004):
    idx = np.where(p >= thr)[0]; keep = []
    for i in idx[np.argsort(-p[idx])]:
        if all(abs(t[i] - t[j]) > win for j in keep): keep.append(i)
    return np.sort(np.array(keep, int))
def rule_paper(d, th_len, th_gap):
    F = d["Fp"]; return np.where((F[:, 0] == 1) & (np.expm1(F[:, 1]) > th_len) & (np.expm1(F[:, 4]) / 1e3 > th_gap))[0]

def match(pred_t, true_t, tol):
    pred_t = np.sort(pred_t); used = np.zeros(len(pred_t), bool); tp = 0
    for tt in np.sort(true_t):
        lo, hi = np.searchsorted(pred_t, tt - tol), np.searchsorted(pred_t, tt + tol, "right")
        cand = [j for j in range(lo, hi) if not used[j]]
        if cand: j = min(cand, key=lambda j: abs(pred_t[j] - tt)); used[j] = True; tp += 1
    return tp, len(pred_t) - tp, len(true_t) - tp

CLOCK = "app"   # app: truth = client t_send (paper protocol); wire: truth = time the request's first record crossed the observer
def evaluate(name, ds, fn, clocks=None):
    rows = []
    for d in ds:
        idx = fn(d); pt = d["o"].t.values[idx] if len(idx) else np.array([])
        for clk in (clocks or [CLOCK]):
            tt = d["true_t"] if clk == "app" else d["wire_t"]
            r = dict(method=name, clock=clk, tag=d["tag"], conn=d["conn"], mux=d["mux"], maxconc=d["maxconc"], n_true=len(tt), n_pred=len(pt))
            for tol in TOLS:
                tp, fp, fn_ = match(pt, tt, tol * TS); r.update({f"tp{tol}": tp, f"fp{tol}": fp, f"fn{tol}": fn_})
            rows.append(r)
    return rows
def f1_of(rows, tol=0.1):
    tp = sum(r[f"tp{tol}"] for r in rows); fp = sum(r[f"fp{tol}"] for r in rows); fn = sum(r[f"fn{tol}"] for r in rows)
    return 2 * tp / max(2 * tp + fp + fn, 1)

# ---------------------------------------------------------------- dilated-conv + transformer
class DCT(torch.nn.Module):
    def __init__(s, nin, h=64):
        super().__init__(); s.inp = torch.nn.Linear(nin, h); L = []
        for dil in (1, 2, 4, 8): L += [torch.nn.Conv1d(h, h, 3, padding=dil, dilation=dil), torch.nn.GELU()]
        s.conv = torch.nn.Sequential(*L)
        s.tr = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(h, 4, 128, 0.1, batch_first=True), 2)
        s.out = torch.nn.Linear(h, 1)
    def forward(s, x):
        h = s.inp(x); h = h + s.conv(h.transpose(1, 2)).transpose(1, 2); return s.out(s.tr(h)).squeeze(-1)

def windows(F, y, n=256, step=192):
    for i in range(0, max(len(F) - n, 0) + 1, step): yield F[i:i+n], y[i:i+n]
    if len(F) > n and (len(F) - n) % step: yield F[-n:], y[-n:]

def dct_prob(net, mu, sd, F, n=256):
    if not len(F): return np.zeros(0)
    p = np.zeros(len(F)); w = np.zeros(len(F))
    with torch.no_grad():
        for i in list(range(0, max(len(F) - n, 0) + 1, 128)) + ([len(F) - n] if len(F) > n else []):
            x = torch.tensor(((F[i:i+n] - mu) / sd)[None], dtype=torch.float32)
            p[i:i+n] += torch.sigmoid(net(x))[0].numpy(); w[i:i+n] += 1
    return p / np.maximum(w, 1)

if __name__ == "__main__":
    t0 = time.time()
    tr = sum([prep(t) for t in TRAIN if chunks_of(t)], []); te = {t: prep(t) for t in TEST if chunks_of(t)}
    print("loaded", len(tr), {k: len(v) for k, v in te.items()}, round(time.time() - t0), "s", flush=True)
    for d in tr + sum(te.values(), []): d["Fp"] = paper_record_features(d["o"])
    # label audit: how many GT requests does each labeller retain?
    aud = []
    for tag, ds in [("train", tr)] + list(te.items()):
        for mux in (False, True):
            s = [d for d in ds if d["mux"] == mux]
            if s: aud.append(dict(set=tag, multiplexed=mux, n_requests=sum(len(d["true_t"]) for d in s),
                                  paper_labels=int(sum(d["y_paper"].sum() for d in s)), sizeaware_labels=int(sum(d["y_sa"].sum() for d in s))))
    pd.DataFrame(aud).to_csv(f"{RES}/seg_label_audit.csv", index=False); print(pd.DataFrame(aud).to_string(), flush=True)
    # (1) paper rule grid (train subset)
    sub = tr[::3]; best = max(((f1_of(evaluate("r", sub, lambda d: rule_paper(d, a, b))), a, b) for a in (64, 128, 256, 512, 1024)
                               for b in (0.001, 0.002, 0.004, 0.008, 0.016)))
    print("paper rule", best, flush=True)
    # (2) paper GBM+NMS: all records, paper labels; negatives subsampled 1:20 for tractability
    rng = np.random.default_rng(0); X, Y = [], []
    for d in tr:
        y = d["y_paper"]; keep = (y == 1) | (rng.random(len(y)) < 0.05); X.append(d["Fp"][keep]); Y.append(y[keep])
    gbm_p = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=63, verbose=-1, random_state=0).fit(np.vstack(X), np.concatenate(Y))
    for d in tr[::4]: d["pp"] = gbm_p.predict_proba(d["Fp"])[:, 1]
    thr_p = max((.2, .35, .5, .65, .8), key=lambda th: f1_of(evaluate("g", tr[::4], lambda d: nms(d["o"].t.values, d["pp"], th))))
    # (3) frame GBM on candidates with corrected labels
    Xc = np.vstack([d["Fc"] for d in tr]); Yc = np.concatenate([d["y_sa"][d["c"]] for d in tr])
    gbm_f = lgb.LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=31, verbose=-1, random_state=0).fit(Xc, Yc)
    val = tr[1::5]
    for d in val: d["pf"] = gbm_f.predict_proba(d["Fc"])[:, 1] if len(d["c"]) else np.zeros(0)
    thr_f = max((.3, .4, .5, .6, .7), key=lambda th: f1_of(evaluate("f", val, lambda d: d["c"][d["pf"] >= th])))
    # (4) DCT
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-6; net = DCT(Xc.shape[1]); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    W = [(F, d["y_sa"][d["c"]]) for d in tr for F, _ in [(d["Fc"], None)] if len(F)]
    W = [w for F, y in W for w in windows(F, y)]
    for ep in range(8):
        rng.shuffle(W); tot = 0; net.train()
        for i in range(0, len(W), 16):
            B = W[i:i+16]; n = max(len(b[0]) for b in B)
            x = np.zeros((len(B), n, Xc.shape[1]), np.float32); yy = np.zeros((len(B), n), np.float32); m = np.zeros((len(B), n), np.float32)
            for k, (F, y) in enumerate(B): x[k, :len(F)] = (F - mu) / sd; yy[k, :len(y)] = y; m[k, :len(y)] = 1
            lo = net(torch.tensor(x)); l = (torch.nn.functional.binary_cross_entropy_with_logits(lo, torch.tensor(yy), reduction="none") * torch.tensor(m)).sum() / m.sum()
            opt.zero_grad(); l.backward(); opt.step(); tot += l.item()
        print("dct epoch", ep, round(tot / max(len(W) / 16, 1), 4), flush=True)
    net.eval()
    for d in val: d["pd"] = dct_prob(net, mu, sd, d["Fc"])
    thr_d = max((.3, .4, .5, .6, .7), key=lambda th: f1_of(evaluate("d", val, lambda d: d["c"][d["pd"] >= th])))
    json.dump(dict(paper_rule=best, gbm_nms_thr=thr_p, frame_gbm_thr=thr_f, frame_dct_thr=thr_d), open(f"{RES}/seg_params.json", "w"))
    # ------------------------------------------------ test
    rows = []
    for tag, ds in te.items():
        for d in ds:
            d["pp"] = gbm_p.predict_proba(d["Fp"])[:, 1]
            d["pf"] = gbm_f.predict_proba(d["Fc"])[:, 1] if len(d["c"]) else np.zeros(0); d["pd"] = dct_prob(net, mu, sd, d["Fc"])
        rows += evaluate("rule_paper", ds, lambda d: rule_paper(d, best[1], best[2]), clocks=["app", "wire"])
        rows += evaluate("gbm_nms_paper", ds, lambda d: nms(d["o"].t.values, d["pp"], thr_p), clocks=["app", "wire"])
        rows += evaluate("frame_rule", ds, frame_rule, clocks=["app", "wire"])
        rows += evaluate("frame_gbm", ds, lambda d: d["c"][d["pf"] >= thr_f], clocks=["app", "wire"])
        rows += evaluate("frame_dct", ds, lambda d: d["c"][d["pd"] >= thr_d], clocks=["app", "wire"])
        print("evaluated", tag, flush=True)
    R = pd.DataFrame(rows); R.to_parquet(f"{RES}/seg_per_conn.parquet")
    summ = []
    for (m, clk, tag, mux), g in R.groupby(["method", "clock", "tag", "mux"]):
        r = dict(method=m, clock=clk, tag=tag, multiplexed=mux, n_conn=len(g), n_req=int(g.n_true.sum()))
        for tol in TOLS:
            f = boot_ratio_ci(2 * g[f"tp{tol}"], 2 * g[f"tp{tol}"] + g[f"fp{tol}"] + g[f"fn{tol}"])
            ms = int(tol * 1e3); r.update({f"F1@{ms}": f[0], f"F1@{ms}_lo": f[1], f"F1@{ms}_hi": f[2],
                                            f"P@{ms}": g[f"tp{tol}"].sum() / max((g[f"tp{tol}"] + g[f"fp{tol}"]).sum(), 1),
                                            f"R@{ms}": g[f"tp{tol}"].sum() / max((g[f"tp{tol}"] + g[f"fn{tol}"]).sum(), 1)})
        summ.append(r)
    S = pd.DataFrame(summ); S.to_csv(f"{RES}/seg_summary.csv", index=False)
    print(S[["method", "clock", "tag", "multiplexed", "n_conn", "F1@50", "F1@100", "F1@100_lo", "F1@100_hi", "F1@250"]].round(3).to_string(), flush=True)
    lag = []
    for tag, ds in te.items():
        for mux in (False, True):
            v = np.concatenate([d["lag"] for d in ds if d["mux"] == mux] or [np.zeros(0)]) / TS * 1e3
            if len(v): lag.append(dict(tag=tag, multiplexed=mux, n=len(v), **{f"lag_p{q}_ms": float(np.percentile(v, q)) for q in (50, 75, 90, 95, 99)},
                                       frac_gt_50ms=float((v > 50).mean()), frac_gt_100ms=float((v > 100).mean())))
    pd.DataFrame(lag).to_csv(f"{RES}/seg_observer_lag.csv", index=False); print(pd.DataFrame(lag).round(3).to_string(), flush=True)
    # ------------------------------------------------ stress: TLS write coalescing (middlebox / kernel / CDN re-framing)
    st = []
    for p in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        rr = np.random.default_rng(1); res_ = {"frame_rule": [], "frame_gbm": []}
        for d in te.get(os.environ.get("T2_STRESS", "subM8"), []):
            o = d["o"].copy(); c = d["c"]; tc = o.t.values[c]
            same = np.r_[False, tc[1:] == tc[:-1]] & (rr.random(len(c)) < p)
            # merge record c[k] into c[k-1] when same-read & coin; sizes add, record removed
            keep = np.ones(len(o), bool); L = o.plen.values.copy()
            for k in np.where(same)[0][::-1]: L[c[k-1]] = min(L[c[k-1]] + L[c[k]], MAXREC); keep[c[k]] = False
            o["plen"] = L; o = o[keep].reset_index(drop=True); cc, Fc = frame_candidates(o); dd = dict(d, o=o, c=cc, Fc=Fc)
            res_["frame_rule"] += evaluate("x", [dd], frame_rule, clocks=["wire"])
            pf = gbm_f.predict_proba(Fc)[:, 1] if len(cc) else np.zeros(0); res_["frame_gbm"] += evaluate("x", [dd], lambda _: cc[pf >= thr_f], clocks=["wire"])
        for k, v in res_.items(): st.append(dict(method=k, p_coalesce=p, F1_100=f1_of([r for r in v if r["mux"]], 0.1), F1_250=f1_of([r for r in v if r["mux"]], 0.25)))
    pd.DataFrame(st).to_csv(f"{RES}/seg_coalescing_stress.csv", index=False); print(pd.DataFrame(st).round(3).to_string())
    print("total", round(time.time() - t0), "s")
