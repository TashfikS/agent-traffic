"""13 - Per-corpus Mondrian split-conformal quantile regression on real stacks.
Quantile LightGBM (5/50/95%) trained once on real_main even episodes (features of 09).  Each corpus is a Mondrian
group, further split by an observable size bucket (predicted median < 1 KB vs >= 1 KB): conformity scores from that
corpus' EVEN episodes calibrate the interval; coverage is measured on its ODD episodes (1,000x episode bootstrap CI).
Outputs: results_v3/cqr_percorpus.csv"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled"); os.environ.setdefault("T2_TS", "1"); os.environ.setdefault("T2_LABEL_WIN", "1.0")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
spec = importlib.util.spec_from_file_location("s09", f"{ROOT}/scripts/09_real_eval.py"); S9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S9)
from t2.data import RES
from sklearn.linear_model import HuberRegressor
CORP = [t for t in S9.TAGS if not t.endswith("_smoke")]
base = S9.turns("real_main"); tr = base[(base.gt_ep % 2 == 0) & base.tool & base.h_dreq.notna()]
H = HuberRegressor(max_iter=3000).fit(np.c_[tr.h_dreq, tr.h_a_hat, tr.h_ndown] / 1e3, tr.gt_tool_bytes_added / 1e3)
X = lambda d: S9.slog(np.c_[d[S9.FEAT].fillna(0).values, H.predict(np.c_[d.h_dreq, d.h_a_hat, d.h_ndown] / 1e3) * 1e3])
Q = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=400, learning_rate=.04, num_leaves=15, min_child_samples=10, verbose=-1, random_state=0).fit(X(tr), np.log(tr.gt_tool_bytes_added)) for a in (.05, .5, .95)}
def qhat(E, a=.1): n = len(E); return np.quantile(E, min(1, np.ceil((n + 1) * (1 - a)) / n)) if n else np.inf
rows = []
for t in CORP:
    d = S9.turns(t); d = d[d.tool & d.h_dreq.notna()].copy()
    if len(d) < 60: continue
    y = np.log(d.gt_tool_bytes_added.values.astype(float)); lo, md, hi = (Q[a].predict(X(d)) for a in (.05, .5, .95))
    E = np.maximum(lo - y, y - hi); grp = (md >= np.log(1e3)).astype(int)
    if t == "real_main":   # even episodes trained the quantile model -> calibrate on odd ep%4==1, test on ep%4==3
        cal = (d.gt_ep.values % 4 == 1); te = (d.gt_ep.values % 4 == 3)
    else: cal = (d.gt_ep.values % 2 == 0); te = ~cal
    for scheme in ("global_real_main", "per_corpus", "per_corpus_x_size"):
        q = np.zeros(len(d))
        if scheme == "global_real_main":
            b = base[(base.gt_ep % 4 == 1) & base.tool & base.h_dreq.notna()]; yb = np.log(b.gt_tool_bytes_added.values.astype(float))
            q[:] = qhat(np.maximum(Q[.05].predict(X(b)) - yb, yb - Q[.95].predict(X(b))))
        elif scheme == "per_corpus": q[:] = qhat(E[cal])
        else:
            for gg in (0, 1): q[grp == gg] = qhat(E[cal & (grp == gg)]) if (cal & (grp == gg)).sum() >= 20 else qhat(E[cal])
        cov = (y >= lo - q) & (y <= hi + q); ep = d.gt_ep.values
        eps_ = np.unique(ep[te]); rr = np.random.default_rng(0); bs = []
        for _ in range(1000):
            s_ = rr.choice(eps_, len(eps_)); m_ = np.concatenate([np.where((ep == e) & te)[0] for e in s_]); bs.append(cov[m_].mean())
        for reg, mk in (("all", te), ("<1KB", te & (y < np.log(1e3))), (">=1KB", te & (y >= np.log(1e3)))):
            if mk.sum() >= 10: rows.append(dict(corpus=t, scheme=scheme, regime=reg, n_test=int(mk.sum()), n_cal=int(cal.sum()), PICP90=float(cov[mk].mean()),
                                               lo=float(np.percentile(bs, 2.5)) if reg == "all" else np.nan, hi=float(np.percentile(bs, 97.5)) if reg == "all" else np.nan,
                                               width_ratio_med=float(np.median(np.exp(hi[mk] + q[mk]) / np.exp(lo[mk] - q[mk])))))
R = pd.DataFrame(rows); R.to_csv(f"{RES}/cqr_percorpus.csv", index=False)
print(R[R.regime == "all"].pivot(index="corpus", columns="scheme", values="PICP90").round(3).to_string())
print(R[(R.scheme == "per_corpus_x_size")][["corpus", "regime", "n_test", "PICP90", "lo", "hi", "width_ratio_med"]].round(3).to_string())
