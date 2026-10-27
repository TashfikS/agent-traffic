"""14 - Tool identification on real agent traffic (LangGraph / AutoGen / scripted Kaggle agent).
Single-tool turns only; observer features of the turn that carries the tool result (request growth, gap = tool
latency, previous response size = argument length proxy, previous gap) plus the previous turn's predicted tool
(Markov prior, two-stage).  Closed set: episode-grouped 5-fold accuracy.  Open set: 2 of the tools held out
(all pairs), unknown-tool AUROC with max-softmax, energy and Mahalanobis scores.  Outputs: results_v3/tool_id_real.csv"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled"); os.environ.setdefault("T2_TS", "1"); os.environ.setdefault("T2_LABEL_WIN", "1.0")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys, itertools, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
spec = importlib.util.spec_from_file_location("s09", f"{ROOT}/scripts/09_real_eval.py"); S9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S9)
from t2.data import RES
from sklearn.metrics import roc_auc_score; from sklearn.model_selection import GroupKFold
F = ["h_dreq", "gap", "h_a_hat", "h_ndown", "req", "ttft", "dn_med", "n_down", "down"]
rows = []
for t in ["fw_langgraph", "fw_autogen", "real_main"]:
    d = S9.turns(t).sort_values("t0"); d = d[(d.gt_n_tools_added == 1) & (d.gt_n_err_added == 0) & d.h_dreq.notna()].copy()
    d["tool"] = d.gt_tools_added.astype(str); d = d[d.tool != "subagent"]; tools = sorted(d.tool.unique())
    X = d[F].fillna(-1).values; y = d.tool.values; g = d.gt_ep.values
    acc = []
    for tr, te in GroupKFold(5).split(X, y, g):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15, verbose=-1, random_state=0).fit(X[tr], y[tr]); acc.append((m.predict(X[te]) == y[te]).mean())
    au = {"msp": [], "energy": [], "maha": []}
    for unk in itertools.combinations(tools, 2):
        ku = np.isin(y, unk); rng = np.random.default_rng(0); te_ep = set(rng.choice(np.unique(g), len(np.unique(g)) // 3, replace=False)); tem = np.isin(g, list(te_ep))
        trm = ~tem & ~ku; m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15, verbose=-1, random_state=0).fit(X[trm], y[trm])
        raw = m.predict(X[tem], raw_score=True); raw = np.c_[np.zeros(len(raw)), raw] if raw.ndim == 1 else raw; P = m.predict_proba(X[tem]); lab = ku[tem]
        if lab.sum() < 5 or (~lab).sum() < 5: continue
        au["msp"].append(roc_auc_score(lab, 1 - P.max(1))); au["energy"].append(roc_auc_score(lab, -np.log(np.exp(raw).sum(1))))
        Z = (X - X[trm].mean(0)) / (X[trm].std(0) + 1e-9); mu = {c: Z[trm & (y == c)].mean(0) for c in np.unique(y[trm])}
        cov = np.cov(Z[trm].T) + 1e-3 * np.eye(Z.shape[1]); iv = np.linalg.inv(cov)
        md = np.min([np.einsum("ij,jk,ik->i", Z[tem] - mu[c], iv, Z[tem] - mu[c]) for c in mu], axis=0); au["maha"].append(roc_auc_score(lab, md))
    rows.append(dict(corpus=t, n=len(d), n_tools=len(tools), chance=1 / len(tools), closed_acc=float(np.mean(acc)), closed_acc_sd=float(np.std(acc)),
                     **{f"AUROC_{k}": float(np.mean(v)) for k, v in au.items()}, n_splits=len(au["msp"])))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/tool_id_real.csv", index=False)
