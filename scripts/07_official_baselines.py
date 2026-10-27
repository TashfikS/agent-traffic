"""07 - Official baseline code on our per-turn record slices (only the input loader is ours).
  Whisper-Leak (McDonald & Or, MIT, github.com/yo-yo-yo-jbo/whisper_leak @9dc95fd):
      core/classifiers/lightgbm_classifier.LightGBMClassifier, configs/lgbm.yaml hyper-parameters
      (n_estimators 5000, learning_rate 0.02, early-stopping patience 40, max_len = 95th pct of sequence length).
      Adapter: objective='multiclass' (their default is binary; passed through their kwargs), argmax of booster probs.
  Deep Fingerprinting (Sirinam et al., github.com/deep-fingerprinting/df @38df0c1): src/Model_NoDef.DFNet imported
      unchanged through a keras-1 -> tf.keras import shim; ClosedWorld_DF_NoDef.py settings (LENGTH 5000, direction
      sequence +-1, Adamax lr 0.002, batch 128, 30 epochs).  Runs on Kaggle GPU (kaggle/df_official); this script
      exports its inputs.
Input slice per detected turn = TLS records from its boundary to the next boundary on the connection
(time_diffs = inter-record gaps, data_lengths = signed record lengths, +up / -down).  Undetected turns count as errors.
Outputs: data/parquet/slices_<tag>.parquet, results_v3/official_baselines.csv"""
import os; os.environ.setdefault("KMP_AFFINITY", "disabled")
import sys, json, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, ROOT)
from t2.data import *; from t2.turns import build_turns
from sklearn.metrics import f1_score
EDGE = ["start", "sequential", "parallel_fork_join", "transport_retry", "semantic_retry", "sub_agent"]
WL = f"{ROOT}/baselines/whisper_leak"

def slices(tag, mode="frame"):
    p = f"{PQ}/slices_{tag}_{mode}.parquet"
    if os.path.exists(p): return pd.read_parquet(p)
    T = build_turns(tag, mode); gt, obs = load(tag); rows = []
    for conn, o in obs.groupby("conn", sort=False):
        d = T[(T.conn == conn) & ~T.missed].sort_values("rec")
        if not len(d): continue
        t = o.t.values; L = np.where(o.up.values, o.plen.values, -o.plen.values); rec = d.rec.values.astype(int); ends = np.r_[rec[1:], len(t)]
        for (_, r), b, e in zip(d.iterrows(), rec, ends):
            e = min(e, b + 5000); tt = t[b:e]
            rows.append(dict(tag=tag, conn=conn, chunk=r.chunk, edge=r.get("gt_edge"), spurious=bool(r.spurious),
                             time_diffs=np.r_[0.0, np.diff(tt)].astype(np.float32), data_lengths=L[b:e].astype(np.float32)))
    S = pd.DataFrame(rows); S.to_parquet(p); return S

def missed_edges(tag, mode="frame"):
    T = build_turns(tag, mode); return T[T.missed].gt_edge.values

def score(y, yhat, n_missed_by_class):
    """macro-F1 with undetected turns appended as wrong predictions; spurious detections (edge NaN) as FP of predicted class."""
    yt = list(y) + [e for e in n_missed_by_class]; yp = list(yhat) + ["missed"] * len(n_missed_by_class)
    return f1_score(yt, yp, labels=EDGE, average="macro", zero_division=0), f1_score(yt, yp, labels=EDGE, average=None, zero_division=0)

if __name__ == "__main__":
    sys.path.insert(0, WL)
    from core.classifiers.lightgbm_classifier import LightGBMClassifier
    import yaml
    cfg = yaml.safe_load(open(f"{WL}/configs/lgbm.yaml"))
    tr = pd.concat([slices(t) for t in ("main_seed1", "main_seed2")], ignore_index=True)
    tr = tr[~tr.spurious].copy(); tr["target"] = tr.edge.map({e: i for i, e in enumerate(EDGE)}).astype(int)
    for c in ("time_diffs", "data_lengths"): tr[c] = tr[c].apply(list)
    rng = np.random.default_rng(0); conns = tr.conn.unique(); val_c = set(rng.choice(conns, int(.1 * len(conns)), replace=False))
    va = tr[tr.conn.isin(val_c)]; trn = tr[~tr.conn.isin(val_c)]
    max_len = int(np.percentile(trn.data_lengths.apply(len), 95))
    clf = LightGBMClassifier({"max_len": max_len}, objective="multiclass", metric="multi_logloss", num_class=len(EDGE), **cfg["model"]["params"])
    clf.fit(trn, va, patience=cfg["training"]["patience"])
    rows = []
    for tag in ("main_seed3", "failfast", "subM8"):
        if not chunks_of(tag): continue
        te = slices(tag); te = te.copy()
        for c in ("time_diffs", "data_lengths"): te[c] = te[c].apply(list)
        P = clf.model.predict_proba(clf._prepare_features(te)); yhat = np.array(EDGE)[P.argmax(1)]
        y = te.edge.fillna("spurious").values
        mf, per = score(y, yhat, missed_edges(tag))
        rows.append(dict(tag=tag, model="whisperleak_official_lgbm", max_len=max_len, n=len(te), macroF1=mf, **{f"F1_{e}": v for e, v in zip(EDGE, per)}))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R.to_csv(f"{RES}/official_baselines.csv", index=False); print(R.round(3).to_string())
    # export DF inputs (direction sequences, LENGTH 5000) for the Kaggle GPU run
    out = f"{ROOT}/kaggle/df_official/data"; os.makedirs(out, exist_ok=True)
    for name, df in (("train", tr), *[(t, slices(t)) for t in ("main_seed3", "failfast", "subM8") if chunks_of(t)]):
        X = np.zeros((len(df), 5000), np.int8)
        for i, L in enumerate(df.data_lengths.values): L = np.sign(np.asarray(L))[:5000]; X[i, :len(L)] = L
        y = df.edge.fillna("spurious").values
        np.savez_compressed(f"{out}/{name}.npz", X=X, y=y, conn=df.conn.values, missed=missed_edges(name) if name != "train" else np.array([]))
    print("exported DF inputs")
