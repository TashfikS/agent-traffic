"""ANT (benzenenene/ANT, HF) -> TLS-record parquet for LLM-API flows + per-episode SNI summary. CPU job."""
import os, subprocess, time
for i in range(8):
    if subprocess.run("pip -q install dpkt huggingface_hub pyarrow", shell=True).returncode == 0: break
    time.sleep(20)
import dpkt, socket, numpy as np, pandas as pd
def _sni(b):
    try:
        if b[0] != 22 or b[5] != 1: return None
        p = 9 + 2 + 32; p += 1 + b[p]; p += 2 + int.from_bytes(b[p:p + 2], "big"); p += 1 + b[p]
        end = p + 2 + int.from_bytes(b[p:p + 2], "big"); p += 2
        while p + 4 <= end:
            t = int.from_bytes(b[p:p + 2], "big"); l = int.from_bytes(b[p + 2:p + 4], "big")
            if t == 0: return b[p + 9:p + 4 + l].decode(errors="ignore")
            p += 4 + l
    except Exception: return None
def pcap_records(path, port=443):
    flows, rows, sni = {}, [], {}
    with open(path, "rb") as f:
        try: rd = dpkt.pcap.Reader(f)
        except ValueError: f.seek(0); rd = dpkt.pcapng.Reader(f)
        lt = rd.datalink()
        for ts, buf in rd:
            try:
                if lt == 113: ip = dpkt.sll.SLL(buf).data
                elif lt == 276: ip = dpkt.sll2.SLL2(buf).data
                elif lt in (101, 12, 14): ip = dpkt.ip.IP(buf) if buf[0] >> 4 == 4 else dpkt.ip6.IP6(buf)
                elif lt == 0: ip = dpkt.loopback.Loopback(buf).data
                else: ip = dpkt.ethernet.Ethernet(buf).data
                if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)) or not isinstance(ip.data, dpkt.tcp.TCP): continue
                tcp = ip.data
            except Exception: continue
            if port not in (tcp.sport, tcp.dport): continue
            up = tcp.dport == port; cport = tcp.sport if up else tcp.dport
            cid = (socket.inet_ntop(socket.AF_INET if isinstance(ip, dpkt.ip.IP) else socket.AF_INET6, ip.dst if up else ip.src), cport)
            st = flows.setdefault((cid, up), {"next": None, "buf": b"", "ooo": {}})
            if tcp.flags & dpkt.tcp.TH_SYN: st["next"] = (tcp.seq + 1) & 0xffffffff; continue
            data = bytes(tcp.data)
            if not data: continue
            if st["next"] is None: st["next"] = tcp.seq          # capture started mid-flow
            seq = tcp.seq; d = (seq - st["next"]) & 0xffffffff
            if d >= 2**31:
                cut = (st["next"] - seq) & 0xffffffff
                if cut >= len(data): continue
                data = data[cut:]; seq = st["next"]
            elif d > 0:
                if d < 2**24: st["ooo"][seq] = data
                continue
            st["buf"] += data; st["next"] = (seq + len(data)) & 0xffffffff
            while st["next"] in st["ooo"]:
                x = st["ooo"].pop(st["next"]); st["buf"] += x; st["next"] = (st["next"] + len(x)) & 0xffffffff
            b = st["buf"]
            while len(b) >= 5:
                if b[0] not in (20, 21, 22, 23) or b[1] != 3: b = b""; break     # desync -> drop buffer
                ln = int.from_bytes(b[3:5], "big")
                if len(b) < 5 + ln: break
                if up and b[0] == 22 and cid not in sni: sni[cid] = _sni(b[:5 + ln])
                rows.append((cid[0], cid[1], up, ts, b[0], ln)); b = b[5 + ln:]
            st["buf"] = b
    df = pd.DataFrame(rows, columns=["server", "cport", "up", "t", "ctype", "len"])
    df["sni"] = [sni.get((s, p)) for s, p in zip(df.server, df.cport)]
    return df
import re, json, sys, traceback
from multiprocessing import Pool
from huggingface_hub import HfApi, hf_hub_download
REPO = "benzenenene/ANT"; W = "/kaggle/working"; TMP = "/kaggle/tmp"; os.makedirs(TMP, exist_ok=True)
SPLITS = os.environ.get("ANT_SPLITS", "SPLITS_PLACEHOLDER").split(",")
LLM = re.compile(r"(^api\.|openai|anthropic|deepseek|dashscope|aliyun|qwen|mimo|xiaomi|chatgpt|moonshot|bigmodel|volces|siliconflow|openrouter|generativelanguage)", re.I)
def one(p):
    for att in range(4):
        try:
            lp = hf_hub_download(REPO, p, repo_type="dataset", cache_dir=TMP)
            d = pcap_records(lp); ep = os.path.dirname(p)
            summ = d.groupby(d.sni.fillna("?")).agg(n=("len", "size"), byt=("len", "sum"), flows=("cport", "nunique")).reset_index().assign(ep=ep)
            k = d[d.sni.fillna("").str.contains(LLM)].assign(ep=ep)
            try: os.remove(os.path.realpath(lp))
            except Exception: pass
            return k, summ, None
        except Exception as e: err = repr(e)[:300]; time.sleep(5 + 10 * att)
    return None, None, (p, err)
if __name__ == "__main__":
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    msgs = [f for f in files if f.endswith("messages.jsonl") and f.split("/")[0] in SPLITS]
    pcaps = [f for f in files if f.endswith("network.pcap") and f.split("/")[0] in SPLITS]
    print(len(pcaps), "pcaps", flush=True)
    M = []
    for f in msgs:
        try:
            for line in open(hf_hub_download(REPO, f, repo_type="dataset", cache_dir=TMP)):
                r = json.loads(line); M.append({k: r.get(k) for k in ("task_id", "agent", "category", "split", "role", "attack_primitive", "normalized_attack_primitive", "macro_group_label", "message_index")} | {"ep": os.path.dirname(f)})
        except Exception: pass
    pd.DataFrame(M).to_parquet(f"{W}/ant_messages_{'_'.join(SPLITS)}.parquet")
    K, S, E = [], [], []; t0 = time.time()
    with Pool(4) as pool:
        for i, (k, s, e) in enumerate(pool.imap_unordered(one, pcaps, chunksize=2)):
            if e: E.append(e)
            else: K.append(k); S.append(s)
            if i % 100 == 0: print(i, round(time.time() - t0), len(E), flush=True)
    tag = "_".join(SPLITS)
    pd.concat(K).to_parquet(f"{W}/ant_llm_records_{tag}.parquet"); pd.concat(S).to_parquet(f"{W}/ant_sni_summary_{tag}.parquet")
    json.dump(E, open(f"{W}/ant_errors_{tag}.json", "w")); print("DONE", len(K), len(E), round(time.time() - t0), flush=True)
