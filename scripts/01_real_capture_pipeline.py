"""01 - Real-stack capture on Kaggle (GPU T4x2 / P100), $0.  NOT executed in the local audit; run in a Kaggle notebook
with Internet ON.  Produces data/raw_real/{gt,obs}_<tag>_c<chunk>.csv in the SAME schema as testbed2.py, so scripts
02-04 run unchanged (set T2_RAW=/kaggle/working/raw_real).

Stages (each resumable; ~12 h session budget):
  serve   : vLLM OpenAI-compatible server (Qwen/Qwen2.5-7B-Instruct-AWQ, tool-calling parser hermes) on 127.0.0.1:8000
  tunnel  : cloudflared quick tunnel -> https://<random>.trycloudflare.com  (Cloudflare edge terminates TLS 1.3 + h2,
            so the observer sees real edge framing, record coalescing, WAN jitter)
  capture : tcpdump on the CLIENT side (traffic leaves the VM to the Cloudflare edge and returns), SSLKEYLOGFILE for
            ground truth only (decryption is used to label, never as observer input)
  agents  : LangGraph (chat-completions schema) and AutoGen (messages-style) agents with the 12 local tools, scripted
            tasks for the 6 motifs (sequential, parallel, transport retry via injected 529 proxy, semantic retry,
            sub-agent pooled, sub-agent new connection)
  convert : tshark -> per-record (conn, dir, t, ctype, len) observer table; decrypted h2 stream map -> GT turns
  ant     : adapter for the ANT agent-behaviour dataset (ref [11]) - path supplied by the user; see ant_adapter()
  freetier: optional live-provider validation (Gemini / Groq / GitHub Models / OpenRouter free tiers) - respect each
            provider's ToS and rate limits; tool results are local so no third-party data is uploaded beyond prompts.
usage:  python 01_real_capture_pipeline.py serve|tunnel|run --framework langgraph|autogen --episodes N --tag TAG
        python 01_real_capture_pipeline.py convert --tag TAG ;  python 01_real_capture_pipeline.py ant --path DIR
"""
import os, sys, json, time, subprocess, argparse, random, re, shutil, signal
W = os.environ.get("KAGGLE_WORKING", "/kaggle/working"); RAW = f"{W}/raw_real"; os.makedirs(RAW, exist_ok=True)
MODEL = os.environ.get("T2_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
KEYLOG = f"{W}/sslkeys.log"; URLF = f"{W}/tunnel_url.txt"

def sh(cmd, **kw): print("+", cmd, flush=True); return subprocess.run(cmd, shell=True, check=True, **kw)

def serve():
    sh("pip -q install 'vllm>=0.6' langgraph langchain-openai autogen-agentchat autogen-ext[openai] httpx[http2] pyshark")
    sh("apt-get -qq update && apt-get -qq install -y tcpdump tshark >/dev/null")
    log = open(f"{W}/vllm.log", "w")
    subprocess.Popen(["python", "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--host", "127.0.0.1", "--port", "8000",
                      "--quantization", "awq", "--dtype", "half", "--max-model-len", "16384", "--tensor-parallel-size", os.environ.get("T2_TP", "2"),
                      "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--enable-prefix-caching"], stdout=log, stderr=log)
    for _ in range(600):
        try:
            import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2); print("vLLM up"); return
        except Exception: time.sleep(2)
    raise RuntimeError("vLLM did not start; see vllm.log")

def tunnel():
    if not shutil.which("cloudflared"):
        sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared")
    log = open(f"{W}/cloudflared.log", "w"); subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"], stdout=log, stderr=log)
    for _ in range(120):
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(f"{W}/cloudflared.log").read())
        if m: open(URLF, "w").write(m.group(0)); print("tunnel", m.group(0)); return m.group(0)
        time.sleep(1)
    raise RuntimeError("no tunnel URL")

# ------------------------------------------------------------------ tools (same 12 names/size regimes as the testbed, real work)
def make_tools(r):
    import base64, glob, subprocess as sp
    def web_search(q: str) -> str:  return json.dumps([{"id": i, "title": q[:40], "text": " ".join(r.choice(WORDS) for _ in range(r.randint(20, 80)))} for i in range(r.randint(3, 8))])
    def url_fetch(q: str) -> str:   return " ".join(r.choice(WORDS) for _ in range(int(10 ** r.gauss(3.3, .5))))
    def sql_query(q: str) -> str:   return json.dumps([{"row": i, "v": r.random()} for i in range(int(10 ** r.gauss(1.4, .6)))])
    def file_read(q: str) -> str:   fs = glob.glob("/usr/lib/python3*/**/*.py", recursive=True); return open(r.choice(fs), errors="ignore").read()[: int(10 ** r.gauss(3.6, .6))]
    def code_exec(q: str) -> str:   return sp.run([sys.executable, "-c", "import random;print(sum(random.random() for _ in range(10**5)))"], capture_output=True, text=True).stdout
    def api_call(q: str) -> str:    return json.dumps({"status": "ok", "data": [r.random() for _ in range(int(10 ** r.gauss(1.8, .4)))]})
    def calculator(q: str) -> str:  return str(r.random() * 1e6)
    def grep_repo(q: str) -> str:   return sp.run("grep -rn import /usr/lib/python3*/json | head -%d" % r.randint(5, 80), shell=True, capture_output=True, text=True).stdout
    def list_dir(q: str) -> str:    return "\n".join(os.listdir("/usr/lib")[: r.randint(5, 60)])
    def image_read(q: str) -> str:  return base64.b64encode(os.urandom(int(10 ** r.gauss(4.5, .3) * .75))).decode()
    def vector_search(q: str) -> str: return json.dumps([{"doc": i, "score": r.random(), "text": " ".join(r.choice(WORDS) for _ in range(60))} for i in range(r.randint(3, 10))])
    def shell(q: str) -> str:       return sp.run("ps aux | head -%d" % r.randint(3, 40), shell=True, capture_output=True, text=True).stdout
    return [web_search, url_fetch, sql_query, file_read, code_exec, api_call, calculator, grep_repo, list_dir, image_read, vector_search, shell]
WORDS = "agent tool result search query data model token stream graph node edge latency packet record frame size cache".split()

MOTIF_PROMPTS = {  # task templates that reliably induce each motif with a 7B tool-calling model
    "sequential": "Use {t1} on '{q}', then use {t2} on the result's first word. Then answer in one sentence.",
    "parallel": "In ONE step call {t1}, {t2} and {t3} in parallel on '{q}', then summarise.",
    "semantic_retry": "Call {t1} with argument 'FAIL' (it will raise), read the error, then call it again with '{q}'.",
    "sub_pooled": "Delegate: ask the research sub-agent to investigate '{q}' and report back.",
    "sub_new_conn": "Delegate to the isolated analyst sub-agent (separate session) to study '{q}'.",
    "transport_retry": "Use {t1} on '{q}' and answer.",   # 529 injected by the local fault proxy below
}

def run_agents(framework, episodes, tag, seed=0):
    """GT logging: an httpx event hook records per-request (t_send, req_bytes, resp_bytes, t_end, conv, parent, edge).
    Connection identity comes from the client socket (local port), matching the pcap 4-tuple."""
    import httpx
    url = open(URLF).read().strip() + "/v1"; os.environ["SSLKEYLOGFILE"] = KEYLOG
    cap = subprocess.Popen(["tcpdump", "-i", "any", "-w", f"{RAW}/{tag}.pcap", "-s", "200", "tcp port 443"]); time.sleep(2)
    GT = []; r = random.Random(seed)
    # ... framework-specific agent construction (LangGraph create_react_agent / AutoGen AssistantAgent + tools,
    #     sub-agents as tools; pooled = shared httpx.AsyncClient(http2=True), new_conn = fresh client).
    # The GT writer must emit: turn, ep, arch, depth, conv, edge, parent, status, req_bytes, resp_bytes,
    #     tool_bytes_added, n_tools_added, n_err_added, tools_added, t_send, t_sent, t_first, t_end, stream, port
    raise NotImplementedError("Agent loop is framework-version specific; complete on Kaggle after `serve` + `tunnel` succeed. "
                              "Keep the GT schema above so 02-04 run unchanged.")

def convert(tag):
    """pcap -> obs_<tag>_c0000.csv (port, dir, t, ctype, len) ; observer fields only (TLS record headers)."""
    out = f"{RAW}/obs_{tag}_c0000.csv"
    sh(f"tshark -r {RAW}/{tag}.pcap -Y tls.record -T fields -E separator=, -e tcp.srcport -e tcp.dstport -e frame.time_epoch "
       f"-e tls.record.content_type -e tls.record.length > {RAW}/{tag}_rec.csv")
    rows = []
    for line in open(f"{RAW}/{tag}_rec.csv"):
        sp_, dp, t, ct, ln = line.strip().split(",")[:5]
        for c, l in zip(ct.split(";"), ln.split(";")):       # several records per TCP segment
            up = dp == "443"; rows.append(((sp_ if up else dp), "c2s" if up else "s2c", t, c, l))
    with open(out, "w") as f:
        f.write("port,dir,t,ctype,len\n"); [f.write(",".join(x) + "\n") for x in rows]
    print("wrote", out, len(rows))

def ant_adapter(path):
    """ANT dataset adapter.  The schema of the public release must be checked by hand; map its per-request
    records to the gt/obs schema above.  We deliberately do not hard-code a download URL."""
    raise NotImplementedError(f"Inspect {path} and implement the field mapping (see docs/ROADMAP.md, item R2).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--framework", default="langgraph")
    ap.add_argument("--episodes", type=int, default=50); ap.add_argument("--tag", default="real_lg"); ap.add_argument("--path", default="")
    a = ap.parse_args()
    {"serve": serve, "tunnel": tunnel, "run": lambda: run_agents(a.framework, a.episodes, a.tag), "convert": lambda: convert(a.tag),
     "ant": lambda: ant_adapter(a.path)}[a.stage]()
