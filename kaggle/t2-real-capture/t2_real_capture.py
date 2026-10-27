"""Topic 2 real-stack capture (Kaggle T4, $0).
vLLM (Qwen2.5-3B-Instruct, real BPE + streaming + prefix cache) -> local fault proxy (529 injection) -> cloudflared quick
tunnel (Cloudflare edge: TLS 1.3 + HTTP/2) <- httpx HTTP/2 agents over the public internet.  tcpdump on client side.
Outputs /kaggle/working/raw_real/{gt,obs,meta}_<tag>_c0000.csv|json in testbed2.py schema (t in real seconds; TS=1)."""
import os, sys, subprocess, time, json, re, random, string, base64, asyncio, signal, traceback
W = "/kaggle/working"; RAW = f"{W}/raw_real"; os.makedirs(RAW, exist_ok=True)
LOG = open(f"{W}/run_log.txt", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def sh(c, check=True):
    log("+", c); r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if r.returncode and check: log(r.stdout[-2000:], r.stderr[-3000:]); raise RuntimeError(c)
    return r
MODEL = os.environ.get("T2_MODEL", "Qwen/Qwen2.5-3B-Instruct")
EPISODES = int(os.environ.get("T2_EPISODES", "150")); CONC = int(os.environ.get("T2_CONC", "4")); TAG = os.environ.get("T2_TAG", "real_cf")
BUDGET_S = float(os.environ.get("T2_BUDGET_S", str(7.5 * 3600)))
T_START = time.time()

# ------------------------------------------------------------ setup
def setup():
    sh("pip -q install 'httpx[http2]' aiohttp dpkt", check=True)
    if os.environ.get("T2_BACKEND", "hf") != "hf": sh("pip -q install vllm", check=False)
    sh("DEBIAN_FRONTEND=noninteractive apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y tcpdump tshark >/dev/null", check=False)
    if not os.path.exists("/usr/local/bin/cloudflared"):
        sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared")
    log(sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False).stdout, sh("which tcpdump tshark", check=False).stdout)

HF_SERVER = r'''
import sys, json, time, asyncio, threading, uuid, re, torch
from aiohttp import web
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
MODEL, PORT, DEV = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
LOCK = asyncio.Lock()
def chunk(rid, delta, fin=None):
    return b"data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": "agent-model",
                                   "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}).encode() + b"\n\n"
async def h(req):
    b = await req.json(); msgs = b["messages"]; tools = b.get("tools"); tc = b.get("tool_choice", "auto"); mx = int(b.get("max_tokens", 256))
    for m in msgs:
        for c in m.get("tool_calls") or []:
            if isinstance(c["function"].get("arguments"), str):
                try: c["function"]["arguments"] = json.loads(c["function"]["arguments"] or "{}")
                except Exception: c["function"]["arguments"] = {"q": c["function"]["arguments"]}
    prompt = tok.apply_chat_template(msgs, tools=tools if tc != "none" else None, tokenize=False, add_generation_prompt=True)
    forced = None
    if isinstance(tc, dict): forced = tc["function"]["name"]; prompt += '<tool_call>\n{"name": "%s", "arguments": {"q": "' % forced
    elif tc == "required": prompt += "<tool_call>\n"
    ids = tok(prompt, return_tensors="pt").input_ids[:, -7000:].to(DEV)
    rid = "chatcmpl-" + uuid.uuid4().hex[:24]
    resp = web.StreamResponse(headers={"content-type": "text/event-stream"}); await resp.prepare(req)
    async with LOCK:
        st = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
        th = threading.Thread(target=model.generate, kwargs=dict(input_ids=ids, max_new_tokens=mx, do_sample=True, temperature=0.7, top_p=0.9, streamer=st,
                                                                 pad_token_id=tok.eos_token_id)); th.start()
        await resp.write(chunk(rid, {"role": "assistant", "content": ""}))
        full = ""; ncall = 0; it = iter(st); mode_tool = forced is not None or tc == "required"; args_buf = ""; cur_name = forced
        if forced: await resp.write(chunk(rid, {"tool_calls": [{"index": 0, "id": "call_" + uuid.uuid4().hex[:8], "type": "function", "function": {"name": forced, "arguments": ""}}]})); args_buf = '{"q": "'
        while True:
            t = await asyncio.to_thread(next, it, None)
            if t is None: break
            full += t
            if mode_tool:
                if forced:     # stream argument tokens as tool_call deltas until the closing quote
                    seg = t.split('"')[0]
                    if seg: await resp.write(chunk(rid, {"tool_calls": [{"index": 0, "function": {"arguments": seg.replace("\\", "")}}]})); args_buf += seg
                    if '"' in t: break
                else:
                    pass       # required: buffer, emit parsed calls at the end
            else:
                if "<tool_call>" in full: mode_tool = True; continue
                t2 = t.replace("<|im_end|>", "").replace("<|endoftext|>", "")
                if t2: await resp.write(chunk(rid, {"content": t2}))
        th.join()
    if forced:
        await resp.write(chunk(rid, {"tool_calls": [{"index": 0, "function": {"arguments": '"}'}}]}))
    elif mode_tool:
        body = ("<tool_call>\n" if tc == "required" else "") + full
        for i, m in enumerate(re.findall(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", body, re.S)):
            try: d = json.loads(m)
            except Exception: continue
            a = json.dumps(d.get("arguments", {}))
            await resp.write(chunk(rid, {"tool_calls": [{"index": i, "id": "call_" + uuid.uuid4().hex[:8], "type": "function", "function": {"name": d.get("name", ""), "arguments": ""}}]}))
            for k in range(0, len(a), 6): await resp.write(chunk(rid, {"tool_calls": [{"index": i, "function": {"arguments": a[k:k+6]}}]}))
    await resp.write(chunk(rid, {}, "tool_calls" if mode_tool else "stop")); await resp.write(b"data: [DONE]\n\n"); await resp.write_eof(); return resp
async def models(req): return web.json_response({"data": [{"id": "agent-model"}]})
app = web.Application(client_max_size=64 * 2**20); app.router.add_post("/v1/chat/completions", h); app.router.add_get("/v1/models", models)
web.run_app(app, host="127.0.0.1", port=PORT, print=None)
'''
BACKENDS = ["http://127.0.0.1:8000"]
def start_hf():
    global BACKENDS
    import torch; n = max(1, torch.cuda.device_count()); open(f"{W}/hf_server.py", "w").write(HF_SERVER); BACKENDS = []
    m = os.environ.get("T2_HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    for g in range(n):
        subprocess.Popen([sys.executable, f"{W}/hf_server.py", m, str(8100 + g), f"cuda:{g}"], stdout=open(f"{W}/hf_{g}.log", "w"), stderr=subprocess.STDOUT)
        BACKENDS.append(f"http://127.0.0.1:{8100 + g}")
    import urllib.request
    for i in range(400):
        try:
            for b in BACKENDS: urllib.request.urlopen(b + "/v1/models", timeout=2)
            log("hf backends up", BACKENDS, "model", m); return "hf:" + m
        except Exception: time.sleep(3)
    log(open(f"{W}/hf_0.log").read()[-3000:]); raise RuntimeError("hf server failed")

def start_llm():
    if os.environ.get("T2_BACKEND", "hf") != "hf":
        try: start_vllm(); return "vllm:" + MODEL
        except Exception as e: log("vllm unavailable -> HF fallback", repr(e)[:300]); sh("pkill -f vllm.entrypoints", check=False)
    return start_hf()

def start_vllm():
    env = dict(os.environ)
    p = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--host", "127.0.0.1", "--port", "8000",
                          "--dtype", "half", "--max-model-len", "16384", "--gpu-memory-utilization", "0.88", "--enable-prefix-caching",
                          "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--served-model-name", "agent-model"],
                         stdout=open(f"{W}/vllm.log", "w"), stderr=subprocess.STDOUT, env=env)
    import urllib.request
    for i in range(450):
        try: urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2); log("vllm up after", i * 2, "s"); return p
        except Exception: time.sleep(2)
        if p.poll() is not None: break
    log(open(f"{W}/vllm.log").read()[-4000:]); raise RuntimeError("vllm failed")

PROXY = r'''
import asyncio, aiohttp
from aiohttp import web
import sys, json
BK = json.loads(sys.argv[1]); RR = 0
async def h(req):
    body = await req.read()
    if req.headers.get("x-t2-fail") == "1":
        await asyncio.sleep(0.05); return web.json_response({"error": {"type": "overloaded_error", "message": "Overloaded"}}, status=529)
    global RR
    RR = (RR + 1) % len(BK); be = BK[RR]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as s:
        async with s.post(be + req.path, data=body, headers={"content-type": "application/json"}) as r:
            resp = web.StreamResponse(status=r.status, headers={"content-type": r.headers.get("content-type", "application/json")})
            await resp.prepare(req)
            async for chunk in r.content.iter_any(): await resp.write(chunk)
            await resp.write_eof(); return resp
app = web.Application(client_max_size=64 * 2**20); app.router.add_post("/{tail:.*}", h); web.run_app(app, host="127.0.0.1", port=8001, print=None)
'''
def start_proxy_tunnel():
    open(f"{W}/proxy.py", "w").write(PROXY)
    subprocess.Popen([sys.executable, f"{W}/proxy.py", json.dumps(BACKENDS)], stdout=open(f"{W}/proxy.log", "w"), stderr=subprocess.STDOUT); time.sleep(3)
    subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", "http://127.0.0.1:8001"],
                     stdout=open(f"{W}/cloudflared.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(f"{W}/cloudflared.log").read())
        if m: url = m.group(0); break
        time.sleep(1)
    else: raise RuntimeError("no tunnel url")
    import urllib.request
    for i in range(120):    # wait for DNS + edge registration
        try:
            rq = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps({"model": "agent-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).encode(),
                                        headers={"content-type": "application/json"}); urllib.request.urlopen(rq, timeout=30); log("tunnel ok", url, i); return url
        except Exception as e: time.sleep(3)
    raise RuntimeError("tunnel not reachable")

# ------------------------------------------------------------ tools (real execution, testbed size regimes; capped for 16k context)
WORDS = open("/usr/share/dict/words").read().split()[:50000] if os.path.exists("/usr/share/dict/words") else \
    "agent tool result search query data model token stream graph node edge latency packet record frame size cache index table".split()
TOOLS = {"web_search": (3.3, .3, 1.2, "json"), "url_fetch": (3.7, .4, 1.5, "text"), "sql_query": (3.0, .5, .4, "json"), "file_read": (3.4, .5, .05, "file"),
         "code_exec": (2.6, .5, 2.0, "exec"), "api_call": (2.9, .4, .6, "json"), "calculator": (1.6, .2, .02, "calc"), "grep_repo": (3.1, .4, .3, "grep"),
         "list_dir": (2.6, .4, .03, "ls"), "image_read": (3.6, .2, .2, "b64"), "vector_search": (3.4, .3, .5, "json"), "shell": (2.7, .5, .8, "ps")}
TOOL_NAMES = sorted(TOOLS); CAP = 6000
def words(r, n):
    out, t = [], 0
    while t < n: w = r.choice(WORDS); out.append(w); t += len(w) + 1
    return " ".join(out)[:n]
PYFILES = [os.path.join(d, f) for d, _, fs in os.walk("/usr/lib/python3") for f in fs if f.endswith(".py")][:3000] or [__file__]
def run_tool(name, r, fail):
    if fail:
        return "Traceback (most recent call last):\n  File \"tool.py\", line %d, in run\n    %s\n%s: %s" % (
            r.randint(10, 400), words(r, r.randint(40, 200)), r.choice(["KeyError", "ValueError", "TimeoutError", "ToolError"]), words(r, r.randint(30, 120))), 0.0
    mu, sd, lat, kind = TOOLS[name]; n = min(CAP, max(20, int(10 ** r.gauss(mu, sd)))); t0 = time.time()
    if kind == "json":
        items, tot = [], 0
        while tot < n: it = {"id": r.randint(0, 10**6), "title": words(r, 40), "text": words(r, r.randint(80, 400))}; items.append(it); tot += len(json.dumps(it))
        out = json.dumps(items)[:n]
    elif kind == "file": out = open(r.choice(PYFILES), errors="ignore").read()[:n] or words(r, n)
    elif kind == "exec": out = subprocess.run([sys.executable, "-c", f"import random;random.seed({r.randint(0,999)});print([round(random.random(),4) for _ in range({max(1, n // 8)})])"], capture_output=True, text=True).stdout[:n]
    elif kind == "calc": out = repr(r.random() * 10 ** r.randint(1, 9))
    elif kind == "grep": out = subprocess.run(f"grep -rn 'def ' /usr/lib/python3/dist-packages 2>/dev/null | head -c {n}", shell=True, capture_output=True, text=True).stdout or words(r, n)
    elif kind == "ls": out = "\n".join(sorted(os.listdir("/usr/lib")))[:n]
    elif kind == "b64": out = base64.b64encode(os.urandom(int(n * .75))).decode()[:n]
    elif kind == "ps": out = subprocess.run("ps aux", shell=True, capture_output=True, text=True).stdout[:n]
    else: out = words(r, n)
    return out, max(0.0, lat * r.lognormvariate(0, .5) - (time.time() - t0))

def schema(n): return {"type": "function", "function": {"name": n, "description": f"{n.replace('_', ' ')} tool. " + words(random.Random(n), 160),
                                                      "parameters": {"type": "object", "properties": {"q": {"type": "string", "description": "query or argument"}}, "required": ["q"]}}}
SCHEMAS = [schema(n) for n in TOOL_NAMES + ["subagent"]]

SEED = int(os.environ.get("T2_SEED", "0")); FORCE_M = int(os.environ.get("T2_FORCE_M", "0"))
def make_plan(r, depth):
    steps = []
    for i in range(r.randint(3, 8) if depth == 0 else r.randint(1, 3)):
        u = r.random()
        if FORCE_M and depth == 0 and i == 1: motif, k = "sub_pooled", FORCE_M
        elif u < .45: motif, k = "sequential", 1
        elif u < .72: motif, k = "parallel", r.randint(2, 3)
        elif u < .86 and depth == 0: motif, k = "sub_pooled", r.randint(1, 3)
        elif depth == 0: motif, k = "sub_new_conn", 1
        else: motif, k = "sequential", 1
        tools = ["subagent"] * k if motif.startswith("sub") else [r.choice(TOOL_NAMES) for _ in range(k)]
        fail = [False if t == "subagent" else r.random() < .12 for t in tools]
        steps.append(dict(motif=motif, tools=tools, fail=fail, transport_retry=r.random() < .08))
        if any(fail): steps.append(dict(motif="semantic_retry", tools=[t for t, f in zip(tools, fail) if f], fail=[False] * sum(fail), transport_retry=False))
    return steps

# ------------------------------------------------------------ agent
GT = []; TURN = [0]; URL = None
SYS = ("You are a careful autonomous research agent. You solve tasks by calling the provided tools, reading their results and deciding the next action. "
       "Always explain your reasoning briefly before calling a tool. ")
async def post(client, body, fail):
    hdr = {"content-type": "application/json", "authorization": "Bearer sk-local-" + "x" * 40, "user-agent": "agent-sdk/1.0"}
    if fail: hdr["x-t2-fail"] = "1"
    raw = json.dumps(body).encode(); t0 = time.time(); text, calls, port, status, nbytes = "", {}, -1, 0, 0
    async with client.stream("POST", URL + "/v1/chat/completions", content=raw, headers=hdr, timeout=600) as resp:
        status = resp.status_code
        try: port = resp.extensions["network_stream"].get_extra_info("client_addr")[1]
        except Exception: port = -1
        t_first = None; buf = b""
        async for ch in resp.aiter_bytes():
            if t_first is None: t_first = time.time()
            nbytes += len(ch); buf += ch
        t_end = time.time()
    if status == 200:
        for line in buf.decode(errors="ignore").split("\n"):
            if not line.startswith("data: ") or line[6:].strip() == "[DONE]": continue
            try: d = json.loads(line[6:])["choices"][0].get("delta", {})
            except Exception: continue
            text += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                c["id"] = tc.get("id") or c["id"]; f = tc.get("function") or {}; c["name"] = f.get("name") or c["name"]; c["args"] += f.get("arguments") or ""
    return status, text, [calls[i] for i in sorted(calls)], dict(t_send=t0, t_first=t_first or t_end, t_end=t_end, port=port, req_bytes=len(raw), resp_bytes=nbytes)

def new_client():
    import httpx
    return httpx.AsyncClient(http2=True, timeout=600, limits=httpx.Limits(max_connections=1, max_keepalive_connections=1))

async def agent(ep, client, depth, r, parent_turn=None, spawn_edge=None):
    conv = f"{TAG}-{ep}-{depth}-{r.getrandbits(24):06x}"; plan = make_plan(r, depth)
    task = f"Task {ep}.{depth}: investigate " + words(r, r.randint(60, 600))
    hist = [{"role": "system", "content": SYS + words(r, r.randint(600, 2500))}, {"role": "user", "content": task}]
    last = parent_turn; edge = spawn_edge or "start"; added = dict(bytes=0, n=0, err=0, tools=[]); i = 0; retried = set()
    while True:
        if time.time() - T_START > BUDGET_S: return "budget"
        step = plan[i] if i < len(plan) else None
        body = {"model": "agent-model", "messages": hist, "tools": SCHEMAS, "stream": True, "temperature": 0.7}
        if step is None: body.update(tool_choice="none", max_tokens=r.randint(80, 220))
        elif len(step["tools"]) == 1: body.update(tool_choice={"type": "function", "function": {"name": step["tools"][0]}}, max_tokens=120)
        else: body.update(tool_choice="required", max_tokens=200)
        fail = bool(step and step["transport_retry"] and i not in retried)
        for att in range(4):
            try: status, text, calls, tm = await post(client, body, fail); break
            except Exception as e:
                log("post error", ep, att, repr(e)[:160]); await asyncio.sleep(2 + 4 * att)
        else: return "error"
        TURN[0] += 1; tid = TURN[0]
        GT.append(dict(turn=tid, ep=ep, arch="chat", depth=depth, conv=conv, edge=edge, parent=last, status=status, tool_bytes_added=added["bytes"],
                       n_tools_added=added["n"], n_err_added=added["err"], tools_added="|".join(added["tools"]), t_sent=tm["t_send"], stream=0, **tm))
        last = tid
        if status in (524, 502, 503) and r.random() < 0.9: await asyncio.sleep(5); edge = "transport_retry"; continue
        if status == 529: retried.add(i); edge = "transport_retry"; await asyncio.sleep(r.uniform(.5, 2.0)); continue
        if status != 200: log("status", status); return "error"
        if step is None or not calls:
            hist.append({"role": "assistant", "content": text or ""})
            if step is None: return text
            i += 1; edge = "sequential"; added = dict(bytes=0, n=0, err=0, tools=[]); continue
        # bind returned calls to planned tools/failures (names as returned by the model)
        calls = calls[:max(1, len(step["tools"]))]
        # orchestrator fan-out: planned parallel steps (k tools / k sub-agents) are issued even if the small model
        # emitted fewer tool calls (frameworks such as LangGraph Send / AutoGen teams spawn workers programmatically)
        while len(calls) < len(step["tools"]):
            calls.append({"id": "call_%08x" % r.getrandbits(32), "name": step["tools"][len(calls)], "args": json.dumps({"q": words(r, r.randint(20, 120))})})
        for j, c in enumerate(calls):
            c["id"] = c["id"] or "call_%08x" % r.getrandbits(32)
            if c["name"] not in TOOLS and c["name"] != "subagent": c["name"] = step["tools"][min(j, len(step["tools"]) - 1)]
        hist.append({"role": "assistant", "content": text or None, "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"] or "{}"}, **({"extra_content": c["extra"]} if c.get("extra") else {})} for c in calls]})
        async def run(j, c):
            f = step["fail"][j] if j < len(step["fail"]) else False
            if c["name"] == "subagent":
                sc = client if step["motif"] != "sub_new_conn" else new_client()
                out = await agent(ep, sc, depth + 1, random.Random(r.random()), parent_turn=tid, spawn_edge="sub_agent")
                if sc is not client: await sc.aclose()
                return c["id"], out or "", False
            out, wait = run_tool(c["name"], r, f); await asyncio.sleep(wait); return c["id"], out, f
        res = await asyncio.gather(*[run(j, c) for j, c in enumerate(calls)])
        hist += [{"role": "tool", "tool_call_id": cid, "content": o} for cid, o, _ in res]
        added = dict(bytes=sum(len(o) for _, o, _ in res), n=len(res), err=sum(f for _, _, f in res), tools=[c["name"] for c in calls])
        edge = "semantic_retry" if any(f for _, _, f in res) else ("sub_agent" if step["motif"].startswith("sub") else ("parallel_fork_join" if len(calls) > 1 else "sequential"))
        i += 1
        if sum(len(json.dumps(m)) for m in hist) > 38000: plan = plan[:i]   # stay inside the 16k-token window: finish next turn

async def main_agents():
    sem = asyncio.Semaphore(CONC)
    async def one(ep):
        async with sem:
            if time.time() - T_START > BUDGET_S: return
            c = new_client()
            try: await agent(ep, c, 0, random.Random(7919 * ep + 1 + 1000003 * SEED))
            except Exception: log("episode fail", ep, traceback.format_exc()[-500:])
            finally: await c.aclose()
            if ep % 10 == 0: log("episode", ep, "turns", len(GT), "elapsed", round(time.time() - T_START))
    await asyncio.gather(*[one(ep) for ep in range(EPISODES)])

def convert(pcap):
    """Observer: per TCP direction, in-order reassembly (retransmissions dropped by seq), TLS record headers only.
    Record timestamp = capture time of the segment that completes the record (wire time)."""
    import dpkt, socket
    flows = {}; rows = []
    with open(pcap, "rb") as f:
        try: rd = dpkt.pcap.Reader(f)
        except ValueError: f.seek(0); rd = dpkt.pcapng.Reader(f)
        lt = rd.datalink()
        for ts, buf in rd:
            try:
                if lt == 113: pkt = dpkt.sll.SLL(buf).data
                elif lt == 276: pkt = dpkt.sll2.SLL2(buf).data
                else: pkt = dpkt.ethernet.Ethernet(buf).data
                if not isinstance(pkt, (dpkt.ip.IP, dpkt.ip6.IP6)) or not isinstance(pkt.data, dpkt.tcp.TCP): continue
                tcp = pkt.data
            except Exception: continue
            up = tcp.dport == 443; key = (tcp.sport, tcp.dport)
            st = flows.setdefault(key, {"next": None, "buf": b"", "ooo": {}})
            if tcp.flags & dpkt.tcp.TH_SYN: st["next"] = (tcp.seq + 1) & 0xffffffff; continue
            data = bytes(tcp.data)
            if not data or st["next"] is None: continue
            seq = tcp.seq; d = (seq - st["next"]) & 0xffffffff
            if d >= 2**31:                       # starts before next: overlap/retransmission
                cut = (st["next"] - seq) & 0xffffffff
                if cut >= len(data): continue
                data = data[cut:]; seq = st["next"]
            elif d > 0: st["ooo"][seq] = data; continue
            st["buf"] += data; st["next"] = (seq + len(data)) & 0xffffffff
            while st["next"] in st["ooo"]:
                x = st["ooo"].pop(st["next"]); st["buf"] += x; st["next"] = (st["next"] + len(x)) & 0xffffffff
            b_ = st["buf"]
            while len(b_) >= 5:
                ln = int.from_bytes(b_[3:5], "big")
                if len(b_) < 5 + ln: break
                rows.append(f"{tcp.sport if up else tcp.dport},{'c2s' if up else 's2c'},{ts:.6f},{b_[0]},{ln}"); b_ = b_[5 + ln:]
            st["buf"] = b_
    open(f"{RAW}/obs_{TAG}_c0000.csv", "w").write("port,dir,t,ctype,len\n" + "\n".join(rows) + "\n"); log("obs records", len(rows), "flows", len(flows))

if __name__ == "__main__":
    setup(); BACKEND = start_llm(); URL = start_proxy_tunnel(); log("backend", BACKEND)
    pcap = f"{W}/{TAG}.pcap"
    cap = subprocess.Popen(["tcpdump", "-i", "any", "-s", "0", "-w", pcap, "tcp port 443"], stdout=subprocess.DEVNULL, stderr=open(f"{W}/tcpdump.log", "w")); time.sleep(3)
    try: asyncio.run(main_agents())
    finally:
        time.sleep(2); cap.send_signal(signal.SIGINT); cap.wait(timeout=60)
        import csv
        if GT:
            keys = list(GT[0].keys())
            with open(f"{RAW}/gt_{TAG}_c0000.csv", "w", newline="") as f: w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(GT)
        json.dump(dict(backend=BACKEND, model=MODEL, episodes=EPISODES, conc=CONC, url_host="trycloudflare", proto="h2", stream=1, ts=1.0), open(f"{RAW}/meta_{TAG}_c0000.json", "w"))
        convert(pcap); sh(f"ls -la {RAW} {W}", check=False); log("DONE turns", len(GT), "elapsed", round(time.time() - T_START))
