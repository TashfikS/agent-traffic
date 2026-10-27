"""06 - Live free-tier provider validation set ($0).  Providers: Gemini (OpenAI-compatible endpoint), Groq, OpenRouter
(:free models ONLY - the script refuses any model id without the ':free' suffix).
Observer = an on-path TCP relay (own thread + event loop, i.e. independent of the agent's loop) between the client and the
outbound HTTP proxy; it records the raw bytes of each tunnelled connection and parses TLS 1.3 record headers.  The relay
never sees plaintext (end-to-end TLS through CONNECT).  Ground truth = client-side request log (same schema as testbed2).
Natural 429/503 responses are kept as transport retries.
usage: python scripts/06_freetier_capture.py --provider gemini --episodes 40 --conc 2 --tag ft_gemini"""
import os, sys, json, time, asyncio, threading, argparse, random, csv, importlib.util, socket, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); OUT = f"{ROOT}/data/raw_freetier"; os.makedirs(OUT, exist_ok=True)
PROV = {"gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI", "gemini-3.5-flash-lite"),
        "groq": ("https://api.groq.com/openai/v1", "GROQ", "openai/gpt-oss-20b"),
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER", "google/gemma-4-31b-it:free"),
        "cloudflare": ("https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1", "CLOUDFLARE_API_TOKEN", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")}
ap = argparse.ArgumentParser(); ap.add_argument("--provider", required=True); ap.add_argument("--model", default=None)
ap.add_argument("--episodes", type=int, default=30); ap.add_argument("--conc", type=int, default=2); ap.add_argument("--tag", default=None)
ap.add_argument("--rpm", type=float, default=12); ap.add_argument("--max_requests", type=int, default=600); ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args(); BASE, KEYVAR, DEFM = PROV[A.provider]; MODEL = A.model or DEFM; TAG = A.tag or f"ft_{A.provider}"
if A.provider == "openrouter" and not MODEL.endswith(":free"): sys.exit("refusing non-free OpenRouter model")
KEY = os.environ[KEYVAR]
if "{acct}" in BASE: BASE = BASE.format(acct=os.environ["CLOUDFLARE_ACCCOUNT_ID"])   # Workers AI free tier (10k neurons/day)

# ---------------------------------------------------------------- on-path relay observer
UP = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"); m = re.match(r"https?://(?:[^@]*@)?([^:/]+):(\d+)", UP or "")
UPH, UPP = (m.group(1), int(m.group(2))) if m else (None, None)
REC = []; RLOCK = threading.Lock(); CID = [0]
class Parser:
    def __init__(self, cid, up): self.cid, self.up, self.buf, self.plain = cid, up, b"", True
    def feed(self, data, t):
        self.buf += data
        if self.plain:                                   # CONNECT request / 200 response (plaintext, proxy only)
            i = self.buf.find(b"\r\n\r\n")
            if i < 0: return
            self.buf = self.buf[i + 4:]; self.plain = False
        while len(self.buf) >= 5:
            ln = int.from_bytes(self.buf[3:5], "big")
            if len(self.buf) < 5 + ln: break
            with RLOCK: REC.append((self.cid, "c2s" if self.up else "s2c", t, self.buf[0], ln + 5))
            self.buf = self.buf[5 + ln:]
async def relay(cr, cw):
    CID[0] += 1; cid = CID[0]; sp = cw.get_extra_info("peername")[1]
    if UPH: ur, uw = await asyncio.open_connection(UPH, UPP)
    else:
        line = await cr.readuntil(b"\r\n\r\n"); host, port = line.split()[1].decode().split(":")
        ur, uw = await asyncio.open_connection(host, int(port)); cw.write(b"HTTP/1.1 200 OK\r\n\r\n"); await cw.drain()
    pu, pd_ = Parser(sp, True), Parser(sp, False)
    if not UPH: pu.plain = pd_.plain = False
    async def pipe(r, w, p):
        try:
            while True:
                d = await r.read(65536)
                if not d: break
                p.feed(d, time.time()); w.write(d); await w.drain()
        except Exception: pass
        finally:
            try: w.close()
            except Exception: pass
    await asyncio.gather(pipe(cr, uw, pu), pipe(ur, cw, pd_))
def start_relay():
    loop = asyncio.new_event_loop(); ready = threading.Event(); port = [0]
    def run():
        asyncio.set_event_loop(loop); srv = loop.run_until_complete(asyncio.start_server(relay, "127.0.0.1", 0))
        port[0] = srv.sockets[0].getsockname()[1]; ready.set(); loop.run_forever()
    threading.Thread(target=run, daemon=True).start(); ready.wait(); return port[0]

# ---------------------------------------------------------------- agent (reuse Kaggle real-capture agent)
spec = importlib.util.spec_from_file_location("rc", f"{ROOT}/kaggle/t2-real-capture/t2_real_capture.py")
os.environ.setdefault("T2_TAG", TAG); rc = importlib.util.module_from_spec(spec)
_W = rc.__dict__; _W["__name__"] = "rc"
src = open(spec.origin).read().replace('W = "/kaggle/working"', f'W = "{OUT}"')
exec(compile(src, spec.origin, "exec"), _W)
rc.TAG = TAG; rc.CAP = 2500; rc.BUDGET_S = 6 * 3600; rc.SEED = A.seed
_orig_words = rc.words
SYS_N = (150, 500)
RPM_GAP = 60.0 / A.rpm; LAST = [0.0]; NREQ = [0]; LOCK = None
async def post(client, body, fail):
    global LOCK
    LOCK = LOCK or asyncio.Lock()
    async with LOCK:                                   # client-side rate limiter (free tier RPM)
        w = LAST[0] + RPM_GAP - time.time()
        if w > 0: await asyncio.sleep(w)
        LAST[0] = time.time(); NREQ[0] += 1
    if NREQ[0] > A.max_requests: raise RuntimeError("request cap")
    body = dict(body, model=MODEL)
    tc = body.get("tool_choice")
    if isinstance(tc, dict) and A.provider != "groq": body["tool_choice"] = "required"
    if A.provider == "gemini": body.pop("temperature", None)
    if A.provider == "cloudflare":
        if body.get("tool_choice") == "required": body["tool_choice"] = "auto"
        body["messages"] = [dict(m, content=m.get("content") or "") for m in body["messages"]]
    hdr = {"content-type": "application/json", "authorization": "Bearer " + KEY}
    raw = json.dumps(body).encode(); t0 = time.time(); text, calls, status, nbytes = "", {}, 0, 0
    async with client.stream("POST", BASE + "/chat/completions", content=raw, headers=hdr, timeout=300) as resp:
        status = resp.status_code; t_first = None; buf = b""
        try: client._t2_port = resp.extensions["network_stream"].get_extra_info("client_addr")[1]
        except Exception: pass
        async for ch in resp.aiter_bytes():
            if t_first is None: t_first = time.time()
            nbytes += len(ch); buf += ch
        t_end = time.time()
    port = getattr(client, "_t2_port", -1)
    if status == 200:
        for line in buf.decode(errors="ignore").split("\n"):
            if not line.startswith("data: ") or line[6:].strip() == "[DONE]": continue
            try: d = json.loads(line[6:])["choices"][0].get("delta", {})
            except Exception: continue
            text += d.get("content") or ""
            for tcl in d.get("tool_calls") or []:
                c = calls.setdefault(tcl.get("index", len(calls)), {"id": "", "name": "", "args": ""})
                c["id"] = tcl.get("id") or c["id"]; f = tcl.get("function") or {}
                if tcl.get("extra_content"): c["extra"] = tcl["extra_content"]
                c["name"] = f.get("name") or c["name"]; c["args"] += f.get("arguments") or ""
    else: rc.log("status", status, buf[:200])
    if status in (429, 503, 500, 502): status = 529               # normalise to the testbed's transport-retry code
    return status, text, [calls[i] for i in sorted(calls)], dict(t_send=t0, t_first=t_first or t_end, t_end=t_end, port=port, req_bytes=len(raw), resp_bytes=nbytes)
rc.post = post
RELAY = start_relay()
def new_client():
    import httpx
    c = httpx.AsyncClient(http2=True, timeout=300, proxy=f"http://127.0.0.1:{RELAY}", trust_env=False,
                          limits=httpx.Limits(max_connections=1, max_keepalive_connections=1))
    c._t2_port = -1; return c
rc.new_client = new_client
# GT port <-> relay connection: the relay keys records by the client's ephemeral port; httpx exposes it per response
_orig_agent = rc.agent
async def _post_with_port(client, body, fail):
    r = await post(client, body, fail); return r
# patch post to read the local port from the live connection pool
async def post_p(client, body, fail):
    st, tx, ca, tm = await post(client, body, fail)
    try:
        for c in client._transport._pool.connections:
            s = c._connection._network_stream if hasattr(c, "_connection") and c._connection else None
            if s is not None: tm["port"] = s.get_extra_info("socket").getsockname()[1]; break
    except Exception: pass
    return st, tx, ca, tm
rc.post = post_p
# make the planned transport-retry flag inert (no fault proxy for live providers; real 429s are labelled instead)
_mp = rc.make_plan
def make_plan(r, depth):
    P = _mp(r, depth)
    for s in P: s["transport_retry"] = False
    return P
rc.make_plan = make_plan
_orig_SYS_words = rc.words
rc.EPISODES = A.episodes; rc.CONC = A.conc
if __name__ == "__main__":
    rc.URL = BASE; t0 = time.time()
    try: asyncio.run(rc.main_agents())
    except Exception as e: rc.log("main error", repr(e)[:300])
    time.sleep(1)
    G = rc.GT
    for g in G: g["arch"] = f"chat:{A.provider}"
    if G:
        with open(f"{OUT}/gt_{TAG}_c0000.csv", "w", newline="") as f: w = csv.DictWriter(f, list(G[0].keys())); w.writeheader(); w.writerows(G)
    with RLOCK, open(f"{OUT}/obs_{TAG}_c0000.csv", "w") as f:
        f.write("port,dir,t,ctype,len\n"); [f.write(f"{c},{d},{t:.6f},{ct},{l}\n") for c, d, t, ct, l in REC]
    json.dump(dict(provider=A.provider, model=MODEL, episodes=A.episodes, requests=NREQ[0], ts=1.0, observer="on-path relay"), open(f"{OUT}/meta_{TAG}_c0000.json", "w"))
    print("DONE turns", len(G), "records", len(REC), "s", round(time.time() - t0))
