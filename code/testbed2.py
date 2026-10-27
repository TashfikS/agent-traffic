"""Track 1 testbed v2: dual-stack (HTTP/2 via hyper-h2, and HTTP/1.1) over TLS 1.3 on loopback,
with an on-path observer that records only TLS record headers (direction, time, type, length).

Agent styles : 'chat'      (OpenAI chat-completions schema, LangGraph-like router loop)
               'messages'  (Anthropic messages schema, content blocks, AutoGen-like reflect turns)
               'responses' (OpenAI Responses/Agents-SDK-like items; supports previous_response_id)
Motifs       : sequential, parallel fork-join (k=2-4), transport retry (529), in-band semantic retry
               (HTTP 200 + tool traceback -> re-call), sub-agent on pooled connection (sub_pooled,
               shares the HTTP/2 connection) or on a new connection (sub_new_conn).
12 tools with log-normal output sizes and latencies.  All delays are time-compressed by TS
(default 0.1); analysis rescales to real-time-equivalent units.

usage: python testbed2.py --out DIR --seed S --episodes N --proto h2|h1 --mode full|server_state|gzip
       [--stream 0|1] [--concurrency C] [--force_sub_m M] [--pad BYTES] [--jitter_ms MS] [--chaff P]
"""
import asyncio, ssl, json, os, sys, time, random, gzip, argparse, string, base64, struct
import numpy as np
import h2.connection, h2.config, h2.events, h2.settings

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--episodes", type=int, default=100); ap.add_argument("--proto", default="h2")
ap.add_argument("--mode", default="full"); ap.add_argument("--stream", type=int, default=1)
ap.add_argument("--concurrency", type=int, default=24); ap.add_argument("--force_sub_m", type=int, default=0)
ap.add_argument("--pad", type=int, default=0); ap.add_argument("--jitter_ms", type=float, default=0)
ap.add_argument("--chaff", type=float, default=0); ap.add_argument("--ts", type=float, default=0.1)
ap.add_argument("--size_mode", default="natural")  # natural | sweep (log-uniform 1e2-2e5) | fixed
ap.add_argument("--fixed_size", type=int, default=5000); ap.add_argument("--motif_only", default="")
ap.add_argument("--ep_start", type=int, default=0)
ap.add_argument("--port", type=int, default=19443); ap.add_argument("--tag", default="")
A = ap.parse_args()
TS = A.ts; RND = random.Random(A.seed); os.makedirs(A.out, exist_ok=True)
TAG = (A.tag or f"{A.proto}_{A.mode}_s{int(A.stream)}_seed{A.seed}") + f"_c{A.ep_start:04d}"

def rtext(n, r=RND):
    out, tot = [], 0
    while tot < n:
        w = "".join(r.choice(string.ascii_lowercase) for _ in range(r.randint(2, 9)))
        if r.random() < .05: w += "\n"
        if r.random() < .02: w = '"' + w + '"'
        out.append(w); tot += len(w) + 1
    return " ".join(out)[:n]

# ------------------------------------------------------------------ tools
TOOLS = {  # name: (log10 median bytes, log10 sd, median latency s, kind)
    "web_search": (3.3, .3, 1.2, "json"), "url_fetch": (4.0, .5, 1.5, "text"), "sql_query": (3.0, .6, .4, "json"),
    "file_read": (3.6, .6, .05, "text"), "code_exec": (2.6, .6, 2.0, "text"), "api_call": (2.9, .4, .6, "json"),
    "calculator": (1.6, .2, .02, "text"), "grep_repo": (3.2, .5, .3, "text"), "list_dir": (2.7, .4, .03, "text"),
    "image_read": (4.5, .3, .2, "b64"), "vector_search": (3.5, .3, .5, "json"), "shell": (2.8, .7, .8, "text")}
TOOL_NAMES = sorted(TOOLS)
def tool_schema(name): return {"type": "function", "function": {"name": name, "description": rtext(120, random.Random(name)),
                               "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}
SCHEMAS = [tool_schema(n) for n in TOOL_NAMES]

def tool_output(name, r, fail, size=None):
    if fail:
        return "Traceback (most recent call last):\n  File \"tool.py\", line %d, in run\n%s\nToolError: %s" % (
            r.randint(10, 400), rtext(r.randint(60, 380), r), rtext(r.randint(40, 120), r))
    mu, sd, _, kind = TOOLS[name]
    if A.size_mode == "sweep": size = int(10 ** r.uniform(2, np.log10(2e5)))
    elif A.size_mode == "fixed": size = A.fixed_size
    n = size or max(20, int(10 ** r.gauss(mu, sd)))
    n = min(n, 200000)
    if kind == "b64": return base64.b64encode(os.urandom(int(n * .75))).decode()[:n]
    if kind == "json":
        items, tot = [], 0
        while tot < n:
            it = {"id": r.randint(0, 10**6), "title": rtext(40, r), "text": rtext(r.randint(80, 400), r)}
            s = len(json.dumps(it)); items.append(it); tot += s
        return json.dumps(items)[:n]
    return rtext(n, r)

# ------------------------------------------------------------------ plans
def make_plan(r, depth, arch, force_m=0):
    steps = []
    nsteps = r.randint(3, 9) if depth == 0 else r.randint(1, 3)
    for i in range(nsteps):
        u = r.random()
        if force_m and depth == 0 and i == 1: motif, k = "sub_pooled", force_m
        elif A.motif_only: motif, k = A.motif_only, 1
        elif u < .45: motif, k = "sequential", 1
        elif u < .75: motif, k = "parallel", r.randint(2, 4)
        elif u < .87 and depth == 0: motif, k = "sub_pooled", r.randint(1, 2)
        elif depth == 0: motif, k = "sub_new_conn", 1
        else: motif, k = "sequential", 1
        tools = ["subagent"] * k if motif.startswith("sub") else [r.choice(TOOL_NAMES) for _ in range(k)]
        fail = [False if t == "subagent" else r.random() < .12 for t in tools]
        steps.append(dict(motif=motif, tools=tools, fail=fail, transport_retry=r.random() < .08))
        if any(fail):   # in-band semantic retry: next LLM turn re-calls the failed tools
            steps.append(dict(motif="semantic_retry", tools=[t for t, f in zip(tools, fail) if f],
                              fail=[False]*sum(fail), transport_retry=False))
        if arch == "messages" and r.random() < .3:  # AutoGen-like reflection turn (no tool)
            steps.append(dict(motif="reflect", tools=[], fail=[], transport_retry=False))
    return steps

# ------------------------------------------------------------------ mock LLM API
SERVER_STATE = {}; FAILED = set()
def llm_respond(req, path):
    r = random.Random(RND.random())
    conv = req["x_conv"]; plan = req["x_plan"]
    if A.mode == "server_state" and "previous_response_id" in req:
        hist = SERVER_STATE.get(req["previous_response_id"], []) + req["input"]
    else: hist = req.get("messages") or req.get("input") or []
    nturn = sum(1 for m in hist if (m.get("role") == "assistant" or m.get("type") == "assistant_turn"))
    if req.get("x_chaff"): return 200, {"kind": "text", "text": rtext(r.randint(40, 300), r)}, hist
    if nturn < len(plan) and plan[nturn]["transport_retry"] and (conv, nturn) not in FAILED:
        FAILED.add((conv, nturn)); return 529, {"error": {"type": "overloaded_error", "message": "Overloaded"}}, hist
    if nturn < len(plan):
        st = plan[nturn]
        if st["motif"] == "reflect": return 200, {"kind": "text", "text": rtext(r.randint(100, 600), r)}, hist
        calls = [{"id": "call_%08x" % r.getrandbits(32), "name": t, "args": json.dumps({"q": rtext(r.randint(20, 300), r), "x_fail": f})}
                 for t, f in zip(st["tools"], st["fail"])]
        return 200, {"kind": "tools", "pre": rtext(r.randint(0, 300), r), "calls": calls}, hist
    return 200, {"kind": "text", "text": rtext(r.randint(400, 3000), r)}, hist

def render_full(arch, out, rid):
    if arch == "chat":
        msg = {"role": "assistant", "content": out.get("text") or out.get("pre") or None}
        if out["kind"] == "tools":
            msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}} for c in out["calls"]]
        return {"id": rid, "object": "chat.completion", "model": "gpt-x", "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if out["kind"] == "tools" else "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
    if arch == "messages":
        content = []
        if out.get("text") or out.get("pre"): content.append({"type": "text", "text": out.get("text") or out.get("pre")})
        for c in out.get("calls", []): content.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": json.loads(c["args"])})
        return {"id": rid, "type": "message", "role": "assistant", "content": content, "stop_reason": "tool_use" if out["kind"] == "tools" else "end_turn"}
    items = []
    if out.get("text") or out.get("pre"): items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": out.get("text") or out.get("pre")}]})
    for c in out.get("calls", []): items.append({"type": "function_call", "call_id": c["id"], "name": c["name"], "arguments": c["args"]})
    return {"id": rid, "object": "response", "output": items}

def sse_events(arch, out, rid, r):
    """Yield SSE event byte strings token-by-token in the provider's framing style."""
    def toks(s):
        i = 0
        while i < len(s):
            n = r.randint(2, 6); yield s[i:i+n]; i += n
    text = out.get("text") or out.get("pre") or ""
    if arch == "chat":
        yield b"data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}).encode() + b"\n\n"
        for t in toks(text): yield b"data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": t}}]}).encode() + b"\n\n"
        for j, c in enumerate(out.get("calls", [])):
            yield b"data: " + json.dumps({"id": rid, "choices": [{"delta": {"tool_calls": [{"index": j, "id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": ""}}]}}]}).encode() + b"\n\n"
            for t in toks(c["args"]): yield b"data: " + json.dumps({"id": rid, "choices": [{"delta": {"tool_calls": [{"index": j, "function": {"arguments": t}}]}}]}).encode() + b"\n\n"
        yield b"data: " + json.dumps({"id": rid, "choices": [{"delta": {}, "finish_reason": "tool_calls" if out["kind"] == "tools" else "stop"}]}).encode() + b"\n\ndata: [DONE]\n\n"
    elif arch == "messages":
        yield b"event: message_start\ndata: " + json.dumps({"type": "message_start", "message": {"id": rid, "role": "assistant"}}).encode() + b"\n\n"
        if text:
            yield b"event: content_block_start\ndata: " + json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}).encode() + b"\n\n"
            for t in toks(text): yield b"event: content_block_delta\ndata: " + json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}}).encode() + b"\n\n"
        for j, c in enumerate(out.get("calls", [])):
            yield b"event: content_block_start\ndata: " + json.dumps({"type": "content_block_start", "index": j+1, "content_block": {"type": "tool_use", "id": c["id"], "name": c["name"], "input": {}}}).encode() + b"\n\n"
            for t in toks(c["args"]): yield b"event: content_block_delta\ndata: " + json.dumps({"type": "content_block_delta", "index": j+1, "delta": {"type": "input_json_delta", "partial_json": t}}).encode() + b"\n\n"
        yield b"event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"
    else:
        for t in toks(text): yield b"event: response.output_text.delta\ndata: " + json.dumps({"type": "response.output_text.delta", "delta": t}).encode() + b"\n\n"
        for c in out.get("calls", []):
            for t in toks(c["args"]): yield b"event: response.function_call_arguments.delta\ndata: " + json.dumps({"type": "response.function_call_arguments.delta", "item_id": c["id"], "delta": t}).encode() + b"\n\n"
        yield b"event: response.completed\ndata: " + json.dumps({"type": "response.completed", "response": render_full("responses", out, rid)}).encode() + b"\n\n"

async def handle_request(body, headers):
    """Returns (status, content_type, async iterator of body chunks)."""
    if headers.get("content-encoding") == "gzip": body = gzip.decompress(body)
    req = json.loads(body); arch = req["x_arch"]; r = random.Random(RND.random())
    status, out, hist = llm_respond(req, "")
    rid = "resp_%016x" % r.getrandbits(64)
    async def gen():
        await asyncio.sleep(TS * r.uniform(.25, 1.2))           # time-to-first-token
        if status != 200:
            yield json.dumps(out).encode(); return
        if A.mode == "server_state":
            SERVER_STATE[rid] = hist + [{"type": "assistant_turn", "role": "assistant", "out": out}]
        if req.get("stream"):
            for ev in sse_events(arch, out, rid, r):
                await asyncio.sleep(TS * r.uniform(.015, .05)); yield ev
        else:
            ntok = len(json.dumps(out)) / 4
            await asyncio.sleep(TS * .02 * ntok); yield json.dumps(render_full(arch, out, rid)).encode()
    return status, ("text/event-stream" if req.get("stream") and status == 200 else "application/json"), gen()

class H2Server:
    def __init__(self, reader, writer):
        self.r, self.w = reader, writer
        self.c = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False, header_encoding="utf-8"))
        self.bufs, self.hdrs = {}, {}; self.wlock = asyncio.Lock()
    async def flush(self):
        d = self.c.data_to_send()
        if d: self.w.write(d); await self.w.drain()
    async def run(self):
        self.c.initiate_connection(); self.c.update_settings({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 2**30})
        self.c.increment_flow_control_window(2**30); await self.flush()
        while True:
            data = await self.r.read(65536)
            if not data: break
            for ev in self.c.receive_data(data):
                if isinstance(ev, h2.events.RequestReceived):
                    self.hdrs[ev.stream_id] = dict(ev.headers); self.bufs[ev.stream_id] = bytearray()
                elif isinstance(ev, h2.events.DataReceived):
                    self.bufs[ev.stream_id] += ev.data; self.c.acknowledge_received_data(ev.flow_controlled_length, ev.stream_id)
                elif isinstance(ev, h2.events.StreamEnded):
                    asyncio.ensure_future(self.respond(ev.stream_id))
            await self.flush()
    async def respond(self, sid):
        status, ctype, gen = await handle_request(bytes(self.bufs.pop(sid)), self.hdrs.pop(sid))
        async with self.wlock:
            self.c.send_headers(sid, [(":status", str(status)), ("content-type", ctype), ("server", "mock")]); await self.flush()
        async for chunk in gen:
            async with self.wlock:
                for i in range(0, len(chunk), 16384): self.c.send_data(sid, chunk[i:i+16384])
                await self.flush()
        async with self.wlock:
            self.c.end_stream(sid); await self.flush()

async def h1_server(reader, writer):
    while True:
        line = await reader.readline()
        if not line: break
        hdrs = {}
        while True:
            h = await reader.readline()
            if h in (b"\r\n", b""): break
            k, v = h.decode().split(":", 1); hdrs[k.strip().lower()] = v.strip()
        body = await reader.readexactly(int(hdrs.get("content-length", 0)))
        status, ctype, gen = await handle_request(body, hdrs)
        if ctype == "text/event-stream":
            writer.write(f"HTTP/1.1 {status} OK\r\nContent-Type: {ctype}\r\nTransfer-Encoding: chunked\r\n\r\n".encode()); await writer.drain()
            async for ch in gen: writer.write(b"%x\r\n" % len(ch) + ch + b"\r\n"); await writer.drain()
            writer.write(b"0\r\n\r\n"); await writer.drain()
        else:
            chunks = b"".join([c async for c in gen])
            writer.write(f"HTTP/1.1 {status} OK\r\nContent-Type: {ctype}\r\nContent-Length: {len(chunks)}\r\n\r\n".encode() + chunks); await writer.drain()

async def server_conn(reader, writer):
    proto = writer.get_extra_info("ssl_object").selected_alpn_protocol()
    try:
        if proto == "h2": await H2Server(reader, writer).run()
        else: await h1_server(reader, writer)
    except (ConnectionError, asyncio.IncompleteReadError): pass
    finally: writer.close()

# ------------------------------------------------------------------ observer (on-path relay)
OBS = []
async def pipe(r, w, port, d):
    buf = b""
    try:
        while True:
            data = await r.read(65536)
            if not data: break
            t = time.time(); w.write(data); await w.drain(); buf += data
            while len(buf) >= 5:
                ln = int.from_bytes(buf[3:5], "big")
                if len(buf) < 5 + ln: break
                OBS.append((port, d, t, buf[0], ln)); buf = buf[5+ln:]
    except ConnectionError: pass
    finally: w.close()
async def relay_conn(cr, cw):
    port = cw.get_extra_info("peername")[1]
    sr, sw = await asyncio.open_connection("127.0.0.1", A.port)
    await asyncio.gather(pipe(cr, sw, port, "c2s"), pipe(sr, cw, port, "s2c"))

# ------------------------------------------------------------------ clients
CCTX = ssl.create_default_context(); CCTX.check_hostname = False; CCTX.verify_mode = ssl.CERT_NONE
CCTX.minimum_version = ssl.TLSVersion.TLSv1_3
class H2Client:
    async def connect(self, alpn="h2"):
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3; ctx.set_alpn_protocols([alpn])
        self.r, self.w = await asyncio.open_connection("127.0.0.1", A.port + 1, ssl=ctx)
        self.port = self.w.get_extra_info("sockname")[1]
        self.c = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True, header_encoding="utf-8"))
        self.c.initiate_connection(); self.c.update_settings({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 2**30})
        self.c.increment_flow_control_window(2**30); self.w.write(self.c.data_to_send())
        self.st = {}; self.lock = asyncio.Lock(); self.win = asyncio.Event(); self.task = asyncio.ensure_future(self.reader()); return self
    async def reader(self):
        while True:
            data = await self.r.read(65536)
            if not data: break
            for ev in self.c.receive_data(data):
                s = self.st.get(getattr(ev, "stream_id", None))
                if isinstance(ev, h2.events.ResponseReceived): s["status"] = int(dict(ev.headers)[":status"]); s["t_first"] = time.time()
                elif isinstance(ev, h2.events.DataReceived): s["body"] += ev.data; self.c.acknowledge_received_data(ev.flow_controlled_length, ev.stream_id)
                elif isinstance(ev, h2.events.StreamEnded): s["t_end"] = time.time(); s["fut"].set_result(True)
                if isinstance(ev, (h2.events.WindowUpdated, h2.events.RemoteSettingsChanged)): self.win.set()
            d = self.c.data_to_send()
            if d: self.w.write(d)
    async def post(self, path, body, headers):
        async with self.lock:
            sid = self.c.get_next_available_stream_id()
            self.st[sid] = dict(body=bytearray(), fut=asyncio.get_event_loop().create_future())
            t0 = time.time()
            self.c.send_headers(sid, [(":method", "POST"), (":path", path), (":scheme", "https"), (":authority", "api.example.com"),
                                      ("content-type", "application/json"), ("authorization", "Bearer sk-" + "x"*48), ("user-agent", "agent-sdk/1.0")] + list(headers.items()))
            i = 0
            while i < len(body):
                n = min(16384, len(body) - i, self.c.local_flow_control_window(sid))
                if n <= 0:
                    self.w.write(self.c.data_to_send()); self.win.clear(); await self.win.wait(); continue
                self.c.send_data(sid, body[i:i+n]); i += n
            self.c.end_stream(sid); self.w.write(self.c.data_to_send()); await self.w.drain(); t1 = time.time()
        s = self.st[sid]; await s["fut"]; del self.st[sid]
        return s["status"], bytes(s["body"]), dict(t_send=t0, t_sent=t1, t_first=s["t_first"], t_end=s["t_end"], stream=sid, port=self.port)
    async def close(self):
        self.c.close_connection(); self.w.write(self.c.data_to_send()); self.w.close(); self.task.cancel()

class H1Client:
    async def connect(self):
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3; ctx.set_alpn_protocols(["http/1.1"])
        self.r, self.w = await asyncio.open_connection("127.0.0.1", A.port + 1, ssl=ctx)
        self.port = self.w.get_extra_info("sockname")[1]; self.lock = asyncio.Lock(); return self
    async def post(self, path, body, headers):
        async with self.lock:
            t0 = time.time()
            h = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
            self.w.write(f"POST {path} HTTP/1.1\r\nHost: api.example.com\r\nContent-Type: application/json\r\nAuthorization: Bearer sk-{'x'*48}\r\nUser-Agent: agent-sdk/1.0\r\n{h}Content-Length: {len(body)}\r\n\r\n".encode() + body)
            await self.w.drain(); t1 = time.time()
            line = await self.r.readline(); tf = time.time(); status = int(line.split()[1]); hd = {}
            while True:
                x = await self.r.readline()
                if x == b"\r\n": break
                k, v = x.decode().split(":", 1); hd[k.strip().lower()] = v.strip()
            if hd.get("transfer-encoding") == "chunked":
                out = bytearray()
                while True:
                    n = int((await self.r.readline()).strip(), 16)
                    if n == 0: await self.r.readline(); break
                    out += await self.r.readexactly(n); await self.r.readline()
            else: out = await self.r.readexactly(int(hd["content-length"]))
            return status, bytes(out), dict(t_send=t0, t_sent=t1, t_first=tf, t_end=time.time(), stream=0, port=self.port)
    async def close(self): self.w.close()

async def new_client():
    return await (H2Client().connect() if A.proto == "h2" else H1Client().connect())

# ------------------------------------------------------------------ agent
GT = []; DAGS = {}
SYS = {"chat": 1500, "messages": 4000, "responses": 8000}
def parse_response(arch, status, body, stream):
    if status != 200: return None
    if stream:
        evs = [json.loads(l[6:]) for l in body.decode().split("\n") if l.startswith("data: ") and l[6:] != "[DONE]"]
        text, calls = "", {}
        if arch == "chat":
            for e in evs:
                d = e["choices"][0].get("delta", {})
                text += d.get("content") or ""
                for tc in d.get("tool_calls", []):
                    c = calls.setdefault(tc["index"], {"id": "", "name": "", "args": ""})
                    c["id"] = tc.get("id", c["id"]); c["name"] = tc.get("function", {}).get("name", "") or c["name"]; c["args"] += tc.get("function", {}).get("arguments", "")
            return {"text": text, "calls": [calls[i] for i in sorted(calls)], "rid": evs[0].get("id")}
        if arch == "messages":
            for e in evs:
                if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use":
                    calls[e["index"]] = {"id": e["content_block"]["id"], "name": e["content_block"]["name"], "args": ""}
                elif e["type"] == "content_block_delta":
                    if e["delta"]["type"] == "text_delta": text += e["delta"]["text"]
                    else: calls[e["index"]]["args"] += e["delta"]["partial_json"]
            return {"text": text, "calls": [calls[i] for i in sorted(calls)], "rid": evs[0]["message"]["id"]}
        resp = [e for e in evs if e["type"] == "response.completed"][0]["response"]; body = json.dumps(resp).encode()
    j = json.loads(body)
    if arch == "chat":
        m = j["choices"][0]["message"]; return {"text": m.get("content") or "", "calls": [{"id": c["id"], "name": c["function"]["name"], "args": c["function"]["arguments"]} for c in m.get("tool_calls", [])], "rid": j["id"]}
    if arch == "messages":
        return {"text": "".join(b.get("text", "") for b in j["content"] if b["type"] == "text"),
                "calls": [{"id": b["id"], "name": b["name"], "args": json.dumps(b["input"])} for b in j["content"] if b["type"] == "tool_use"], "rid": j["id"]}
    return {"text": "", "calls": [{"id": i["call_id"], "name": i["name"], "args": i["arguments"]} for i in j["output"] if i["type"] == "function_call"], "rid": j["id"]}

def assistant_msg(arch, p):
    if arch == "chat":
        m = {"role": "assistant", "content": p["text"] or None}
        if p["calls"]: m["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}} for c in p["calls"]]
        return [m]
    if arch == "messages":
        content = ([{"type": "text", "text": p["text"]}] if p["text"] else []) + [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": json.loads(c["args"])} for c in p["calls"]]
        return [{"role": "assistant", "content": content}]
    return ([{"type": "assistant_turn", "role": "assistant", "content": p["text"]}] +
            [{"type": "function_call", "call_id": c["id"], "name": c["name"], "arguments": c["args"]} for c in p["calls"]])

def tool_msgs(arch, results):
    if arch == "chat": return [{"role": "tool", "tool_call_id": cid, "content": out} for cid, out in results]
    if arch == "messages": return [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": out} for cid, out in results]}]
    return [{"type": "function_call_output", "call_id": cid, "output": out} for cid, out in results]

TURN_ID = [0]
async def agent(ep, arch, client, depth, r, parent_turn=None, spawn_edge=None, force_m=0):
    conv = f"{TAG}-{ep}-{depth}-{r.getrandbits(24):06x}"
    plan = make_plan(r, depth, arch, force_m)
    hist = [{"role": "system", "content": rtext(SYS[arch], r)} if arch != "messages" else {"role": "user", "content": rtext(200, r)},
            {"role": "user", "content": rtext(r.randint(80, 1200), r)}]
    sent_upto = 0; prev_rid = None; last_turn = parent_turn; edge = spawn_edge or "start"
    added = dict(bytes=0, n=0, err=0, tools=[]); final_text = ""
    while True:
        if A.chaff and r.random() < A.chaff:   # defence: dummy chaff turn on the same connection
            body = json.dumps({"x_arch": arch, "x_conv": conv + "-chaff", "x_plan": [], "x_chaff": True, "stream": bool(A.stream),
                               "messages": [{"role": "user", "content": rtext(r.randint(200, 20000), r)}]}).encode()
            await client.post("/v1/chat", body, {})
        if A.jitter_ms: await asyncio.sleep(TS * r.uniform(0, A.jitter_ms) / 1000)
        if A.mode == "server_state" and arch == "responses" and prev_rid:
            payload = {"previous_response_id": prev_rid, "input": hist[sent_upto:]}
        elif A.mode == "server_state":
            payload = {"x_state_conv": conv, ("messages" if arch != "responses" else "input"): hist[sent_upto:]}
        else: payload = {("messages" if arch != "responses" else "input"): hist}
        req = {"model": "model-x", "x_arch": arch, "x_conv": conv, "x_plan": plan, "stream": bool(A.stream), "tools": SCHEMAS, **payload}
        if A.mode == "server_state" and arch != "responses":
            # emulate provider-side thread state for non-responses schemas
            req["previous_response_id"] = prev_rid or "none"; req["input"] = hist[sent_upto:]
            req.pop("messages", None)
        raw = json.dumps(req).encode()
        if A.pad:
            n = (len(raw) // A.pad + 1) * A.pad; req["x_pad"] = "x" * max(0, n - len(raw) - 12); raw = json.dumps(req).encode()
        hdr = {}
        if A.mode == "gzip": raw = gzip.compress(raw); hdr["content-encoding"] = "gzip"
        status, body, tm = await client.post("/v1/" + arch, raw, hdr)
        TURN_ID[0] += 1; tid = TURN_ID[0]
        GT.append(dict(turn=tid, ep=ep, arch=arch, depth=depth, conv=conv, edge=edge, parent=last_turn, status=status,
                       req_bytes=len(raw), resp_bytes=len(body), tool_bytes_added=added["bytes"], n_tools_added=added["n"],
                       n_err_added=added["err"], tools_added="|".join(added["tools"]), **tm))
        DAGS.setdefault(ep, []).append((last_turn, tid, edge))
        last_turn = tid
        if status != 200:
            edge = "transport_retry"; await asyncio.sleep(TS * r.uniform(.5, 2.0)); continue
        p = parse_response(arch, status, body, bool(A.stream))
        if A.mode == "server_state":
            prev_rid = (p.get("rid") or "x"); sent_upto = len(hist) + len(assistant_msg(arch, p))
        hist += assistant_msg(arch, p)
        if not p["calls"]:
            nturn = sum(1 for m in hist if m.get("role") == "assistant" or m.get("type") == "assistant_turn")
            if nturn > len(plan): final_text = p["text"]; break
            edge = "sequential"; added = dict(bytes=0, n=0, err=0, tools=[]); continue
        step = plan[min(sum(1 for m in hist if m.get("role") == "assistant" or m.get("type") == "assistant_turn") - 1, len(plan) - 1)]
        motif = step["motif"]
        async def run_call(c):
            a = json.loads(c["args"]) if c["args"] else {}
            if c["name"] == "subagent":
                if motif == "sub_pooled": sub_client = client
                else: sub_client = await new_client()
                out = await agent(ep, arch, sub_client, depth + 1, random.Random(r.random()), parent_turn=tid, spawn_edge="sub_agent")
                if sub_client is not client: await sub_client.close()
                return c["id"], out, False
            await asyncio.sleep(TS * TOOLS[c["name"]][2] * r.lognormvariate(0, .5))
            return c["id"], tool_output(c["name"], r, a.get("x_fail", False)), a.get("x_fail", False)
        res = await asyncio.gather(*[run_call(c) for c in p["calls"]])
        added = dict(bytes=sum(len(o) for _, o, _ in res), n=len(res), err=sum(f for _, _, f in res), tools=[c["name"] for c in p["calls"]])
        edge = {"sequential": "sequential", "parallel": "parallel_fork_join", "semantic_retry": "sequential",
                "sub_pooled": "sub_agent", "sub_new_conn": "sub_agent"}.get(motif, "sequential")
        if any(f for _, _, f in res): edge_next_retry = True
        else: edge_next_retry = False
        hist += tool_msgs(arch, [(cid, o) for cid, o, _ in res])
        if edge_next_retry: edge = "semantic_retry"
    return final_text

async def main():
    certd = A.out
    if not os.path.exists(f"{certd}/key.pem"):
        os.system(f'openssl req -x509 -newkey rsa:2048 -nodes -keyout "{certd}/key.pem" -out "{certd}/cert.pem" -days 2 -subj /CN=api.example.com 2>/dev/null')
    sctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); sctx.load_cert_chain(f"{certd}/cert.pem", f"{certd}/key.pem")
    sctx.minimum_version = ssl.TLSVersion.TLSv1_3; sctx.set_alpn_protocols(["h2", "http/1.1"])
    srv = await asyncio.start_server(server_conn, "127.0.0.1", A.port, ssl=sctx)
    rel = await asyncio.start_server(relay_conn, "127.0.0.1", A.port + 1)
    sem = asyncio.Semaphore(A.concurrency)
    async def one(ep):
        async with sem:
            r = random.Random(A.seed * 100003 + ep); arch = ["chat", "messages", "responses"][ep % 3]
            c = await new_client()
            try: await agent(ep, arch, c, 0, r, force_m=A.force_sub_m)
            finally: await c.close()
    t0 = time.time()
    await asyncio.gather(*[one(ep) for ep in range(A.ep_start, A.ep_start + A.episodes)])
    await asyncio.sleep(.3)
    import pandas as pd
    pd.DataFrame(GT).to_csv(f"{A.out}/gt_{TAG}.csv", index=False)
    pd.DataFrame(OBS, columns=["port", "dir", "t", "ctype", "len"]).to_csv(f"{A.out}/obs_{TAG}.csv", index=False)
    json.dump({str(k): v for k, v in DAGS.items()}, open(f"{A.out}/dag_{TAG}.json", "w"))
    json.dump(vars(A), open(f"{A.out}/meta_{TAG}.json", "w"))
    print(TAG, "episodes", A.episodes, "turns", len(GT), "records", len(OBS), "wall", round(time.time() - t0, 1), "s", flush=True)

asyncio.run(main())
