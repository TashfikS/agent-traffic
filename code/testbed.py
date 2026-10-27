"""Topic 2 testbed: agent <-> mock LLM API over real TLS 1.3, observed by an on-path TCP relay
that only sees ciphertext (TLS record headers: type, length, timestamp, direction, connection).

usage: python testbed.py <condition> <n_episodes> <port> <outdir>
condition in {full, server_state, gzip, padding}
"""
import sys, os, json, ssl, socket, threading, time, random, gzip, string, base64, http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COND, NEP, PORT, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
CERT = os.path.join(OUT, "cert.pem"); KEY = os.path.join(OUT, "key.pem")
rnd = random.Random(hash(COND) & 0xffff)

def rtext(n):  # text with realistic punctuation / newlines / quotes (JSON escaping matters)
    words = []
    total = 0
    while total < n:
        w = "".join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randint(2, 9)))
        if rnd.random() < 0.05: w += "\n"
        if rnd.random() < 0.02: w = '"' + w + '"'
        words.append(w); total += len(w) + 1
    return " ".join(words)[:n]

# ---------------- mock LLM server ----------------
STATE = {}; LOCK = threading.Lock()
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.headers.get("Content-Encoding") == "gzip": body = gzip.decompress(body)
        req = json.loads(body)
        cid = req["conv_id"]
        with LOCK:
            st = STATE.setdefault(cid, {"msgs": [], "fails": set()})
            if COND == "server_state": st["msgs"] += req["messages"]; msgs = st["msgs"]
            else: msgs = req["messages"]
            plan = req["plan"]
            step = sum(1 for m in msgs if m["role"] == "assistant")
            fail = step < len(plan) and plan[step].get("retry") and step not in st["fails"]
            if fail: st["fails"].add(step)
        if fail:
            time.sleep(rnd.uniform(0.02, 0.08))
            out = json.dumps({"error": {"type": "overloaded", "message": "try again"}}).encode()
            code = 529
        else:
            if step < len(plan):
                p = plan[step]
                calls = [{"id": "call_%06d" % rnd.randint(0, 999999), "type": "function",
                          "function": {"name": p["tool"], "arguments": json.dumps({"q": rtext(rnd.randint(20, 400))})}}
                         for _ in range(p["k"])]
                msg = {"role": "assistant", "content": None, "tool_calls": calls}
            else:
                msg = {"role": "assistant", "content": rtext(rnd.randint(200, 3000))}
            time.sleep(0.02 + 0.00005*len(json.dumps(msg)))   # generation latency ~ output length
            out = json.dumps({"id": "resp_%d" % rnd.randint(0, 1 << 30), "choices": [{"message": msg}]}).encode()
            code = 200
        if COND == "stream" and code == 200:
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            for i in range(0, len(out), 24):   # ~ a few tokens per SSE event
                ev = b"data: " + json.dumps({"delta": out[i:i+24].decode()}).encode() + b"\n\n"
                self.wfile.write(b"%x\r\n" % len(ev) + ev + b"\r\n"); self.wfile.flush(); time.sleep(0.0015)
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush(); return
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

def serve():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(CERT, KEY)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    srv.serve_forever()

# ---------------- on-path observer (TCP relay parsing TLS record headers only) ----------------
OBS = []; OLOCK = threading.Lock(); CONN = [0]
def pipe(src, dst, cid, direction):
    buf = b""
    try:
        while True:
            d = src.recv(65536)
            if not d: break
            t = time.time(); dst.sendall(d); buf += d
            while len(buf) >= 5:
                ln = int.from_bytes(buf[3:5], "big")
                if len(buf) < 5 + ln: break
                with OLOCK: OBS.append((cid, direction, t, buf[0], ln))
                buf = buf[5 + ln:]
    except OSError: pass
    finally:
        for s in (src, dst):
            try: s.shutdown(socket.SHUT_RDWR)
            except OSError: pass

def relay():
    ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", PORT + 1)); ls.listen(64)
    while True:
        c, _ = ls.accept(); s = socket.create_connection(("127.0.0.1", PORT))
        with OLOCK: CONN[0] += 1; cid = CONN[0]
        threading.Thread(target=pipe, args=(c, s, cid, "c2s"), daemon=True).start()
        threading.Thread(target=pipe, args=(s, c, cid, "s2c"), daemon=True).start()

# ---------------- agent ----------------
GT = []; GLOCK = threading.Lock()
def tool_run(name):
    time.sleep(rnd.uniform(0.02, 0.4))
    if name == "read_file": return rtext(int(10 ** rnd.uniform(2, 4.7)))
    if name == "search": return json.dumps([{"title": rtext(60), "snippet": rtext(rnd.randint(100, 600))}
                                             for _ in range(rnd.randint(1, 10))])
    if name == "screenshot": return base64.b64encode(os.urandom(int(10 ** rnd.uniform(3, 4.8)))).decode()
    return rtext(rnd.randint(50, 2000))

def make_plan(depth=0):
    plan = []
    for _ in range(rnd.randint(2, 7) if depth == 0 else rnd.randint(1, 3)):
        r = rnd.random()
        if r < 0.55: typ, k = "seq", 1
        elif r < 0.85: typ, k = "par", rnd.randint(2, 4)
        else: typ, k = ("sub", 1) if depth == 0 else ("seq", 1)
        tool = "subagent" if typ == "sub" else rnd.choice(["read_file", "search", "screenshot", "shell"])
        plan.append({"type": typ, "k": k, "tool": tool, "retry": rnd.random() < 0.15})
    return plan

def episode(ep, depth=0):
    conn = http.client.HTTPSConnection("127.0.0.1", PORT + 1, context=CCTX)
    plan = make_plan(depth); cid = "%s-%d-%d" % (COND, ep, depth)
    msgs = [{"role": "system", "content": rtext(1500)}, {"role": "user", "content": rtext(rnd.randint(50, 800))}]
    sent = 0; step = 0; new_tool_bytes = 0; ncalls = 0; retry_flag = False
    while True:
        payload_msgs = msgs[sent:] if COND == "server_state" else msgs
        body = {"conv_id": cid, "plan": plan, "messages": payload_msgs}
        raw = json.dumps(body)
        hdr = {"Content-Type": "application/json"}
        if COND == "padding":
            b = 16384; raw = json.dumps({**body, "pad": ""}); n = (len(raw)//b + 1)*b
            raw = json.dumps({**body, "pad": "x"*(n - len(raw))})
        data = raw.encode()
        if COND == "gzip": data = gzip.compress(data); hdr["Content-Encoding"] = "gzip"
        t0 = time.time(); conn.request("POST", "/v1/chat/completions", body=data, headers=hdr)
        resp = conn.getresponse(); rb = resp.read(); t1 = time.time()
        stype = plan[step]["type"] if step < len(plan) else "final"
        with GLOCK:
            GT.append(dict(ep=ep, depth=depth, conv=cid, t_send=t0, t_recv=t1, step=step, step_type=stype,
                           retry=retry_flag, status=resp.status, req_plain=len(raw), body_wire=len(data),
                           resp_plain=len(rb), tool_bytes_added=new_tool_bytes, n_calls_added=ncalls))
        if resp.status != 200:
            retry_flag = True; time.sleep(rnd.uniform(0.2, 0.6)); continue
        retry_flag = False
        if COND == "stream":
            rb2 = "".join(json.loads(l[6:])["delta"] for l in rb.decode().split("\n") if l.startswith("data: "))
            msg = json.loads(rb2)["choices"][0]["message"]
        else: msg = json.loads(rb)["choices"][0]["message"]
        sent = len(msgs); msgs.append(msg)
        if not msg.get("tool_calls"): break
        new_tool_bytes = 0; ncalls = len(msg["tool_calls"])
        res = [None]*ncalls
        def run(i, tc):
            if tc["function"]["name"] == "subagent": res[i] = episode(ep, depth + 1)
            else: res[i] = tool_run(tc["function"]["name"])
        th = [threading.Thread(target=run, args=(i, tc)) for i, tc in enumerate(msg["tool_calls"])]
        [t.start() for t in th]; [t.join() for t in th]
        for tc, r in zip(msg["tool_calls"], res):
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": r}); new_tool_bytes += len(r)
        step += 1
    conn.close()
    return msg["content"]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(CERT):
        os.system(f'openssl req -x509 -newkey rsa:2048 -nodes -keyout "{KEY}" -out "{CERT}" -days 2 -subj /CN=127.0.0.1 2>/dev/null')
    CCTX = ssl.create_default_context(); CCTX.check_hostname = False; CCTX.verify_mode = ssl.CERT_NONE
    threading.Thread(target=serve, daemon=True).start(); threading.Thread(target=relay, daemon=True).start()
    time.sleep(0.5)
    for ep in range(NEP):
        episode(ep)
    time.sleep(0.5)
    import pandas as pd
    pd.DataFrame(GT).to_csv(os.path.join(OUT, f"gt_{COND}.csv"), index=False)
    pd.DataFrame(OBS, columns=["conn", "dir", "t", "ctype", "len"]).to_csv(os.path.join(OUT, f"obs_{COND}.csv"), index=False)
    print(COND, len(GT), "requests", len(OBS), "records")
