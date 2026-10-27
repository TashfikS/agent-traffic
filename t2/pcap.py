"""pcap -> TLS record table (observer view) with per-flow SNI. Pure Python (dpkt)."""
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
