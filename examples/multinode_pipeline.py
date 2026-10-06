"""
Layer-split (pipeline) inference across machines with NetEndpoint (TCP) or NcclEndpoint
(torch.distributed p2p; NCCL over RoCE on RDMA-capable links).

Every rank builds the full module list from the model config but loads, and allocates cache for,
only its own contiguous slice of decoder layers. Rank 0 also holds the embedding and drives
generation; the last rank holds the final norm and head and returns the next token (or the NLL of a
scored chunk) to rank 0. Each rank needs config.json, the tokenizer files and the .safetensors
shards covering its own layers; other shards may be absent.

Split points must start a rank on a layer that does not read per-process state published by an
earlier layer. For DeepSeek-V3.2 / GLM-5 DSA models that means an indexer "full" layer (shared
layers reuse the preceding full layer's top-k selection); the script checks this when the config
has indexer_types.

Example, four hosts (start ranks 3, 2 and 1 before rank 0):

    # Set these from fabric discovery on each host, not from another cluster's names.
    export NCCL_SOCKET_IFNAME="$FABRIC_IFACE" NCCL_IB_HCA="$RDMA_HCA"
    python examples/multinode_pipeline.py -m /models/GLM-5.3-EXL3 --rank 3 \\
        --addrs rank0,rank1,rank2,rank3 \\
        --splits 0:26,26:46,46:62,62:78 --transport nccl
    # Run the same command on ranks 2 and 1, changing --rank.
    # On rank 0, change --rank to 0 and add --prompt "Hello" --max_new 200.

This is sequential layer-split PP, not cross-host tensor parallelism or an overlapped
microbatch scheduler. See doc/multinode.md for setup, precision, memory safety, measured
short-context results and the separate, unshipped TP/CP research paths.
"""

import argparse, os, sys, time, json, math, threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.model.net_transport import NetEndpoint
from exllamav3.util.memory import free_mem
from exllamav3.util.tensor import g_tensor_cache

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True)
ap.add_argument("--rank", type = int, required = True)
ap.add_argument("--addrs", required = True, help = "comma-separated link addresses, indexed by rank")
ap.add_argument("--splits", required = True, help = "comma-separated a:b layer ranges, one per rank")
ap.add_argument("--port", type = int, default = 29650)
ap.add_argument("--transport", choices = ["tcp", "nccl"], default = "tcp")
ap.add_argument("--ctx", type = int, default = 8192)
ap.add_argument("--prompt", default = "Explain pipeline parallelism in three sentences.")
ap.add_argument("--max_new", type = int, default = 200)
ap.add_argument("--nll_file", default = None, help = "rank 0: text file to score (next-token NLL) before generating")
ap.add_argument("--chunk", type = int, default = 512)
args = ap.parse_args()

R = args.rank
addrs = args.addrs.split(",")
W = len(addrs)
splits = [tuple(int(v) for v in s.split(":")) for s in args.splits.split(",")]
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
T0 = time.time()

def log(*x):
    print(f"[rank {R} {time.time() - T0:7.1f}s]", *x, flush = True)

# TCP: accept the upstream link while loading
up_box = {}
if args.transport == "tcp":
    lport = args.port + 1 if R == 0 else args.port
    lt = threading.Thread(target = lambda: up_box.setdefault("ep", NetEndpoint.listen("0.0.0.0", lport, timeout = 7200)),
                          daemon = True)
    lt.start()

# Build the full module list, load only this rank's slice
config = Config.from_directory(args.model)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = args.ctx)
nl = config.num_hidden_layers
fb = model.first_block_idx
assert len(splits) == W and splits[0][0] == 0 and splits[-1][1] == nl
assert all(splits[i][1] == splits[i + 1][0] for i in range(W - 1))
it = getattr(config, "indexer_types", None)
if it:
    for a, _ in splits[1:]:
        assert it[a] == "full", f"rank split at layer {a} is not a DSA 'full' indexer layer"
a, b = splits[R]
lo = 0 if R == 0 else fb + a
hi = len(model.modules) if R == W - 1 else fb + b
my_modules = model.modules[lo:hi]
my_fwd = model.fwd_modules[lo:hi]

my_files = {f for k, f in config.stc.tensor_file_map.items() if any(k.startswith(m.key + ".") for m in my_modules)}
def drop_page_cache():
    # Unified-memory hosts: release this slice's file pages as they are consumed
    for f in my_files:
        try:
            fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        except (OSError, AttributeError):
            pass

t = time.time()
with torch.inference_mode():
    for module in my_modules:
        defer = module.can_defer_load()
        if defer: config.stc.begin_deferred_load(arena = True)
        module.load(dev)
        if defer: config.stc.end_deferred_load()
        drop_page_cache()
config.stc.close()
model.active_devices = [0]
model.output_device = dev
g_tensor_cache.drop_all()
free_mem()
for ref in model.cache_weakrefs.values():
    c = ref()
    if c is not None: c.initialized = True
log(f"layers {a}..{b - 1}: {len(my_modules)} modules in {time.time() - t:.1f} s, "
    f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated")

# Links: up = previous rank (rank 0: the last rank's results), down = next rank (last: rank 0)
if args.transport == "nccl":
    from exllamav3.model.nccl_transport import NcclEndpoint
    NcclEndpoint.init_group(R, W, addrs[0], args.port + 7, device = dev)
    up, down = NcclEndpoint((R - 1) % W, dev), NcclEndpoint((R + 1) % W, dev)
else:
    down = NetEndpoint.connect(addrs[(R + 1) % W], args.port + 1 if R == W - 1 else args.port, timeout = 7200)
    lt.join()
    up = up_box["ep"]
log(f"{args.transport} links up")

stats = {"compute_ms": [], "steps": 0}

@torch.inference_mode()
def run_slice(ids, x, past_len, last_only):
    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": past_len, "batch_shape": (1, args.ctx)}
    model.prepare_inputs(ids, params)
    if R == 0:
        x = ids
    for module, instance, _ in my_fwd:
        params["layer_instance"] = instance
        if module.caps.get("logits_output") and last_only:
            x = x[..., -1:, :].contiguous()
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
    torch.cuda.synchronize()
    return x

def timed(*a):
    t = time.perf_counter()
    y = run_slice(*a)
    stats["compute_ms"].append((time.perf_counter() - t) * 1e3)
    stats["steps"] += 1
    return y

def head_result(y, ids, msg, state):
    logits = y.float()
    if msg["mode"] == "gen":
        lp = torch.log_softmax(logits[0, -1], dim = -1)
        top = torch.topk(lp, 2)
        return {"token": int(top.indices[0]), "margin": float(top.values[0] - top.values[1])}
    lp = torch.log_softmax(logits[0], dim = -1)
    tgt = ids[0].to(dev)
    rows, tg = [], []
    if msg["past_len"] > 0 and state.get("prev") is not None:
        rows.append(state["prev"][None]); tg.append(tgt[:1])
    rows.append(lp[:-1]); tg.append(tgt[1:])
    L, G = torch.cat(rows), torch.cat(tg)
    state["prev"] = lp[-1].clone()
    return {"nll_sum": float(-L.gather(1, G[:, None]).sum()), "n": int(G.numel()),
            "top1": int((L.argmax(-1) == G).sum())}

def compute_summary():
    c = sorted(stats["compute_ms"][-200:])
    return {"rank": R, "steps": stats["steps"], "compute_ms_p50": round(c[len(c) // 2], 2) if c else None}

def service():
    state = {}
    while True:
        msg = up.recv_obj()
        if msg["cmd"] == "stop":
            msg["ranks"].append(compute_summary())
            (down.send_obj(msg) if R < W - 1 else down.send_obj({"cmd": "stopped", "ranks": msg["ranks"]}))
            return
        ids = torch.tensor([msg["ids"]], dtype = torch.long)
        x = up.recv_tensor(device = dev)
        y = timed(ids.to(dev), x, msg["past_len"], msg["last_only"])
        if R < W - 1:
            down.send_obj(msg); down.send_tensor(y)
        else:
            down.send_obj(head_result(y, ids, msg, state))

def drive():
    tok = Tokenizer.from_config(config)

    def step(ids, past_len, mode, last_only):
        y = timed(ids.to(dev), None, past_len, last_only)
        # token ids ride in the control message (one object + one tensor per hop)
        down.send_obj({"cmd": "fwd", "past_len": past_len, "mode": mode, "last_only": last_only,
                       "ids": ids[0].tolist()})
        down.send_tensor(y)
        return up.recv_obj()

    if args.nll_file:
        ids = tok.encode(open(args.nll_file).read(), add_bos = False)[:, : args.ctx - 16]
        tot = cnt = top1 = 0
        t = time.time()
        for p in range(0, ids.shape[-1], args.chunk):
            r = step(ids[:, p:p + args.chunk], p, "nll", False)
            tot += r["nll_sum"]; cnt += r["n"]; top1 += r["top1"]
        log(f"NLL {tot / cnt:.6f}  PPL {math.exp(tot / cnt):.3f}  top-1 {top1 / cnt:.4f}  over {cnt} tokens, "
            f"prefill {ids.shape[-1] / (time.time() - t):.1f} tok/s")

    stop_ids = {tok.eos_token_id}
    for s in ("<|user|>", "<|endoftext|>", "<|observation|>", "<|im_end|>"):
        try: stop_ids.add(tok.single_id(s))
        except Exception: pass
    ids = tok.encode(model.default_chat_prompt(args.prompt), add_bos = False, encode_special_tokens = True)
    if ids.shape[-1] > 1:
        step(ids[:, :-1], 0, "nll", False)
    past, cur, out, lat = ids.shape[-1] - 1, ids[:, -1:], [], []
    for _ in range(args.max_new):
        t = time.perf_counter()
        r = step(cur, past, "gen", True)
        lat.append(time.perf_counter() - t)
        out.append(r["token"]); past += 1
        cur = torch.tensor([[r["token"]]], dtype = torch.long)
        if r["token"] in stop_ids:
            break
    print(tok.decode(torch.tensor([out]), decode_special_tokens = True)[0], flush = True)
    s = sorted(lat)
    log(f"decode {len(lat)} tokens: {len(lat) / sum(lat):.2f} tok/s, per token p50 {1e3 * s[len(s) // 2]:.1f} ms")
    down.send_obj({"cmd": "stop", "ranks": [compute_summary()]})
    r = up.recv_obj()
    tot = sum(x["compute_ms_p50"] or 0 for x in r["ranks"])
    log(f"per-rank compute p50 (ms): {[x['compute_ms_p50'] for x in r['ranks']]}, sum {tot:.1f}; "
        f"links + host ~ {1e3 * s[len(s) // 2] - tot:.1f} ms per token")

try:
    drive() if R == 0 else service()
finally:
    up.close(); down.close()
    if args.transport == "nccl":
        torch.distributed.destroy_process_group()
