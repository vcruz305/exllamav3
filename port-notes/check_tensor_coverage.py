#!/usr/bin/env python3
"""Static tensor-coverage gate for the Step-5 MTP + CSA indexer port.

Checks that every tensor family in the checkpoint index is produced by exactly one
module in the ported module tree. Runs WITHOUT torch (it inspects key strings, not a
built model), so it is cheap enough to run before renting a GPU.

Groups:
  indexer  -> Step5CSAIndexer + Step5SSMaxScale   (this port)
  mtp      -> Step5RoboticsMTPModel                (this port)
  body     -> Step3_5Model                         (proven by the prior encodes)
  vision   -> excluded by design

The real coverage gate after a build is convert_model's own report (2449/2449, zero
unmatched). This is the pre-flight.
"""
import argparse
import collections
import json
import re
import sys
import urllib.request
from pathlib import Path

INDEX_URL = ("https://huggingface.co/rene98c/Step-5-Preview-BF16/"
             "resolve/main/model.safetensors.index.json")

# What each ported module claims, as a regex on the layer-normalised key.
INDEXER_CLAIMS = {
    "sparse_indexer_q.weight":        r"\.self_attn\.sparse_indexer_q\.weight$",
    "sparse_indexer_q_norm.weight":   r"\.self_attn\.sparse_indexer_q_norm\.weight$",
    "sparse_indexer_k.weight":        r"\.self_attn\.sparse_indexer_k\.weight$",
    "sparse_indexer_k_norm.weight":   r"\.self_attn\.sparse_indexer_k_norm\.weight$",
    "sparse_indexer_k_norm.bias":     r"\.self_attn\.sparse_indexer_k_norm\.bias$",
    "sparse_indexer_w.weight":        r"\.self_attn\.sparse_indexer_w\.weight$",
    "sparse_indexer_z.weight":        r"\.self_attn\.sparse_indexer_z\.weight$",
    "ssmax_s":                        r"\.self_attn\.ssmax_s$",
}
MTP_CLAIMS = {
    "eh_proj.weight":                        r"\.eh_proj\.weight$",
    "enorm.weight":                          r"\.enorm\.weight$",
    "hnorm.weight":                          r"\.hnorm\.weight$",
    "input_layernorm.weight":                r"\.input_layernorm\.weight$",
    "mlp.down_proj.weight":                  r"\.mlp\.down_proj\.weight$",
    "mlp.gate_proj.weight":                  r"\.mlp\.gate_proj\.weight$",
    "mlp.up_proj.weight":                    r"\.mlp\.up_proj\.weight$",
    "post_attention_layernorm.weight":       r"\.post_attention_layernorm\.weight$",
    "self_attn.g_proj.weight":               r"\.self_attn\.g_proj\.weight$",
    "self_attn.k_norm.weight":               r"\.self_attn\.k_norm\.weight$",
    "self_attn.k_proj.weight":               r"\.self_attn\.k_proj\.weight$",
    "self_attn.o_proj.weight":               r"\.self_attn\.o_proj\.weight$",
    "self_attn.q_norm.weight":               r"\.self_attn\.q_norm\.weight$",
    "self_attn.q_proj.weight":               r"\.self_attn\.q_proj\.weight$",
    "self_attn.v_proj.weight":               r"\.self_attn\.v_proj\.weight$",
    "transformer.shared_head.norm.weight":   r"\.transformer\.shared_head\.norm\.weight$",
    "transformer.shared_head.output.weight": r"\.transformer\.shared_head\.output\.weight$",
}


def fetch_index() -> dict:
    tokp = Path.home() / ".hf_artifacts_token"
    headers = {"User-Agent": "curl/8"}
    if tokp.exists():
        headers["Authorization"] = "Bearer " + tokp.read_text().strip()
    req = urllib.request.Request(INDEX_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["weight_map"]


def classify(k: str) -> str:
    if "sparse_indexer" in k or k.endswith(".ssmax_s"):
        return "indexer"
    if re.search(r"\.layers\.(9[2-4])\.", k):
        return "mtp"
    if "vision_model" in k or k.startswith("vit."):
        return "vision"
    return "body"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", help="local copy of model.safetensors.index.json")
    a = ap.parse_args()

    wm = (json.loads(Path(a.index).read_text())["weight_map"] if a.index else fetch_index())
    groups = collections.defaultdict(list)
    for k in wm:
        groups[classify(k)].append(k)

    fails = []
    for gname, claims, expect_per in (("indexer", INDEXER_CLAIMS, 23), ("mtp", MTP_CLAIMS, 3)):
        print(f"=== {gname}: {len(groups[gname])} tensors ===")
        claimed = 0
        for label, pat in claims.items():
            rx = re.compile(pat)
            hits = [k for k in groups[gname] if rx.search(k)]
            claimed += len(hits)
            status = "ok" if len(hits) == expect_per else "BAD"
            if status == "BAD":
                fails.append(f"{gname}:{label} got {len(hits)} expected {expect_per}")
            print(f"  {status:3} {len(hits):3}/{expect_per}  {label}")
        unclaimed = [k for k in groups[gname]
                     if not any(re.search(p, k) for p in claims.values())]
        if unclaimed:
            fails.append(f"{gname}: {len(unclaimed)} unclaimed keys e.g. {unclaimed[:3]}")
        print(f"  claimed {claimed}/{len(groups[gname])}, unclaimed {len(unclaimed)}")

    print(f"\nbody={len(groups['body'])} (Step3_5Model, proven)  "
          f"vision={len(groups['vision'])} (excluded by design)")
    total = len(wm)
    ported = len(groups["indexer"]) + len(groups["mtp"])
    print(f"TOTAL={total}  ported by this change={ported}")

    if fails:
        print("\nFAIL")
        for f in fails:
            print("  -", f)
        return 3
    print("\nCOVERAGE OK -- every indexer + MTP tensor is claimed by a ported module")
    return 0


if __name__ == "__main__":
    sys.exit(main())
