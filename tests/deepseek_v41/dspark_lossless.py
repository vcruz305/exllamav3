#!/usr/bin/env python3
"""DeepSeek-V4.1 DSpark speculative decoding: lossless greedy check, acceptance and speed.

Builds the target and the DSpark drafter from the same pack the way model_init does for --mtp
(same config, component="mtp"), then generates greedily with and without the drafter from a few
prompts. Pass = identical token ids for every prompt. Reports mean accepted draft tokens per
verification round, acceptance rate and decode tokens/s for both runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

TREE = os.environ.get("EXL3_TREE")
if TREE:
    sys.path.insert(0, TREE)

PROMPTS = [
    "The history of the printing press begins",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain, step by step, why the sky appears blue during the day and red at sunset.",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default = os.path.expanduser("~/tp1/v41port/model"))
    p.add_argument("--split", type = int, default = 256)
    p.add_argument("--tokens", type = int, default = 128)
    p.add_argument("--cache-tokens", type = int, default = 8192)
    p.add_argument("--output", default = None)
    args = p.parse_args()

    os.environ.setdefault("EXL3_MOE_CPU_SPLIT", str(args.split))
    os.environ.setdefault("EXL3_MOE_CPU_THREADS", "16")
    os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
    os.environ.setdefault("EXL3_MOE_STREAM_MIN_ROWS", "1000000000")

    import torch
    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.cache.cache import Cache
    from exllamav3.generator import Generator, Job
    from exllamav3.generator.sampler import GreedySampler

    config = Config.from_directory(args.model)
    assert "mtp" in config.model_classes, "pack has no DSpark (mtp) component"
    model = Model.from_config(config)
    draft = Model.from_config(config, component = "mtp")
    max_history = draft.caps.get("default_draft_size", 4)
    cache = Cache(model, max_num_tokens = args.cache_tokens, max_history = max_history, max_batch_size = 1)
    draft_cache = Cache(draft, max_num_tokens = args.cache_tokens)
    t0 = time.time()
    model.load("cuda:0", progressbar = False, verbose = False)
    draft.load("cuda:0", progressbar = False, verbose = False)
    load_s = time.time() - t0
    tokenizer = Tokenizer.from_config(config)
    print(f"loaded in {load_s:.1f}s, drafter block size {max_history}", flush = True)

    def run(generator, prompt):
        ids = tokenizer.encode(prompt, add_bos = True)
        job = Job(input_ids = ids, max_new_tokens = args.tokens, stop_conditions = [], sampler = GreedySampler())
        generator.enqueue(job)
        out, last = [], None
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                stage = r.get("stage")
                if stage == "error":
                    info = {k: v for k, v in r.items() if not torch.is_tensor(v) and k != "job"}
                    print("GENERATOR ERROR:", info, flush = True)
                    err = r.get("error")
                    if isinstance(err, BaseException):
                        print("".join(__import__("traceback").format_exception(err)), flush = True)
                    raise RuntimeError(f"generator job failed: {info}")
                if stage != "streaming":
                    continue
                t = r.get("token_ids")
                if t is not None and t.numel():
                    out.append(t.flatten().cpu())
                last = r
        toks = torch.cat(out).tolist() if out else []
        tps = last["new_tokens"] / last["time_generate"] if last and last.get("time_generate") else 0.0
        return toks, tps, last.get("accepted_draft_tokens"), last.get("rejected_draft_tokens"), last.get("new_tokens")

    base_gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = 2048)
    base = [run(base_gen, pr) for pr in PROMPTS]
    del base_gen

    spec_gen = Generator(model = model, cache = cache, tokenizer = tokenizer, draft_model = draft,
                         draft_cache = draft_cache, max_chunk_size = 2048)
    spec = [run(spec_gen, pr) for pr in PROMPTS]

    rows, lossless = [], True
    for pr, (bt, btps, *_), (st, stps, dacc, drej, n) in zip(PROMPTS, base, spec):
        same = bt == st
        lossless &= same
        first_diff = next((i for i, (a, b) in enumerate(zip(bt, st)) if a != b), None if len(bt) == len(st) else min(len(bt), len(st)))
        rounds = (n - dacc) if (n is not None and dacc is not None) else None
        rows.append({
            "prompt": pr[:40], "tokens": len(bt), "identical": same, "first_diff": first_diff,
            "base_tps": round(btps, 2), "spec_tps": round(stps, 2),
            "accepted_per_round": round(dacc / rounds, 3) if rounds else None,
            "acceptance_rate": round(dacc / (dacc + drej), 3) if dacc is not None and (dacc + drej) else None,
        })
        print(json.dumps(rows[-1]), flush = True)
        print("  base text:", repr(tokenizer.decode(torch.tensor(bt[:60]))), flush = True)

    result = {"lossless": lossless, "load_s": load_s, "prompts": rows, "split": args.split}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent = 2)
    print("DSPARK_LOSSLESS_OK" if lossless else "DSPARK_LOSSLESS_FAIL")
    return 0 if lossless else 1


if __name__ == "__main__":
    sys.exit(main())
