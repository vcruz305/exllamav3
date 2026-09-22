"""Recover K2-Horizon learned routing biases as a standalone BF16 safetensors overlay.

Public, pinned Hugging Face endpoints only. No credentials or implicit model
directory writes; no full-shard downloads. BF16 bytes are copied as-is.
Run from anywhere: python extract_k2_routing_bias_overlay.py --output-dir <directory>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import struct
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SOURCE_REPO = "IFM/K2-Horizon-MoVA-36B-A4B"
SOURCE_REV = "7730b92d1b574e04663b04023d5d6fa83475432f"
QUANT_REPO = "vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3"
QUANT_REV = "c88277ce7f6b90b723f79b5188c0f1b951732099"
QUANT_SUBDIR = "6.50bpw/"
PATTERN = re.compile(r"^model\.layers\.(\d+)\.(self_attn\.v_router|mlp\.gate)\.bias$")
NONE_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.None$")
SIZES = {"self_attn.v_router": 64, "mlp.gate": 100}


def url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def session() -> requests.Session:
    client = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    client.mount("https://", HTTPAdapter(max_retries=retry))
    return client


def range_bytes(client: requests.Session, address: str, start: int, end: int,
                *, total: int | None = None, etag: str | None = None) -> tuple[bytes, int, str | None, str]:
    # Inspect status/headers before buffering: a CDN ignoring Range may send a full shard.
    response = client.get(address, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                          timeout=90, stream=True)
    try:
        response.raise_for_status()
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status_code != 206 or not match:
            raise ValueError(f"HTTP range not honored: {response.status_code} {address.split('?')[0]}")
        lo, hi, size = map(int, match.groups())
        if (lo, hi) != (start, end) or int(response.headers.get("Content-Length", end - start + 1)) != end - start + 1:
            raise ValueError(f"Truncated/wrong range {lo}-{hi} for {start}-{end}")
        if total is not None and total != size:
            raise ValueError("Shard size changed between range requests")
        got_etag = response.headers.get("ETag")
        if etag and got_etag and etag != got_etag:
            raise ValueError("Shard ETag changed between range requests")
        raw = response.content
        if len(raw) != end - start + 1:
            raise ValueError(f"Truncated/wrong range {lo}-{hi} for {start}-{end}")
        return raw, size, got_etag, response.url
    finally:
        response.close()


def shard_header(client: requests.Session, address: str) -> tuple[dict, int, int, str | None, str]:
    # Source shard headers fit in 64 KiB; large quant headers use the fallback.
    first, total, etag, resolved = range_bytes(client, address, 0, 65535)
    length = struct.unpack("<Q", first[:8])[0]
    if not 0 < length < total - 8:
        raise ValueError(f"Invalid safetensors header length: {length}")
    if length + 8 > len(first):
        header, _, _, resolved = range_bytes(client, resolved, 8, 7 + length, total=total, etag=etag)
    else:
        header = first[8:8 + length]
    parsed = json.loads(header)
    if not isinstance(parsed, dict):
        raise ValueError("Invalid safetensors header")
    return parsed, length, total, etag, resolved


def index(client: requests.Session, repo: str, revision: str, filename: str) -> tuple[dict, str]:
    response = client.get(url(repo, revision, filename), timeout=90)
    response.raise_for_status()
    return response.json()["weight_map"], hashlib.sha256(response.content).hexdigest()


def inspect_quant(shard: str, keys: list[str]) -> dict[str, bytes]:
    with session() as client:
        header, length, total, etag, resolved = shard_header(client, url(QUANT_REPO, QUANT_REV, QUANT_SUBDIR + shard))
        if any(PATTERN.fullmatch(k) for k in header):
            raise ValueError(f"Quant shard {shard} contains a routing bias despite index absence")
        if {k for k in header if NONE_PATTERN.fullmatch(k)} != set(keys):
            raise ValueError(f"Quant index/header .mlp.None discrepancy in {shard}")
        results = {}
        for key in keys:
            info = header[key]
            if info["dtype"] != "F32" or info["shape"] != [100]:
                raise ValueError(f"Unexpected bogus tensor metadata: {key}: {info}")
            begin, end = info["data_offsets"]
            if end - begin != 400:
                raise ValueError(f"Wrong .mlp.None data length: {key}")
            raw, _, _, _ = range_bytes(client, resolved, 8 + length + begin, 7 + length + end,
                                       total=total, etag=etag)
            results[key] = raw
    return results


def extract_shard(shard: str, keys: list[str]) -> list[tuple[str, bytes, dict]]:
    with session() as client:
        header, length, total, etag, resolved = shard_header(client, url(SOURCE_REPO, SOURCE_REV, shard))
        result = []
        for key in keys:
            if key not in header:
                raise ValueError(f"Source index/header mismatch: {shard}: {key}")
            info = header[key]
            match = PATTERN.fullmatch(key)
            assert match is not None
            expected_len = SIZES[match.group(2)] * 2
            begin, end = info["data_offsets"]
            if info["dtype"] != "BF16" or info["shape"] != [SIZES[match.group(2)]] or end - begin != expected_len:
                raise ValueError(f"Wrong source dtype/shape/size: {key}: {info}")
            raw, _, _, _ = range_bytes(client, resolved, 8 + length + begin, 7 + length + end,
                                       total=total, etag=etag)
            result.append((key, raw, {"source_shard": shard, "source_data_offsets": [begin, end],
                                      "source_file_byte_range": [8 + length + begin, 7 + length + end],
                                      "source_shard_etag": etag, "dtype": "BF16", "shape": info["shape"],
                                      "sha256": hashlib.sha256(raw).hexdigest()}))
        return result


def make_overlay(tensors: dict[str, bytes], metadata: dict[str, dict]) -> bytes:
    offset = 0
    header = {}
    for key in sorted(tensors):
        size = len(tensors[key])
        header[key] = {"dtype": "BF16", "shape": metadata[key]["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-(8 + len(encoded)) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(tensors[k] for k in sorted(tensors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for the generated .safetensors overlay and JSON manifest (outside the checkout recommended)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    output_dir = parse_args(argv).output_dir
    overlay_path = output_dir / "k2-routing-bias-overlay.safetensors"
    manifest_path = output_dir / "k2-routing-bias-overlay.manifest.json"
    with session() as client:
        source, source_sha = index(client, SOURCE_REPO, SOURCE_REV, "model.safetensors.index.json")
        quant, quant_sha = index(client, QUANT_REPO, QUANT_REV, QUANT_SUBDIR + "model.safetensors.index.json")
    expected = {f"model.layers.{layer}.{part}.bias" for layer in range(3, 48) for part in SIZES}
    actual = {key for key in source if PATTERN.fullmatch(key)}
    quant_actual = {key for key in quant if PATTERN.fullmatch(key)}
    none = {key for key in quant if NONE_PATTERN.fullmatch(key)}
    expected_none = {f"model.layers.{layer}.mlp.None" for layer in range(3, 48)}
    if actual != expected or quant_actual or none != expected_none:
        raise ValueError(f"Index discrepancy changed: source missing={sorted(expected - actual)}, "
                         f"source extras={sorted(actual - expected)}, quant biases={sorted(quant_actual)}, "
                         f"None missing={sorted(expected_none - none)}, None extras={sorted(none - expected_none)}")
    source_groups: dict[str, list[str]] = {}
    for key in sorted(expected):
        source_groups.setdefault(source[key], []).append(key)
    quant_groups: dict[str, list[str]] = {}
    for key in sorted(none):
        quant_groups.setdefault(quant[key], []).append(key)
    # Fetch in parallel by shard; each worker checks headers, ranges and immutable object identity.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        source_jobs = [pool.submit(extract_shard, shard, keys) for shard, keys in sorted(source_groups.items())]
        quant_jobs = [pool.submit(inspect_quant, shard, keys) for shard, keys in sorted(quant_groups.items())]
        records = [item for job in source_jobs for item in job.result()]
        quant_none_raw = {k: v for job in quant_jobs for k, v in job.result().items()}
    if len(records) != 90 or len(quant_none_raw) != 45:
        raise ValueError("Incomplete extraction")
    tensors = {key: raw for key, raw, _ in records}
    details = {key: detail for key, _, detail in records}
    if len(tensors) != 90:
        raise ValueError("Duplicate source tensor name")
    none_classifications = {}
    for key, raw in sorted(quant_none_raw.items()):
        source_key = key.removesuffix(".None") + ".gate.bias"
        expected_f32 = b"".join(b"\x00\x00" + tensors[source_key][i:i + 2]
                                for i in range(0, len(tensors[source_key]), 2))
        status = "source_gate_bias_widened_f32" if raw == expected_f32 else "all_zero" if not any(raw) else "different_nonzero"
        none_classifications[key] = {"classification": status, "sha256": hashlib.sha256(raw).hexdigest()}
    overlay = make_overlay(tensors, details)
    manifest = {
        "schema": "k2-routing-bias-overlay-v1", "source_repo": SOURCE_REPO, "source_revision": SOURCE_REV,
        "source_index_sha256": source_sha, "quant_repo": QUANT_REPO, "quant_revision": QUANT_REV,
        "quant_subdir": QUANT_SUBDIR.rstrip("/"), "quant_index_sha256": quant_sha,
        "tensor_count": len(tensors),
        "quant_mlp_None": none_classifications,
        "quant_mlp_None_classification_counts": {status: sum(v["classification"] == status for v in none_classifications.values())
                                                   for status in ("source_gate_bias_widened_f32", "all_zero", "different_nonzero")},
        "overlay_file": overlay_path.name, "overlay_sha256": hashlib.sha256(overlay).hexdigest(),
        "tensors": {key: details[key] for key in sorted(details)},
        "note": "All 90 bias names are absent from the 6.50bpw quant index and shard headers. 45 quant .mlp.None tensors are F32 and misnamed; their content classification is recorded separately. No equivalent v_router.bias tensor name was found. This overlay is not installed into the quant model."
    }
    # Complete all remote checks before writing either local output.
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path.write_bytes(overlay)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"tensor_count": len(tensors), "quant_None_classifications": manifest["quant_mlp_None_classification_counts"],
                      "overlay_sha256": manifest["overlay_sha256"], "overlay": str(overlay_path),
                      "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
