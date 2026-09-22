"""Offline checks for the standalone, pinned K2 bias-overlay extractor."""
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "extract_k2_routing_bias_overlay.py"


def extractor():
    spec = importlib.util.spec_from_file_location("k2_bias_extractor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_requires_explicit_output_directory(tmp_path):
    module = extractor()
    with pytest.raises(SystemExit):
        module.parse_args([])
    args = module.parse_args(["--output-dir", str(tmp_path)])
    assert args.output_dir == tmp_path
    assert not hasattr(module, "OVERLAY")  # no implicit writes alongside the script


def test_help_does_not_fetch_or_write(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert "--output-dir" in result.stdout
    assert not list(tmp_path.iterdir())


def test_overlay_preserves_exact_bf16_bytes_and_is_deterministic():
    module = extractor()
    tensors = {"model.layers.3.self_attn.v_router.bias": b"\x34\x12\x78\x56",
               "model.layers.3.mlp.gate.bias": b"\xab\xcd"}
    metadata = {key: {"shape": [len(raw) // 2]} for key, raw in tensors.items()}
    blob = module.make_overlay(tensors, metadata)
    header_len = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + header_len])
    assert (header_len + 8) % 8 == 0
    for key, raw in tensors.items():
        info = header[key]
        assert info["dtype"] == "BF16" and info["shape"] == metadata[key]["shape"]
        lo, hi = info["data_offsets"]
        assert blob[8 + header_len + lo:8 + header_len + hi] == raw
    assert module.make_overlay(dict(reversed(list(tensors.items()))), metadata) == blob


def test_range_bytes_rejects_full_response():
    module = extractor()

    class FakeResponse:
        status_code = 200
        headers = {}
        content = b"data"
        url = "https://example.test/shard"

        def raise_for_status(self):
            pass

        def close(self):
            self.closed = True

    class FakeClient:
        def get(self, *args, **kwargs):
            self.kwargs = kwargs
            return FakeResponse()

    client = FakeClient()
    with pytest.raises(ValueError, match="range not honored"):
        module.range_bytes(client, "https://example.test/shard", 0, 3)
    assert client.kwargs["stream"] is True  # never buffer a full shard on ignored Range
