"""Offline checks for the standalone, pinned K2 bias-overlay extractor."""
import contextlib
import hashlib
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


def test_session_disables_implicit_netrc_and_environment_credentials():
    module = extractor()
    with module.session() as client:
        assert client.trust_env is False


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


@pytest.mark.parametrize("later_etag", [None, '"different"'])
def test_range_bytes_rejects_missing_or_changed_etag(later_etag):
    module = extractor()

    class FakeResponse:
        status_code = 206
        url = "https://example.test/shard"

        def __init__(self, etag):
            self.headers = {"Content-Range": "bytes 8-11/100", "Content-Length": "4"}
            if etag is not None:
                self.headers["ETag"] = etag

        def raise_for_status(self):
            pass

        @property
        def content(self):
            raise AssertionError("Must reject before reading response body")

        def close(self):
            self.closed = True

    response = FakeResponse(later_etag)

    class FakeClient:
        def get(self, *args, **kwargs):
            assert kwargs["stream"] is True
            return response

    with pytest.raises(ValueError, match="ETag"):
        module.range_bytes(FakeClient(), response.url, 8, 11, total=100, etag='"original"')
    assert response.closed


def test_shard_header_rejects_missing_initial_etag_before_extra_ranges():
    module = extractor()

    class FakeResponse:
        status_code = 206
        url = "https://example.test/shard"
        headers = {"Content-Range": "bytes 0-65535/100000", "Content-Length": "65536"}

        def raise_for_status(self):
            pass

        @property
        def content(self):
            return struct.pack("<Q", 70000) + b" " * (65536 - 8)

        def close(self):
            self.closed = True

    response = FakeResponse()

    class FakeClient:
        def get(self, *args, **kwargs):
            self.calls += 1
            return response

        calls = 0

    client = FakeClient()
    with pytest.raises(ValueError, match="ETag"):
        module.shard_header(client, response.url)
    assert client.calls == 1
    assert response.closed


def test_shard_header_refuses_giant_header_before_fetching_it():
    module = extractor()
    total = 128 * 1024 * 1024

    class FakeResponse:
        status_code = 206
        url = "https://example.test/shard"
        headers = {"Content-Range": f"bytes 0-65535/{total}", "Content-Length": "65536", "ETag": '"pinned"'}

        def raise_for_status(self):
            pass

        @property
        def content(self):
            return struct.pack("<Q", total - 16) + b" " * (65536 - 8)

        def close(self):
            self.closed = True

    response = FakeResponse()

    class FakeClient:
        calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            assert kwargs["stream"] is True
            return response

    client = FakeClient()
    with pytest.raises(ValueError, match="header length"):
        module.shard_header(client, response.url)
    assert client.calls == 1
    assert response.closed


def test_shard_header_allows_bounded_large_header_with_stable_etag():
    module = extractor()
    header = b'{"tensor":{"dtype":"BF16"}}'.ljust(70000, b" ")
    blob = struct.pack("<Q", len(header)) + header

    class FakeResponse:
        status_code = 206
        url = "https://example.test/shard"

        def __init__(self, start, end):
            self.headers = {"Content-Range": f"bytes {start}-{end}/100000",
                            "Content-Length": str(end - start + 1), "ETag": '"stable"'}
            self.content = blob[start:end + 1]

        def raise_for_status(self):
            pass

        def close(self):
            self.closed = True

    class FakeClient:
        def __init__(self):
            self.calls = []

        def get(self, address, *, headers, **kwargs):
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            self.calls.append((start, end))
            return FakeResponse(start, end)

    client = FakeClient()
    parsed, length, total, etag, _ = module.shard_header(client, "https://example.test/shard")
    assert parsed == {"tensor": {"dtype": "BF16"}}
    assert (length, total, etag) == (70000, 100000, '"stable"')
    assert client.calls == [(0, 65535), (8, 70007)]


def stub_offline_extraction(module, monkeypatch):
    source = {f"model.layers.{layer}.{part}.bias": "source-shard" for layer in range(3, 48)
              for part in module.SIZES}
    quant = {f"model.layers.{layer}.mlp.None": "quant-shard" for layer in range(3, 48)}
    monkeypatch.setattr(module, "session", lambda: contextlib.nullcontext(object()))
    monkeypatch.setattr(module, "index", lambda client, repo, rev, filename:
                        (source if repo == module.SOURCE_REPO else quant, "index-hash"))

    def fake_extract(shard, keys):
        assert shard == "source-shard"
        return [(key, b"\x00\x00" * module.SIZES[module.PATTERN.fullmatch(key).group(2)],
                 {"shape": [module.SIZES[module.PATTERN.fullmatch(key).group(2)]]}) for key in keys]

    monkeypatch.setattr(module, "extract_shard", fake_extract)
    monkeypatch.setattr(module, "inspect_quant", lambda shard, keys: {key: b"\x00" * 400 for key in keys})
    records = fake_extract("source-shard", list(source))
    return hashlib.sha256(module.make_overlay({k: raw for k, raw, _ in records},
                                              {k: meta for k, _, meta in records})).hexdigest()


def test_wrong_final_digest_refuses_output_artifacts(tmp_path, monkeypatch):
    module = extractor()
    stub_offline_extraction(module, monkeypatch)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA-256"):
        module.main(["--output-dir", str(output)])
    assert not output.exists()


def test_pinned_digest_matches_documented_verified_digest():
    module = extractor()
    verified = "8038de808fb396f4d5d337d373435523f167bfbc8558b5a6af09c1900408f53c"
    guide = SCRIPT.parents[1] / "doc" / "k2-horizon-bias-overlay.md"
    assert verified in guide.read_text(encoding="utf-8")
    assert module.EXPECTED_OVERLAY_SHA256 == verified


def test_matching_final_digest_writes_overlay_and_manifest(tmp_path, monkeypatch):
    module = extractor()
    expected = stub_offline_extraction(module, monkeypatch)
    monkeypatch.setattr(module, "EXPECTED_OVERLAY_SHA256", expected)
    module.main(["--output-dir", str(tmp_path)])
    overlay = tmp_path / "k2-routing-bias-overlay.safetensors"
    manifest = tmp_path / "k2-routing-bias-overlay.manifest.json"
    assert hashlib.sha256(overlay.read_bytes()).hexdigest() == expected
    assert json.loads(manifest.read_text(encoding="utf-8"))["overlay_sha256"] == expected
