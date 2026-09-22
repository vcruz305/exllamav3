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


class RedirectResponse:
    def __init__(self, address, status, location=None, body=b"abcd"):
        self.url = address
        self.status_code = status
        self.headers = ({"Location": location} if location else {})
        if status == 206:
            self.headers.update({"Content-Range": "bytes 0-3/100", "Content-Length": "4", "ETag": '"stable"'})
        self.body = body
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from (self.body[i:i + chunk_size] for i in range(0, len(self.body), chunk_size))

    @property
    def content(self):
        return self.body

    def close(self):
        self.closed = True


@pytest.mark.parametrize("destination", [
    "http://127.0.0.1/private", "https://127.0.0.1/private", "https://example.org/private",
    "https://huggingface.co.evil.test/private", "https://huggingface.co:8443/private",
    "https://user:password@huggingface.co/private", "//localhost/private",
])
def test_range_redirect_rejects_unsafe_destination_before_followup(destination):
    module = extractor()
    initial = module.url(module.SOURCE_REPO, module.SOURCE_REV, "shard.safetensors")
    redirect = RedirectResponse(initial, 302, destination)

    class Client:
        def get(self, address, **kwargs):
            assert address == initial
            assert kwargs["allow_redirects"] is False
            return redirect

    with pytest.raises(ValueError, match="URL|redirect|host|HTTPS"):
        module.range_bytes(Client(), initial, 0, 3)
    assert redirect.closed


def test_range_redirect_allows_hf_https_and_reuses_safe_resolved_url():
    module = extractor()
    initial = module.url(module.SOURCE_REPO, module.SOURCE_REV, "shard.safetensors")
    cdn = "https://us.aws.cdn.hf.co/safe-shard"
    redirect = RedirectResponse(initial, 302, cdn)
    first = RedirectResponse(cdn, 206)
    second = RedirectResponse(cdn, 206)

    class Client:
        def __init__(self):
            self.calls = []

        def get(self, address, **kwargs):
            self.calls.append(address)
            assert kwargs["allow_redirects"] is False and kwargs["stream"] is True
            assert kwargs["headers"]["Range"] == "bytes=0-3"
            assert kwargs["headers"]["Accept-Encoding"] == "identity"
            return {initial: redirect, cdn: first if self.calls.count(cdn) == 1 else second}[address]

    client = Client()
    assert module.range_bytes(client, initial, 0, 3)[-1] == cdn
    assert module.range_bytes(client, cdn, 0, 3, total=100, etag='"stable"')[0] == b"abcd"
    assert client.calls == [initial, cdn, cdn]
    assert all(response.closed for response in (redirect, first, second))


def test_index_redirect_refuses_unsafe_destination_without_fetching_it():
    module = extractor()
    initial = module.url(module.SOURCE_REPO, module.SOURCE_REV, "model.safetensors.index.json")
    redirect = RedirectResponse(initial, 302, "http://127.0.0.1/private")

    class Client:
        def get(self, address, **kwargs):
            assert address == initial
            assert kwargs["allow_redirects"] is False
            return redirect

    with pytest.raises(ValueError, match="URL|redirect|host|HTTPS"):
        module.index(Client(), module.SOURCE_REPO, module.SOURCE_REV, "model.safetensors.index.json")
    assert redirect.closed


def test_range_redirect_hop_limit_closes_every_response():
    module = extractor()
    initial = module.url(module.SOURCE_REPO, module.SOURCE_REV, "shard.safetensors")
    responses = []

    class Client:
        def get(self, address, **kwargs):
            response = RedirectResponse(address, 302, f"https://huggingface.co/next-{len(responses)}")
            responses.append(response)
            assert len(responses) < 20
            return response

    with pytest.raises(ValueError, match="redirect"):
        module.range_bytes(Client(), initial, 0, 3)
    assert 1 < len(responses) < 20 and all(response.closed for response in responses)


def test_range_body_overflow_stops_stream_even_with_matching_headers():
    module = extractor()
    address = "https://us.aws.cdn.hf.co/shard"

    class Response(RedirectResponse):
        @property
        def content(self):
            raise AssertionError("Must not buffer response.content")

        def iter_content(self, chunk_size):
            yield b"abcd"
            yield b"extra"
            raise AssertionError("Must stop reading on overflow")

    response = Response(address, 206)

    class Client:
        def get(self, address, **kwargs):
            assert kwargs["stream"] is True
            return response

    with pytest.raises(ValueError, match="range|large|body"):
        module.range_bytes(Client(), address, 0, 3)
    assert response.closed


def test_index_body_cap_rejects_stream_without_buffering_or_json_parsing():
    module = extractor()
    address = module.url(module.SOURCE_REPO, module.SOURCE_REV, "model.safetensors.index.json")

    class Response(RedirectResponse):
        @property
        def content(self):
            raise AssertionError("Must not buffer response.content")

        def json(self):
            raise AssertionError("Must parse only bounded bytes")

        def iter_content(self, chunk_size):
            assert chunk_size <= 65536
            yield b"x" * module.MAX_INDEX_BYTES
            yield b"x"
            raise AssertionError("Must stop reading on overflow")

    response = Response(address, 200)

    class Client:
        def get(self, address, **kwargs):
            assert kwargs["stream"] is True
            return response

    with pytest.raises(ValueError, match="index|large|body"):
        module.index(Client(), module.SOURCE_REPO, module.SOURCE_REV, "model.safetensors.index.json")
    assert response.closed


def test_index_bounded_stream_keeps_exact_digest():
    module = extractor()
    address = module.url(module.QUANT_REPO, module.QUANT_REV, "model.safetensors.index.json")
    payload = b'{"weight_map":{"tensor":"shard"}}'

    class Response(RedirectResponse):
        @property
        def content(self):
            raise AssertionError("Must not buffer response.content")

    response = Response(address, 200, body=payload)

    class Client:
        def get(self, address, **kwargs):
            return response

    assert module.index(Client(), module.QUANT_REPO, module.QUANT_REV, "model.safetensors.index.json") == (
        {"tensor": "shard"}, hashlib.sha256(payload).hexdigest())
    assert response.closed


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
        url = "https://huggingface.co/shard"

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
        module.range_bytes(client, "https://huggingface.co/shard", 0, 3)
    assert client.kwargs["stream"] is True  # never buffer a full shard on ignored Range


@pytest.mark.parametrize("later_etag", [None, '"different"'])
def test_range_bytes_rejects_missing_or_changed_etag(later_etag):
    module = extractor()

    class FakeResponse:
        status_code = 206
        url = "https://huggingface.co/shard"

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
        url = "https://huggingface.co/shard"
        headers = {"Content-Range": "bytes 0-65535/100000", "Content-Length": "65536"}

        def raise_for_status(self):
            pass

        @property
        def content(self):
            return struct.pack("<Q", 70000) + b" " * (65536 - 8)

        def iter_content(self, chunk_size):
            yield self.content

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
        url = "https://huggingface.co/shard"
        headers = {"Content-Range": f"bytes 0-65535/{total}", "Content-Length": "65536", "ETag": '"pinned"'}

        def raise_for_status(self):
            pass

        @property
        def content(self):
            return struct.pack("<Q", total - 16) + b" " * (65536 - 8)

        def iter_content(self, chunk_size):
            yield self.content

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
        url = "https://huggingface.co/shard"

        def __init__(self, start, end):
            self.headers = {"Content-Range": f"bytes {start}-{end}/100000",
                            "Content-Length": str(end - start + 1), "ETag": '"stable"'}
            self.content = blob[start:end + 1]

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield self.content

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
    parsed, length, total, etag, _ = module.shard_header(client, "https://huggingface.co/shard")
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


@pytest.mark.parametrize("failed_artifact", ["safetensors", "manifest.json"])
def test_mid_write_failure_preserves_verified_pair_and_cleans_temps(tmp_path, monkeypatch, failed_artifact):
    module = extractor()
    expected = stub_offline_extraction(module, monkeypatch)
    monkeypatch.setattr(module, "EXPECTED_OVERLAY_SHA256", expected)
    module.main(["--output-dir", str(tmp_path)])
    originals = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    original_write_bytes = Path.write_bytes
    original_write_text = Path.write_text

    def fail_partway(path, data):
        if failed_artifact in path.name:
            with path.open("wb") as output:
                output.write(data[:8])
            raise OSError("injected mid-write disk failure")
        return original_write_bytes(path, data)

    def fail_text(path, data, **kwargs):
        if failed_artifact in path.name:
            with path.open("wb") as output:
                output.write(data.encode("utf-8")[:8])
            raise OSError("injected mid-write disk failure")
        return original_write_text(path, data, **kwargs)

    monkeypatch.setattr(Path, "write_bytes", fail_partway)
    monkeypatch.setattr(Path, "write_text", fail_text)
    with pytest.raises(OSError, match="injected mid-write"):
        module.main(["--output-dir", str(tmp_path)])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == originals
