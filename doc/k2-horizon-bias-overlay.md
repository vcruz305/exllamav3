# K2-Horizon MoVA: recover missing routing biases (6.50bpw only)

The **current** `vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3/6.50bpw` revision
`c88277ce7f6b90b723f79b5188c0f1b951732099` omits 90 learned selection biases:
45 `model.layers.{3..47}.self_attn.v_router.bias` (BF16 `[64]`) and
45 `model.layers.{3..47}.mlp.gate.bias` (BF16 `[100]`). Its 45 misnamed
`mlp.None` F32 tensors are not replacements (40 zero, five different nonzero).
The extractor copies the exact BF16 bytes from public source
`IFM/K2-Horizon-MoVA-36B-A4B` revision
`7730b92d1b574e04663b04023d5d6fa83475432f`. **Do not assume this
is needed or correct for other quant revisions, bitrates, or models.**

From the fork checkout, using Python 3.10+ and `requests`:

```sh
python -m pip install requests
python tools/extract_k2_routing_bias_overlay.py --output-dir /path/outside-repo/k2-biases
sha256sum /path/outside-repo/k2-biases/k2-routing-bias-overlay.safetensors
```

The expected SHA-256 for these pinned revisions is
`8038de808fb396f4d5d337d373435523f167bfbc8558b5a6af09c1900408f53c`.
The script checks both indices, relevant shard headers, byte ranges, shapes and
45 quant `.None` payloads, then verifies the **entire overlay SHA-256 against the
pinned digest above before writing either file**. It requires a stable ETag on
each shard range, refuses safetensors headers over 16 MiB before fetching them,
and disables `requests` environment trust (including implicit `.netrc` auth and
environment proxies). It uses HTTP ranges rather than fetching whole model
shards. It produces the overlay and a local JSON manifest
with revisions, index hashes, per-tensor provenance and hashes, and `.None`
classifications; reruns overwrite these two files **only in the output dir**.
Keep the output outside the checkout and do not commit the generated weights
or manifest to this repository.

Pass the overlay explicitly when loading a local **6.50bpw model directory**:

```python
from exllamav3 import Config
config = Config.from_directory(
    "/path/to/K2-Horizon-MoVA-36B-A4B-EXL3/6.50bpw",
    routing_bias_overlay="/path/outside-repo/k2-biases/k2-routing-bias-overlay.safetensors",
)
```

Do not modify or upload the quant repository; do not rename `.mlp.None` tensors.
K2 loading fails closed on missing required biases or a missing overlay path.
The router bias participates in **selection only**, not the projection logits.
Run offline regression checks with
`python -m pytest -q tests/test_k2_bias_overlay_extractor.py tests/test_k2_horizon.py`.
An earlier Spark2 6.50bpw overlay load and short cached decode (` Paris.`)
succeeded, but the latest source fixes have **not** been rerun on GPU; this
limited smoke is not a broad generation validation.
