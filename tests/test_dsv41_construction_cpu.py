"""CPU-only checks of V4.1 attention construction (DSV41Attention on layer_type "v41"); no GPU.

Each check compares against the fork's base-class behaviour for a "v41" layer: the base builds no
compressor, indexer or kv_cache capability; only the kv sources own a pool, keyed by their own layer;
DSV4LayerState allocates no compressor buffers; tp_export carries the kwargs41 fields that DSV41Attention takes.
"""
import inspect

import pytest
import torch

pytestmark = pytest.mark.nogpu

from exllamav3.cache.dsa import DSV4LayerState  # noqa: E402
from exllamav3.modules.dsv4 import DSV4Attention  # noqa: E402
from exllamav3.modules.dsv41 import DSV41Attention  # noqa: E402

DIMS = dict(
    hidden_size = 64, num_q_heads = 4, head_dim = 32, rope_head_dim = 8, q_lora_rank = 16, o_groups = 2,
    o_lora_rank = 8, sliding_window = 16, index_n_heads = 2, index_head_dim = 16, index_topk = 8,
    rope_theta = 10000.0, compress_rope_theta = 160000.0, rope_scaling = None, rms_norm_eps = 1e-6,
    out_dtype = torch.half, q_head_norm = False,
)

KWARGS41_FIELDS = [
    "is_kv_source", "is_index_source", "kv_source_layer", "index_source_layer", "candidate_role",
    "candidate_topk_blocks", "candidate_block_size", "ref_quant",
]


def _attn(layer_idx, is_kv_source, is_index_source, kv_source_layer, compress_rate):
    return DSV41Attention(
        config = None, key = f"layer{layer_idx}.attn", layer_idx = layer_idx, compress_rate = compress_rate,
        is_kv_source = is_kv_source, is_index_source = is_index_source, kv_source_layer = kv_source_layer,
        index_source_layer = None, candidate_role = None, **DIMS)


def test_shared_v41_layer_owns_no_pool():
    m = _attn(1, False, False, 0, 2)
    assert m.kv_source is None
    assert m.compressor is None
    assert m.indexer is None
    assert not m.caps.get("kv_cache")


def test_kv_source_v41_layer_owns_its_pool():
    m = _attn(0, True, True, 0, 2)
    assert m.kv_source == 0
    assert type(m.compressor).__name__ == "DSV41Compressor"
    assert m.caps.get("kv_cache")
    assert m.idx_wk is not None  # index keys from the pooled latent, owned by the kv source


def test_ratio1_kv_source_owns_its_pool():
    m = _attn(2, True, False, 2, 1)
    assert m.kv_source == 2
    assert type(m.compressor).__name__ == "DSV41Compressor"


def test_v41_layer_state_has_no_compressor_buffers():
    # the fork's DSV4LayerState allocated comp/idx buffers only for layer_type csa/hca
    m = _attn(0, True, True, 0, 2)
    ls = DSV4LayerState(m, 2, 64, 0)
    assert ls.comp_buf_kv is None and ls.comp_buf_gate is None
    assert ls.idx_buf_kv is None and ls.idx_buf_gate is None


def test_tp_export_carries_kwargs41_fields():
    # producer (DSV4Attention.tp_export) and consumer (tp_import: kw.update(exported["kwargs41"]))
    src = inspect.getsource(DSV4Attention.tp_export)
    assert '"kwargs41"' in src
    for field in KWARGS41_FIELDS:
        assert f'"{field}"' in src, field
    params = inspect.signature(DSV41Attention.__init__).parameters
    for field in KWARGS41_FIELDS:
        assert field in params, field
