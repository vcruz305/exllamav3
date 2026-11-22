"""CPU-only test of BCAttn's window_size selection; no native extension, Triton, or GPU.

Loads bc_attn.py under a private package name with its native imports stubbed, then checks
_module_window_size, which BCAttn.__init__ calls for self.window_size.
"""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.nogpu

ROOT = Path(__file__).resolve().parents[1] / "exllamav3"
PREFIX = "bc_window_cpu_exllamav3"


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _module(name, **members):
    module = types.ModuleType(name)
    module.__dict__.update(members)
    sys.modules[name] = module
    return module


@pytest.fixture(scope="module")
def bc_attn():
    _package(PREFIX)
    _package(PREFIX + ".modules")
    _package(PREFIX + ".util")
    _package(PREFIX + ".modules.attention_fn")
    _module(PREFIX + ".util.backend", ROCM=False, ATTN_SPLIT_WARPS_STAGES=(4, 2),
            attn_decode_config=lambda *args, **kwargs: None)
    _module(PREFIX + ".util.tensor", g_tensor_cache=None)
    _module(PREFIX + ".constants", PAGE_SIZE=256)
    _module(PREFIX + ".modules.attention_fn.smem", smem_limit=lambda *args: 98304)
    _module(PREFIX + ".ext", exllamav3_ext=SimpleNamespace())
    spec = importlib.util.spec_from_file_location(
        PREFIX + ".modules.attention_fn.bc_attn", ROOT / "modules/attention_fn/bc_attn.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_window_arg_int_is_used_as_is(bc_attn):
    # fork attn.py: window_arg() returns the plain sliding window when window_right is 0
    m = SimpleNamespace(window_arg=lambda: 128, sliding_window=128, window_right=0)
    assert bc_attn._module_window_size(m) == 128


def test_window_arg_tuple_is_used_as_is(bc_attn):
    # fork attn.py: window_arg() returns (left, right) when window_right is set; must not reach `sw >= 0`
    m = SimpleNamespace(window_arg=lambda: (128, 8), sliding_window=128, window_right=8)
    assert bc_attn._module_window_size(m) == (128, 8)


def test_window_arg_tuple_with_negative_sliding_window(bc_attn):
    # fork attn.py returns the tuple whatever the sign of sliding_window; the helper must not re-wrap it
    m = SimpleNamespace(window_arg=lambda: (-1, 8), sliding_window=-1, window_right=8)
    assert bc_attn._module_window_size(m) == (-1, 8)


def test_no_window_arg_window_right_with_sliding_window(bc_attn):
    # upstream v1.6.7 attn.py: no window_arg; window_right applies when sliding_window >= 0
    m = SimpleNamespace(sliding_window=128, window_right=8)
    assert bc_attn._module_window_size(m) == (128, 8)


def test_no_window_arg_window_right_with_negative_sliding_window(bc_attn):
    # upstream: sliding_window < 0 (full attention) keeps the int even when window_right is set
    m = SimpleNamespace(sliding_window=-1, window_right=8)
    assert bc_attn._module_window_size(m) == -1


def test_no_window_arg_without_window_right(bc_attn):
    m = SimpleNamespace(sliding_window=128, window_right=0)
    assert bc_attn._module_window_size(m) == 128
    m = SimpleNamespace(sliding_window=-1)  # no window_right attribute at all
    assert bc_attn._module_window_size(m) == -1
