"""CPU-only dispatch regression for the fork's decode int8 and upstream's tiled path."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch


def load_hc():
    """Load this one module without importing package startup or building the CUDA extension."""
    ext = types.SimpleNamespace()
    modules = {}
    for name in ("exllamav3", "exllamav3.modules"):
        package = types.ModuleType(name)
        package.__path__ = []
        modules[name] = package
    module = types.ModuleType("exllamav3.modules.module")
    class Module:
        def __init__(self, config=None, key=None, qmap=None):
            self.config, self.key = config, key
    module.Module = Module
    modules[module.__name__] = module
    module = types.ModuleType("exllamav3.modules.rmsnorm")
    module.RMSNorm = type("RMSNorm", (), {})
    modules[module.__name__] = module
    module = types.ModuleType("exllamav3.model.config")
    module.Config = type("Config", (), {})
    modules[module.__name__] = module
    module = types.ModuleType("exllamav3.ext")
    module.exllamav3_ext = ext
    modules[module.__name__] = module
    module = types.ModuleType("exllamav3.util.tensor")
    module.g_tensor_cache = types.SimpleNamespace()
    modules[module.__name__] = module
    name = "exllamav3.modules.hyperconnections"
    path = Path(__file__).resolve().parents[1] / "exllamav3/modules/hyperconnections.py"
    spec = importlib.util.spec_from_file_location(name, path)
    hc = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(hc)
    return hc


def test_decode_int8_dispatches_without_fp16_tables():
    hc = load_hc()
    calls = []
    hc.ext.gr_mix_int8 = lambda *args: calls.append("int8")
    hc.ext.gr_mix = lambda *args: calls.append("fp16")
    m = hc.GatedResidual(config=None, key="site", hc_mult=4, hidden_size=8, rms_norm_eps=1e-6)
    m.fn_h = m.upx_h = None
    m.fn_q = torch.zeros((3, 32), dtype=torch.int8)
    m.fn_s = torch.ones(3)
    m.upx_q = torch.zeros((4, 2, 1, 4), dtype=torch.int8)
    m.upx_s = torch.ones((4, 8))
    m.w_h = torch.ones(32, dtype=torch.half)
    m.tiled = False
    m._mix(torch.zeros((1, 1, 4, 8)), cached=False)
    assert calls == ["int8"]


def test_int8_quantization_preserves_tiled_source_lifecycle():
    hc = load_hc()
    m = hc.GatedResidual(config=None, key="site", hc_mult=4, hidden_size=8, rms_norm_eps=1e-6)
    m.fn_h = torch.ones((5, 32), dtype=torch.half)
    m.upx_h = torch.ones((4, 2, 1, 4), dtype=torch.half)
    m.tiled = True
    m._quantize_int8(4, 8)
    assert m.fn_q.shape == (5, 32) and m.upx_q.shape == (4, 2, 1, 4)
    assert m.fn_h is None and m.upx_h is None
