"""Mixed-K decode switch defaults (block_sparse_mlp module flags, read at import time).

Each case imports the module in a fresh interpreter with a given environment, since the flags are
read once at import.
"""
import json
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROBE = (
    "import json, exllamav3.modules.block_sparse_mlp as m;"
    "print(json.dumps([m.COOP_MIXEDK, m.MIXEDK_NOSYNC, m.MIXEDK_NOSYNC_FUSED]))"
)


def _flags(**env):
    e = {k: v for k, v in os.environ.items()
         if k not in ("EXL3_MOE_COOP_MIXEDK", "EXL3_MOE_MIXEDK_NOSYNC")}
    e.update(env)
    e["PYTHONPATH"] = _ROOT + os.pathsep + e.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", _PROBE], env = e, cwd = _ROOT,
                         capture_output = True, text = True, timeout = 300)
    if out.returncode != 0:
        pytest.skip(f"exllamav3 not importable here: {out.stderr.strip().splitlines()[-1:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_default_coop_and_nosync_on_for_mixedk_only():
    # Unset: cooperative kernels on, sync-free dispatch on for mixed-K layers, uniform fused path untouched
    assert _flags() == [True, True, False]


def test_opt_out():
    assert _flags(EXL3_MOE_COOP_MIXEDK = "0", EXL3_MOE_MIXEDK_NOSYNC = "0") == [False, False, False]


def test_explicit_one_keeps_previous_meaning():
    # =1 was the old opt-in; NOSYNC=1 still extends the device-side count to the uniform fused path
    assert _flags(EXL3_MOE_COOP_MIXEDK = "1", EXL3_MOE_MIXEDK_NOSYNC = "1") == [True, True, True]
