import os

# ROCm: the profiler tool registration torch loads for torch.profiler leaves the HSA runtime's event
# thread spinning on a core for the life of the process, from the first device operation on. Off unless
# the variable is already set (ROCPROFILER_REGISTER_ENABLED=1 restores it, and with it the device events
# in torch.profiler traces). Has no effect on CUDA builds. Set before torch is imported so it is in place
# however early the host touches the device
os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")

try:
    import torch
except ImportError as e:
    raise RuntimeError(
        "PyTorch is required but not installed. exllamav3 deliberately does not install "
        "a default torch because it must match your CUDA setup; install a matching build "
        "first (for example `uv pip install torch --torch-backend=auto`, or select a CUDA "
        "flavor extra such as `uv sync --extra cu130`). "
        "See the README (\"Building from source\") for all install variants. "
        "https://github.com/turboderp-org/exllamav3"
    ) from e


# True once this package switched the CUDA caching allocator to expandable segments
expandable_segments = False

def _default_allocator_settings():
    """
    Expandable segments for the CUDA caching allocator, unless the user configured the
    allocator themselves.

    The runtime setter works before CUDA is initialized and also applies to every segment
    created after it, so hosts that already touched CUDA before importing the library still
    get it for the model.

    Skipped on Windows (virtual-memory API support is uneven there), on ROCm (the HIP allocator
    accepts the option, but where the driver cannot reserve virtual address ranges every later
    allocation fails as out of memory) and on torch builds that reject the option.
    """
    import os, sys
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ or sys.platform == "win32":
        return
    if torch.version.hip:
        return
    if os.environ.get("EXL3_EXPANDABLE_SEGMENTS", "1") == "0":
        return
    try:
        # torch >= 2.13 exposes the setter on the accelerator API and deprecates the cuda one
        setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None) \
            or torch.cuda.memory._set_allocator_settings
        setter("expandable_segments:True")
        global expandable_segments
        expandable_segments = True
    except Exception:
        pass

_default_allocator_settings()


def _raise_nofile_limit(target = 65536):
    """
    Raise the soft open-file limit to min(target, hard limit); never lowers it.

    Under WSL the CUDA driver keeps one event fd per kernel variant the extension has used. A model with
    many variants reaches the usual soft limit of 1024, and the next driver call then fails as
    "CUDA driver error: unknown error".
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except Exception:
        pass

_raise_nofile_limit()


def _default_triton_backend():
    """
    On a ROCm torch, point Triton at its AMD backend. With an NVIDIA driver also installed, both
    Triton backends report themselves active and Triton refuses to choose between them.
    """
    import os
    if torch.version.hip:
        os.environ.setdefault("TRITON_DEFAULT_BACKEND", "amd")

_default_triton_backend()

from .model.config import Config
from .model.model import Model
from .tokenizer import Tokenizer, MMEmbedding
from .cache import Cache, CacheLayer_fp16, CacheLayer_quant
from .generator import Generator, Job, AsyncGenerator, AsyncJob, Filter, FormatronFilter, LLGuidanceFilter
from .generator.sampler import *