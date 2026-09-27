"""Build only MoE + reconstruct in isolated, content-addressed caches. Linux GB10 only.
No exllamav3 import/install; never writes into a checkout or site-packages.
--list-sources is CPU-only. Actual building requires an operator's stopped-model assertion.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

PIN = 'ca4a880e8918e1985fd25e06c6aff561666d3f14'
EXT = 'exllamav3/exllamav3_ext'
BINDING = '''#include <torch/extension.h>
#include "quant/exl3_moe.cuh"
#include "quant/reconstruct.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &exl3_moe_mixedk);
#ifdef EXL3_THREE_STAGE_BINDING
    m.def("run_three", &exl3_moe_mixedk_three_stage);
#endif
    m.def("gather", &exl3_moe_gather);
    m.def("reconstruct", &reconstruct);
}
'''


def sources(ext):
    q = ext / 'quant'
    return ([q / 'exl3_moe.cu', q / 'exl3_devctx.cu', q / 'reconstruct.cu'] +
            sorted((q / 'comp_units').glob('exl3_moe_inst*.cu')) +
            sorted((q / 'comp_units').glob('exl3_moe_mixedk_inst*.cu')))


def require_authorization():
    if os.environ.get('EXL3_TRIAL_MODEL_STOPPED') != 'YES' or os.environ.get('EXL3_THREE_STAGE_AUTHORIZED') != 'YES':
        raise SystemExit('Refusing build/GPU use: operator must stop model and set EXL3_TRIAL_MODEL_STOPPED=YES')
    if sys.platform != 'linux':
        raise SystemExit('Actual trial is Linux sm_121 only; local Windows work is source/CPU validation only')
    if not 1 <= int(os.environ.get('MAX_JOBS', '4')) <= 4:
        raise SystemExit('MAX_JOBS must be 1..4')
    os.environ['MAX_JOBS'] = os.environ.get('MAX_JOBS', '4')
    os.environ['TORCH_CUDA_ARCH_LIST'] = '12.1'
    if sys.getdlopenflags() & os.RTLD_GLOBAL:
        raise SystemExit('Refusing RTLD_GLOBAL: baseline/candidate symbols must stay isolated')


def snapshot_files(source, baseline):
    if not baseline:
        return {p.relative_to(source / EXT).as_posix(): p.read_bytes()
                for p in (source / EXT).rglob('*') if p.is_file() and
                p.suffix in ('.h', '.cuh', '.cu', '.cpp', '.c')}
    blob = subprocess.check_output(['git', '-C', str(source), '-c', 'core.autocrlf=false', 'archive', PIN, EXT])
    result = {}
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        for member in archive:
            if member.isfile():
                rel = Path(member.name).relative_to(EXT).as_posix()
                if '..' in Path(rel).parts:
                    raise RuntimeError('invalid archived path')
                result[rel] = archive.extractfile(member).read()
    return result


def load_trial(source, cache, baseline=False, poison=False):
    require_authorization()
    import torch
    from torch.utils.cpp_extension import load
    source, cache = Path(source).resolve(), Path(cache).resolve()
    if cache == source or source in cache.parents:
        raise SystemExit('Cache must be OUTSIDE the source checkout')
    files = snapshot_files(source, baseline)
    three = 'quant/exl3_moe_three_stage.cuh' in files
    flags = (['-DEXL3_THREE_STAGE_BINDING'] if three else []) + (['-DEXL3_THREE_STAGE_POISON'] if poison and three else [])
    sha = hashlib.sha256(BINDING.encode())
    sha.update((torch.__version__ + str(torch.version.cuda) + 'sm121-three-stage-v1' + repr(flags)).encode())
    for name, data in sorted(files.items()):
        sha.update(name.encode() + b'\0' + data)
    digest = sha.hexdigest()
    folder = cache / digest
    ext = folder / 'source'
    for rel, data in files.items():
        dst = ext / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() != data:
            raise RuntimeError('Content-addressed snapshot changed: ' + str(dst))
        if not dst.exists():
            dst.write_bytes(data)
    binding = folder / 'binding.cpp'
    if not binding.exists():
        binding.write_text(BINDING)
    elif binding.read_text() != BINDING:
        raise RuntimeError('Binding cache mismatch')
    build = folder / 'build'
    build.mkdir(exist_ok=True)
    name = 'exl3_three_trial_' + digest[:16]
    srcs = [binding] + sources(ext)
    manifest = dict(name=name, baseline=baseline, pin=PIN, digest=digest,
                    sources=[str(p) for p in srcs], arch='12.1', jobs=os.environ['MAX_JOBS'], flags=flags, poison=poison, three_stage=three)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    os.environ['TORCH_EXTENSIONS_DIR'] = str(cache / 'torch-private')
    module = load(name=name, sources=[str(p) for p in srcs],
                  extra_include_paths=[str(ext)], build_directory=str(build),
                  extra_cflags=['-O3', '-fvisibility=hidden'] + flags,
                  extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo',
                                    '-Xcompiler=-fvisibility=hidden'] + flags,
                  extra_ldflags=['-lcublas'], verbose=True)
    print(json.dumps({'loaded': module.__file__, **manifest}), flush=True)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path(__file__).parent / 'tree')
    p.add_argument('--cache', type=Path, default=Path(__file__).parent / 'build-cache')
    p.add_argument('--poison', action='store_true', help='debug-only actual intermediate fill; never time')
    p.add_argument('--baseline', action='store_true', help='snapshot the exact clean pin, not working files')
    p.add_argument('--list-sources', action='store_true', help='CPU-only; no imports, builds, copies or device access')
    a = p.parse_args()
    if a.list_sources:
        print(json.dumps([str(p) for p in sources(a.source / EXT)], indent=2))
        return
    load_trial(a.source, a.cache, a.baseline, a.poison)

if __name__ == '__main__':
    main()
