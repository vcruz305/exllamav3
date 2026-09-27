"""Source-executed device-contract probes for the SEALED repaired diagnostic.

This module executes the exact AST-extracted function bodies taken from the
sealed diagnostic sources (`width/candidate/width_diag.py` = repaired,
`width/baseline/width_diag.py` = pre-repair) against an explicit mock
tensor/device contract.

THIS IS NOT CUDA. The mock enforces the *call contract* the real producer must
satisfy: comparison-operand device+dtype alignment, output object preservation,
metadata emitted before assertions, and the active-state lifetime across a
preparatory failure. It does not implement or prove CUDA transfer, cast, stream
or allocating-context semantics. Real device behaviour stays a device gate
(see GAPS.md, gates G-DEVICE-PREFLIGHT / G-WIDTH-NATIVE16).

Every helper here is deliberately pure: no import-time I/O, no torch, no
network, no exllamav3, no model and no writes.
"""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace as S

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- mock tensor


def _prod(shape):
    total = 1
    for value in shape:
        total *= value
    return total


class MockTensor:
    """Minimal row-major tensor stand-in implementing ONLY the audited subset.

    Supported: `.shape`, `.device`, `.dtype`, `.to(dtype=)`/`.to(device=,
    dtype=)`, `.reshape(*shape)`, `.expand(*shape)`, `.clone()`, `.data_ptr()`,
    `.remainder(n)`, `.all()`, `.item()`, `.tolist()`, `__getitem__` with a
    3-tuple of (all, slice-or-int, all), and `[:, a:b, :]` assignment.

    Anything outside that subset raises AssertionError instead of silently
    returning a wrong-shaped or wrong-device result.
    """

    def __init__(self, shape, device, dtype, values=None):
        self.shape = tuple(int(x) for x in shape)
        self.device = device
        self.dtype = dtype
        self.values = list(values) if values is not None else [0.0] * _prod(self.shape)
        assert len(self.values) == _prod(self.shape), 'mock value/shape mismatch'
        self.ptr = id(self)

    def _reindex(self, shape):
        assert _prod(shape) == _prod(self.shape), 'reshape/expand changed element count'
        return MockTensor(shape, self.device, self.dtype, self.values)

    def _broadcast(self, shape):
        old = self.shape
        assert len(old) == len(shape), 'mock broadcast rank mismatch'
        for axis, (was, now) in enumerate(zip(old, shape)):
            assert was == now or was == 1, 'mock cannot broadcast non-singleton axis ' + str(axis)
        values = []
        for i in range(int(shape[0])):
            for j in range(int(shape[1])):
                for k in range(int(shape[2])):
                    oi = 0 if old[0] == 1 else i
                    oj = 0 if old[1] == 1 else j
                    ok = 0 if old[2] == 1 else k
                    values.append(self.values[(oi * old[1] + oj) * old[2] + ok])
        return MockTensor(shape, self.device, self.dtype, values)

    def to(self, dtype=None, device=None, **kw):
        assert not kw, 'unsupported .to() keyword'
        # torch semantics: a dtype-only cast keeps the device; a device-only
        # move keeps the dtype. Both may be given at once.
        return MockTensor(self.shape, self.device if device is None else device,
                          self.dtype if dtype is None else dtype, self.values)

    def reshape(self, *shape):
        flat = list(shape if len(shape) > 1 else shape[0])
        unknown = [i for i, d in enumerate(flat) if d == -1]
        assert len(unknown) <= 1, 'mock reshape accepts at most one inferred dimension'
        if unknown:
            known = 1
            for i, d in enumerate(flat):
                assert d != 0, 'mock reshape of a zero dimension is unsupported'
                if i not in unknown:
                    known *= d
            assert known and _prod(self.shape) % known == 0, 'mock reshape cannot infer dimension'
            flat[unknown[0]] = _prod(self.shape) // known
        return self._reindex(tuple(flat))

    def expand(self, *shape):
        flat = list(shape if len(shape) > 1 else shape[0])
        assert len(flat) == len(self.shape), 'mock expand rank mismatch'
        flat = [was if now == -1 else now for was, now in zip(self.shape, flat)]
        return self._broadcast(tuple(flat))

    def clone(self):
        return self._reindex(self.shape)

    def data_ptr(self):
        return self.ptr

    def remainder(self, n):
        return MockTensor(self.shape, self.device, self.dtype, [v % n for v in self.values])

    def all(self):
        return MockTensor((1,), self.device, self.dtype, [1.0 if all(self.values) else 0.0])

    def item(self):
        assert _prod(self.shape) == 1, 'item() on non-scalar mock tensor'
        return self.values[0]

    def tolist(self):
        return list(self.values)

    def _slice(self, key):
        assert isinstance(key, tuple) and len(key) == 3, 'unsupported mock index'
        dims = self.shape
        spans = []
        for axis, part in enumerate(key):
            if isinstance(part, slice):
                assert part.step is None, 'unsupported mock slice step'
                start = 0 if part.start is None else part.start
                stop = dims[axis] if part.stop is None else part.stop
                assert 0 <= start <= stop <= dims[axis], 'mock slice out of range'
                spans.append(range(start, stop))
            elif isinstance(part, int):
                assert 0 <= part < dims[axis], 'mock index out of range'
                spans.append(range(part, part + 1))
            else:
                raise AssertionError('unsupported mock index component')
        shape = tuple(len(s) for s in spans)
        values = []
        for i in spans[0]:
            for j in spans[1]:
                for k in spans[2]:
                    values.append(self.values[(i * dims[1] + j) * dims[2] + k])
        return MockTensor(shape, self.device, self.dtype, values)

    def __getitem__(self, key):
        return self._slice(key)

    def __setitem__(self, key, value):
        assert isinstance(key, tuple) and len(key) == 3, 'unsupported mock assignment target'
        if all(isinstance(index, int) for index in key):
            dims = self.shape
            assert all(0 <= index < dim for index, dim in zip(key, dims)), 'mock index out of range'
            self.values[(key[0] * dims[1] + key[1]) * dims[2] + key[2]] = (
                value.values[0] if isinstance(value, MockTensor) else value)
            return
        assert key[0] == slice(None) and key[2] == slice(None) and isinstance(key[1], slice), \
            'mock supports only full-slice [:, a:b, :] assignment or a single element'
        source = list(value.values) if isinstance(value, MockTensor) else list(value)
        dims = self.shape
        start = 0 if key[1].start is None else key[1].start
        stop = dims[1] if key[1].stop is None else key[1].stop
        target = dims[0] * (stop - start) * dims[2]
        if len(source) != target:
            # torch broadcasts a right-aligned trailing-dimension source, e.g.
            # a (4096,) learned mask assigned into a (1,15,4096) slice.
            assert len(source) == dims[2] and target % dims[2] == 0, 'mock assignment size mismatch'
            source = source * (target // dims[2])
        cursor = 0
        for i in range(dims[0]):
            for j in range(start, stop):
                for k in range(dims[2]):
                    self.values[(i * dims[1] + j) * dims[2] + k] = source[cursor]
                    cursor += 1


    def copy_(self, other):
        """In-place value copy, used by the device preflight's restore step."""
        assert isinstance(other, MockTensor) and other.shape == self.shape, 'mock copy_ mismatch'
        self.values = list(other.values)
        return self


class MockTorch:
    """Device/type contract stand-in. `equal` enforces the real device rule."""

    float16 = 'float16'
    bfloat16 = 'bfloat16'
    float32 = 'float32'
    half = 'float16'
    long = 'int64'
    bool = 'bool'

    class cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def synchronize():
            raise AssertionError('mock torch must never be synchronized')

    def zeros(self, shape, device='cpu', dtype='float32'):
        return MockTensor(shape, device, dtype)

    def arange(self, n, device='cpu', dtype='float32'):
        return MockTensor((n,), device, dtype, [float(i) for i in range(n)])

    def tensor(self, values, device='cpu', dtype='float32'):
        data = list(values)
        return MockTensor((len(data),), device, dtype, [float(v) for v in data])

    def isfinite(self, tensor):
        assert isinstance(tensor, MockTensor), 'isfinite expects a mock tensor'
        values = []
        for value in tensor.values:
            finite = value == value and value not in (float('inf'), float('-inf'))
            values.append(1.0 if finite else 0.0)
        return MockTensor(tensor.shape, tensor.device, self.bool, values)

    def equal(self, left, right):
        """Mirror torch.equal: a device mismatch raises, value/shape difference is False."""
        assert isinstance(left, MockTensor) and isinstance(right, MockTensor)
        if left.device != right.device:
            raise RuntimeError(
                'Expected all tensors to be on the same device, but got other is on '
                + str(right.device) + ', different from other tensors on ' + str(left.device)
                + ' (when checking argument in method wrapper_CUDA__equal)')
        if tuple(left.shape) != tuple(right.shape):
            return False
        return all(a == b for a, b in zip(left.values, right.values))


# -------------------------------------------------------------- AST extraction

DIAG_FUNCTIONS = ('finite', 'inp')


def source_text(path):
    return Path(path).read_bytes().decode()


def extract_functions(text, names, filename):
    """Compile ONLY the named FunctionDefs, in the caller's requested order."""
    tree = ast.parse(text, filename=filename)
    found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(found) == len(names), \
        'expected ' + str(len(names)) + ' diagnostic functions, found ' + str(len(found))
    order = {name: index for index, name in enumerate(names)}
    found.sort(key=lambda node: order[node.name])
    return ast.Module(body=found, type_ignores=[]), [node.name for node in found]


def load_diagnostic(path, names=DIAG_FUNCTIONS):
    module, loaded = extract_functions(source_text(path), names, str(path))
    return module, loaded


# --------------------------------------------------- input-hook device contract

# The pre-repair comparison operand, as repaired minus the explicit device
# alignment. `width/baseline/width_diag.py` writes the same expression against
# `dm.input_layer.mask_embedding`; the test asserts both sources carry the
# identical `.to(y.dtype).reshape(1,1,-1).expand(1,15,-1)` shape so this
# mutation is a faithful replay of the historical defect, not an invention.
OPERAND_SUFFIX = ".reshape(1,1,-1).expand(1,15,-1)"
HISTORICAL_OPERAND = "mask.to(y.dtype)" + OPERAND_SUFFIX
REPAIRED_OPERAND = "mask.to(device=y.device,dtype=y.dtype).reshape(1,1,-1).expand(1,15,-1)"
HISTORICAL_OPERAND_MUTATION = (REPAIRED_OPERAND, HISTORICAL_OPERAND)


def run_input_contract(diag_path, *, output_device='cpu', mask_device='cuda:0',
                       output_dtype='float32', mask_dtype='bfloat16', mutate=None,
                       mask_size=4096, block=16, width=4096, mask_value=None,
                       output_corruption=None, on_failure='raise'):
    """Execute the sealed `finite`/`inp` pair against the mock device contract.

    `mutate` is an optional (old, new) string pair applied to an IN-MEMORY copy
    of the source only; the sealed file is never written. `mask_value` forces a
    single learned-mask element. `output_corruption` is an (i, j, k, value)
    tuple applied to the producer OUTPUT only, immediately before the snapshot —
    this is the only way to create a genuine value mismatch, because a mask edit
    propagates to both comparison operands.

    Returns an observation dict. Raises whatever the executed source raises.
    """
    text = source_text(diag_path)
    mutated = False
    if mutate is not None:
        old, new = mutate
        assert old in text, 'mutation target absent from sealed diagnostic'
        text = text.replace(old, new)
        mutated = True
    module, loaded = extract_functions(text, DIAG_FUNCTIONS, str(diag_path))
    torch = MockTorch()
    rows = []
    mask = torch.arange(mask_size, device=mask_device, dtype='float32').remainder(7).to(dtype=mask_dtype)
    if mask_value is not None:
        mask.values[0] = mask_value
    output = torch.zeros((1, block, width), device=output_device, dtype=output_dtype)
    # The production slice-assignment contract is exercised, never changed to
    # suit capture: the producer writes the learned mask into rows 1..block-1.
    output[:, 1:, :] = mask.to(dtype=output_dtype)
    if output_corruption is not None:
        i, j, k, value = output_corruption
        output.values[(i * block + j) * width + k] = value
    saved = output.clone()
    pointer = output.data_ptr()
    env = {'torch': torch, 'state': {'active': True},
           'dm': S(input_layer=S(mask_embedding=mask)),
           'original_input': lambda *args, **kw: output,
           'emit': lambda kind, **kw: rows.append(dict(kind=kind, **kw))}
    exec(compile(module, str(diag_path), 'exec'), env)
    error = None
    result = None
    try:
        result = env['inp'](torch.zeros((1, 1), dtype=torch.long), {})
    except BaseException as exc:  # noqa: BLE001 - `on_failure` decides whether it is fatal
        if on_failure != 'capture':
            raise
        error = exc
    return {
        'row': next((r for r in rows if r['kind'] == 'input'), None),
        'metadata': next((r for r in rows if r['kind'] == 'input_metadata'), None),
        'error_type': type(error).__name__ if error is not None else None,
        'error_message': str(error) if error is not None else None,
        'emits': [r['kind'] for r in rows],
        'result_is_output': result is output,
        'output_unchanged': all(a == b for a, b in zip(output.values, saved.values)),
        'output_pointer_stable': output.data_ptr() == pointer,
        'output_device': str(output_device),
        'output_dtype': str(output_dtype),
        'mask_device': str(mask_device),
        'source_mutated': mutated,
        'functions': loaded,
    }


def compare_operands(diag_path, *, mutate=None, output_device='cpu', mask_device='cuda:0',
                     output_dtype='float32', mask_dtype='bfloat16'):
    """Return the operand devices/dtypes the sealed `inp` hands to `torch.equal`.

    The real `inp` body is executed; the operands are captured by wrapping
    `torch.equal`. Returns None when `torch.equal` was never reached.
    """
    text = source_text(diag_path)
    if mutate is not None:
        old, new = mutate
        assert old in text, 'mutation target absent from sealed diagnostic'
        text = text.replace(old, new)
    module, _ = extract_functions(text, DIAG_FUNCTIONS, str(diag_path))
    torch = MockTorch()
    captured = []
    original_equal = torch.equal

    def recording_equal(left, right):
        captured.append({'left_device': str(left.device), 'left_dtype': str(left.dtype),
                         'left_shape': list(left.shape), 'right_device': str(right.device),
                         'right_dtype': str(right.dtype), 'right_shape': list(right.shape)})
        return original_equal(left, right)

    torch.equal = recording_equal
    mask = torch.arange(4096, device=mask_device, dtype='float32').remainder(7).to(dtype=mask_dtype)
    output = torch.zeros((1, 16, 4096), device=output_device, dtype=output_dtype)
    output[:, 1:, :] = mask.to(dtype=output_dtype)
    env = {'torch': torch, 'state': {'active': True}, 'dm': S(input_layer=S(mask_embedding=mask)),
           'original_input': lambda *args, **kw: output, 'emit': lambda kind, **kw: None}
    exec(compile(module, str(diag_path), 'exec'), env)
    env['inp'](torch.zeros((1, 1), dtype=torch.long), {})
    return captured[-1] if captured else None


# ------------------------------------------------- forward active-state lifetime


def _assigned_positions():
    """The real sealed contract helper, loaded by path (never by module name)."""
    helper = HERE / 'width' / 'candidate' / 'width_contract.py'
    namespace = {}
    exec(compile(helper.read_bytes(), str(helper), 'exec'), namespace)
    return namespace['assigned_positions']


def run_forward_lifetime(diag_path):
    """Execute the sealed `forward` body with a preparatory rejection.

    Returns {'active_after': bool, 'emits': [...], 'raised': str|None}.

    The pre-repair source marks the capture active BEFORE preparing the page
    mapping, so a preparation failure leaks `state['active'] = True` into the
    next round. The repaired source spans that preparation with its existing
    try/finally.
    """
    module, loaded = extract_functions(source_text(diag_path), ('forward',), str(diag_path))  # forward-only
    torch = MockTorch()
    rows = []
    state = {'active': False, 'round': 1}
    game = S(active_jobs=[S(sequences=[S(allocated_pages=[])])], draft_cache=S(layers={}))
    # `params` mirrors the real generator call: cache_seqlens[0] is the start
    # offset, block_table[0] is a tensor-like with .tolist(), and the sequence
    # owns NO pages, so `assigned_positions` rejects the request during
    # preparation (the audited pre-forward rejection).
    params = {'cache_seqlens': [0], 'block_table': [S(tolist=lambda: [])]}
    previous = threading.current_thread().name
    threading.current_thread().name = 'native-generator'
    raised = None
    try:
        env = {'threading': threading, 'torch': torch, 'state': state, 'g': game,
               'params': params, 'assigned_positions': _assigned_positions(),
               'emit': lambda kind, **kw: rows.append(dict(kind=kind, **kw)),
               'original_forward': lambda *args, **kw: (_ for _ in ()).throw(
                   AssertionError('original forward must not be reached'))}
        exec(compile(module, str(diag_path), 'exec'), env)
        try:
            env['forward'](params=params)
        except BaseException as exc:  # noqa: BLE001 - the rejection is the point
            raised = type(exc).__name__ + ': ' + str(exc)
    finally:
        threading.current_thread().name = previous
    return {'active_after': bool(state['active']), 'emits': [r['kind'] for r in rows],
            'raised': raised, 'functions': loaded}
