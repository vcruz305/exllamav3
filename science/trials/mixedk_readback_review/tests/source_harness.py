"""CPU/NumPy seams; execute complete source methods, never import torch/exllamav3.
The CUDA ext is an assignment/slot oracle, NOT GPU math or scheduler validation.
AST extraction strips decorators only; no body slices or formula replacements.
"""
from __future__ import annotations
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

ROOT = Path(os.environ['REVIEW_SOURCE_ROOT'])
READBACKS = []

class Tensor:
    def __init__(self, data, label='tensor'):
        self.a = np.asarray(data)
        self.label = label
    @property
    def shape(self): return self.a.shape
    @property
    def dtype(self): return self.a.dtype
    device = NS(type='cpu', index=None)
    is_cuda = False
    def __len__(self): return len(self.a)
    def numel(self): return self.a.size
    def __getitem__(self, key):
        if not isinstance(key, tuple): key = (key,)
        key = tuple(k.a if isinstance(k, Tensor) else k for k in key)
        return Tensor(self.a[key], self.label)
    def __setitem__(self, key, value): self.a[key] = value.a if isinstance(value, Tensor) else value
    def reshape(self, *shape): return Tensor(self.a.reshape(*shape), self.label)
    view = reshape
    def clone(self): return Tensor(self.a.copy(), self.label)
    def copy_(self, value, **kw):
        self.a[...] = value.a if isinstance(value, Tensor) else value
        return self
    def zero_(self): return self.copy_(0)
    def fill_(self, value): return self.copy_(value)
    def to(self, *args, **kw): return self
    def cpu(self): return self
    def half(self): return Tensor(self.a.astype(np.float16), self.label)
    def long(self): return Tensor(self.a.astype(np.int64), self.label)
    def item(self): return self.a.item()
    def tolist(self):
        READBACKS.append(self.label)
        return self.a.tolist()
    def argsort(self): return Tensor(self.a.argsort(kind='stable'))
    def nonzero(self, as_tuple=False):
        result = tuple(Tensor(a, 'nonzero') for a in np.nonzero(self.a))
        return result if as_tuple else Tensor(np.stack([t.a for t in result], axis=1), 'nonzero')
    def scatter_(self, dim, index, src):
        assert dim == 0
        self.a[index.a] = src.a
        return self
    def unsqueeze(self, dim): return Tensor(np.expand_dims(self.a, dim))
    def index_select(self, dim, index): return Tensor(np.take(self.a, index.a, axis=dim))
    def mul_(self, value):
        self.a *= value.a if isinstance(value, Tensor) else value
        return self
    def index_add_(self, dim, index, value):
        assert dim == 0
        np.add.at(self.a, index.a, value.a)
        return self
    def __add__(self, x): return Tensor(self.a + (x.a if isinstance(x, Tensor) else x), self.label)
    def __sub__(self, x): return Tensor(self.a - (x.a if isinstance(x, Tensor) else x), self.label)
    def __gt__(self, x): return Tensor(self.a > x, self.label)
    def __ge__(self, x): return Tensor(self.a >= x, self.label)
    def __lt__(self, x): return Tensor(self.a < x, self.label)
    def __le__(self, x): return Tensor(self.a <= x, self.label)
    def __and__(self, x): return Tensor(self.a & x.a, self.label)
    def __eq__(self, x): return Tensor(self.a == (x.a if isinstance(x, Tensor) else x), self.label)

class Torch:
    Tensor = Tensor
    float = np.float32
    half = np.float16
    long = np.int64
    int32 = np.int32
    cuda = NS(synchronize=lambda *a: None)
    @staticmethod
    def empty(shape, dtype=np.float32, **kw):
        # Poison unwritten slots: the gather oracle must never consume them.
        return Tensor(np.full(shape, np.nan if np.issubdtype(dtype, np.floating) else -987654, dtype=dtype))
    @staticmethod
    def zeros(shape, dtype=np.float32, **kw): return Tensor(np.zeros(shape, dtype=dtype))
    @staticmethod
    def full(shape, value, dtype=np.int64, **kw): return Tensor(np.full(shape, value, dtype=dtype))
    @staticmethod
    def empty_like(t, **kw): return Torch.empty(t.shape, dtype=kw.get('dtype', t.dtype))
    @staticmethod
    def full_like(t, value): return Torch.full(t.shape, value, dtype=t.dtype)
    @staticmethod
    def tensor(data, dtype=None, **kw): return Tensor(np.array(data, dtype=dtype))
    @staticmethod
    def from_numpy(data): return Tensor(data)
    @staticmethod
    def arange(n, **kw): return Tensor(np.arange(n))
    @staticmethod
    def bincount(t, minlength): return Tensor(np.bincount(t.a, minlength=minlength), 'counts')
    @staticmethod
    def cumsum(t, dim): return Tensor(t.a.cumsum(axis=dim))
    @staticmethod
    def stack(ts): return Tensor(np.stack([t.a for t in ts]))
    @staticmethod
    def where(c, a, b): return Tensor(np.where(c.a, a.a, b.a))
    @staticmethod
    def cat(ts, dim=0, out=None):
        value = Tensor(np.concatenate([t.a for t in ts], axis=dim))
        return out.copy_(value) if out is not None else value
    @staticmethod
    def device(x): return NS(index=0)


def extract_class(rel, cls, methods, namespace):
    path = ROOT / rel
    tree = ast.parse(path.read_text(encoding='utf-8'))
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    nodes = []
    for name in methods:
        node = copy.deepcopy(next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name))
        node.decorator_list = []
        nodes.append(node)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             ast.ClassDef(name=cls, bases=[], keywords=[], body=nodes, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[cls]


def module_constants(rel, namespace):
    """Evaluate actual literal/env-derived constants without importing GPU dependencies."""
    tree = ast.parse((ROOT / rel).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(n.id.isupper() for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / rel), 'exec'), namespace)
            except NameError:
                pass


class MoEOracle:
    def __init__(self):
        self.rows = []
        self.launches = []
        self.legacy_experts = None
    def exl3_moe_max_concurrency(self, index): return 6
    def exl3_moe_mixedk(self, *a): self.launch(a, False)
    def exl3_moe(self, *a): self.launch(a, True)
    def launch(self, a, legacy):
        y, out, counts, tokens, weights = a[:5]
        cap = a[5].shape[1]
        active, scratch, base, lo, hi, tile = a[-6:]
        es = [e for e, c in enumerate(counts.a[:-1]) if 0 < c <= cap and lo <= c <= hi]
        self.launches.append((legacy, active, cap, lo, hi, tile, tuple(es)))
        if active >= 0: assert active == len(es), (active, es)
        start = 0
        for e, count in enumerate(counts.a[:-1]):
            if e in es:
                original_e = self.legacy_experts[e] if legacy and self.legacy_experts is not None else e
                for r in range(int(count)):
                    pos = start + r
                    token = int(tokens.a[pos])
                    self.rows.append((original_e, token))
                    value = (original_e + 1) * y.a[token] * weights.a[pos]
                    if scratch is not None:
                        slot = int(base.a[e]) + r
                        assert 0 <= slot < scratch.shape[0]
                        assert np.isnan(scratch.a[slot]).all(), 'duplicate scratch write'
                        scratch.a[slot] = value
                    else: out.a[token] += value
            start += int(count)
    def exl3_moe_gather(self, out, scratch, flat, inv, starts, bases, kinds, weights):
        topk = flat.numel() // out.shape[0]
        for a, e in enumerate(flat.a):
            if e < 0 or e >= kinds.numel(): continue
            if not kinds.a[e]: continue
            pos = int(inv.a[a]); slot = int(bases.a[e] + pos - starts.a[e])
            assert 0 <= slot < scratch.shape[0]
            assert np.isfinite(scratch.a[slot]).all(), 'unwritten/sentinel slot read'
            weight = weights.a[pos] if kinds.a[e] == 2 else 1
            out.a[a // topk] += scratch.a[slot] * weight


def moe_fixture(selected, *, enabled=False, local=4, total=None, first=0, unified=True,
                mtile=False, det=True, rowcap=None, initialized=True, legacy=False, concurrency=6):
    from unittest.mock import patch
    ext = MoEOracle()
    ns = dict(torch=Torch, os=os, ext=ext, FusedBuffers=lambda **kw: NS(**kw),
              g_tensor_cache=NS(get=lambda dev, shape, dtype, key: Torch.empty(shape, dtype=dtype)),
              buffered_interleaved_arange=lambda n,k,device: Tensor(np.repeat(np.arange(n), k)))
    rel = 'exllamav3/modules/block_sparse_mlp.py'
    with patch.dict(os.environ, {'EXL3_MOE_MIXEDK_ELIDE_HANDLED': '1' if enabled else '0'}, clear=False):
        module_constants(rel, ns)
    ns.update(FUSED_DET=det, MTILE=mtile, MIXEDK_MIN_ROWS=0)
    Cls = extract_class(rel, 'BlockSparseMLP', ['forward'], ns)
    m = Cls()
    picks = Torch.tensor(selected, dtype=np.int64)
    n, k = picks.shape
    width = 2
    m.__dict__.update(alt_residual_channel=False, hidden_size=width, expert_size=width, bc=None,
        mixedk_unified=unified, num_experts_per_tok=k, router_pre_norm=None, routing_gate=True,
        routed_pre_norm=None, latent_in=None, routing_device=None, cpu_split_first=None,
        cpu_offload=False, intermediate_size=2, intermediate_size_padded=2, num_local_experts=local,
        num_experts=total or local, routing_first=first, f_threshold=1, is_quantized=True,
        config=NS(infer_params=NS(no_reconstruct=False)), support_quant_paths=False,
        fused_mode_buffers=None, fused_rows=ns['TEMP_ROWS_FUSED'], device='cpu', gated=False,
        mixedk_mul1=True, mixedk_mcg=False, act_limit=0, activation_fn_idx=0,
        latent_out=None, tp_reduce=False, routed_post_norm=None, shared_experts=None,
        interm_dtype=np.float16, routing_cfg=None)
    for name in ['mixedk_K_gate_arr','mixedk_K_up_arr','mixedk_K_down_arr'] + [f'mixedk_ptrs_{p}_{s}' for p in ('gate','up','down') for s in ('trellis','suh','svh')]:
        setattr(m, name, None)
    weights = Torch.full((n,k), 1, dtype=np.float16)
    m.routing_fn = lambda *a: (picks, weights)
    m.cpu_split_combine = lambda final,*a: final
    m._batch_recon_layer = lambda y: None
    m.gateless_act = lambda x: x
    m.ups = [NS(forward=lambda x,p: x) for _ in range(local)]
    def down(e):
        def forward(x, params):
            for v in x.a[:,0]: ext.rows.append((e, int(v)-1))
            return Tensor(x.a * (e+1))
        return NS(forward=forward)
    m.downs = [down(e) for e in range(local)]
    if initialized:
        cap = rowcap or ns['TEMP_ROWS_FUSED']
        m._mkd_bufs = NS(**{key:Torch.empty((concurrency,cap,width),dtype=np.float16) for key in
                            ('temp_state_g','temp_state_u','temp_intermediate_g','temp_intermediate_u')})
        m._mkd_fused_rows = cap
        m._mkd_mtile_ok = mtile
    if legacy:
        group = list(range(local))
        mg = NS(K=1,ptrs_trellis=None,ptrs_suh=None,ptrs_svh=None,mcg=False,mul1=True)
        m.mixedk_k_groups = [(group,mg,mg,mg,None)]
        ext.legacy_experts = group
    x = Tensor(np.repeat(np.arange(1,n+1)[:,None],width,axis=1).astype(np.float16))
    READBACKS.clear()
    return m, x, ext
