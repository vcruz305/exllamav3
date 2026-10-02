"""
Point-to-point tensor transport over torch.distributed (NCCL on GPUs, gloo on CPU), with the same
send_tensor / recv_tensor / send_obj / recv_obj interface as NetEndpoint.

On hosts linked by RDMA-capable NICs (e.g. DGX Spark ConnectX-7 over QSFP, RoCE), NCCL moves the
tensor GPU to GPU without host staging or Python socket loops. Measured on four GB10s on one QSFP
switch (MTU 1500): a 24 KB fp32 hidden state crosses one hop in ~21 us (4-hop ring 83 us), against
~1.1 ms per hop for NetEndpoint over TCP on the same link. Pin the link with NCCL_SOCKET_IFNAME (the
bootstrap interface) and NCCL_IB_HCA (the RoCE device, e.g. rocep1s0f0); NCCL_DEBUG=INFO should
report "via NET/IB". NCCL_IB_DISABLE=1 (socket transport) measured ~1.4 ms per small all-reduce.

Every rank of the job joins one process group (init_group); an endpoint is a (peer rank, device)
pair. Messages are framed by a fixed int64 header, so tensors of any shape and dtype in
_DTYPE_TO_CODE round-trip bit-exactly.
"""

from __future__ import annotations

import json
from typing import Any

import torch
import torch.distributed as dist


class NcclTransportError(Exception):
    pass


class NcclEndpoint:

    _DTYPE_TO_CODE = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
        torch.int32: 3,
        torch.int64: 4,
        torch.uint8: 5,
        torch.bool: 6,
    }
    if hasattr(torch, "float8_e4m3fn"):
        _DTYPE_TO_CODE[torch.float8_e4m3fn] = 7
    _CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}

    _MAGIC = 0x4E43434C  # "NCCL"
    _MSG_TENSOR = 0
    _MSG_OBJECT = 1
    # header: magic, msg_type, dtype_code, ndim, d0..d4, payload_len
    _HEADER_LEN = 10

    def __init__(self, peer: int, device: torch.device | str | int | None = None, group = None):
        """
        :param peer:
            Global rank of the other end.

        :param device:
            Device the communication buffers live on: the rank's CUDA device under NCCL, ignored
            (CPU) under gloo.
        """
        if not dist.is_initialized():
            raise NcclTransportError("torch.distributed is not initialized; call NcclEndpoint.init_group first")
        self.peer = peer
        self.group = group
        backend = dist.get_backend(group)
        self.is_nccl = backend == "nccl"
        if self.is_nccl:
            if device is None:
                device = torch.device("cuda", torch.cuda.current_device())
            self.comm_device = torch.device(device)
        else:
            self.comm_device = torch.device("cpu")
        self._hdr_send = torch.zeros(self._HEADER_LEN, dtype = torch.long, device = self.comm_device)
        self._hdr_recv = torch.zeros(self._HEADER_LEN, dtype = torch.long, device = self.comm_device)

    @staticmethod
    def init_group(
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        device: torch.device | str | int | None = None,
        backend: str | None = None,
        timeout_s: float = 3600,
    ):
        """
        Join (or create, on rank 0's host) the job-wide process group. Every rank calls this once
        before creating endpoints.
        """
        import datetime
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        kwargs = {}
        if backend == "nccl" and device is not None:
            kwargs["device_id"] = torch.device(device)
        dist.init_process_group(
            backend,
            rank = rank,
            world_size = world_size,
            init_method = f"tcp://{master_addr}:{master_port}",
            timeout = datetime.timedelta(seconds = timeout_s),
            **kwargs,
        )

    def _send_header(self, msg_type: int, dtype_code: int, shape: tuple, payload_len: int):
        if len(shape) > 5:
            raise NcclTransportError(f"Tensor has {len(shape)} dimensions; max 5 supported")
        dims = list(shape) + [0] * (5 - len(shape))
        h = [self._MAGIC, msg_type, dtype_code, len(shape), *dims, payload_len]
        self._hdr_send.copy_(torch.tensor(h, dtype = torch.long))
        dist.send(self._hdr_send, self.peer, group = self.group)

    def _recv_header(self, expect_type: int):
        dist.recv(self._hdr_recv, self.peer, group = self.group)
        h = self._hdr_recv.tolist()
        if h[0] != self._MAGIC:
            raise NcclTransportError(f"Bad magic: {hex(h[0])}")
        if h[1] != expect_type:
            raise NcclTransportError(f"Expected message type {expect_type}, got {h[1]}")
        return h

    def send_tensor(self, t: torch.Tensor, stream = None) -> None:
        """Send a tensor (any device; moved to the comm device if needed). stream is accepted for
        interface compatibility with NetEndpoint; NCCL orders its work on its own stream"""
        if t.dtype not in self._DTYPE_TO_CODE:
            raise NcclTransportError(f"Unsupported dtype: {t.dtype}")
        payload = t.contiguous()
        if payload.device != self.comm_device:
            payload = payload.to(self.comm_device)
        self._send_header(self._MSG_TENSOR, self._DTYPE_TO_CODE[t.dtype], tuple(t.shape), payload.numel() * payload.itemsize)
        if payload.numel() > 0:
            # NCCL has no fp8/bool reductions but p2p is a byte copy: send as uint8
            dist.send(payload.view(-1).view(torch.uint8), self.peer, group = self.group)

    def recv_tensor(self, out: torch.Tensor | None = None, device = None, stream = None) -> torch.Tensor:
        h = self._recv_header(self._MSG_TENSOR)
        dtype = self._CODE_TO_DTYPE.get(h[2])
        if dtype is None:
            raise NcclTransportError(f"Unknown dtype code: {h[2]}")
        shape = tuple(h[4:4 + h[3]])
        payload_len = h[9]
        target = torch.device(device) if device is not None else (out.device if out is not None else self.comm_device)
        if out is None or tuple(out.shape) != shape or out.dtype != dtype or out.device != self.comm_device:
            buf = torch.empty(shape, dtype = dtype, device = self.comm_device)
        else:
            buf = out
        if buf.numel() * buf.itemsize != payload_len:
            raise NcclTransportError(f"Payload length {payload_len} does not match shape {shape} {dtype}")
        if payload_len > 0:
            dist.recv(buf.view(-1).view(torch.uint8), self.peer, group = self.group)
        if buf.device != target:
            buf = buf.to(target)
        if out is not None and buf is not out and tuple(out.shape) == shape and out.dtype == dtype:
            out.copy_(buf)
            return out
        return buf

    def send_obj(self, obj: Any) -> None:
        try:
            payload = json.dumps(obj).encode("utf-8")
        except (TypeError, ValueError) as e:
            raise NcclTransportError(f"Object not JSON-serializable: {e}") from e
        self._send_header(self._MSG_OBJECT, 0, (), len(payload))
        if payload:
            t = torch.frombuffer(bytearray(payload), dtype = torch.uint8).to(self.comm_device)
            dist.send(t, self.peer, group = self.group)

    def recv_obj(self) -> Any:
        h = self._recv_header(self._MSG_OBJECT)
        n = h[9]
        if n == 0:
            return None
        t = torch.empty(n, dtype = torch.uint8, device = self.comm_device)
        dist.recv(t, self.peer, group = self.group)
        try:
            return json.loads(bytes(t.cpu().numpy()).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise NcclTransportError(f"Object parse failed: {e}") from e

    def close(self) -> None:
        """Endpoints hold no resources of their own; destroy the group with
        torch.distributed.destroy_process_group() once every endpoint is done"""
        pass
