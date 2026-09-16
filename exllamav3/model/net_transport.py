"""
Standalone tensor transport module for cross-host pipeline mode in exllamav3.

Moves torch tensors and small Python control messages between two processes on
DIFFERENT machines over TCP, as the payload path for a pipeline split where one
node runs layers 0..k and the other k+1..N.

The transport layer handles:
- Fixed-size framing with magic, msg type, dtype code, shape, payload length
- Reliable recv loops that handle partial reads and detect EOF
- CUDA tensor staging via pinned host memory
- All required dtypes (float16, bfloat16, float32, int32, int64, uint8, bool)
- Non-contiguous tensor handling (make contiguous before send)
- Zero-copy where possible (recv_into, memoryview)
- Clear error handling with NetTransportError
"""

from __future__ import annotations

import socket
import struct
import json
import io
from typing import Any

import torch


class NetTransportError(Exception):
    """Raised when framing, EOF, or timeout issues occur during transport."""
    pass


class NetEndpoint:
    """
    Manages TCP socket communication for tensor and object transport.

    Supports bidirectional send/recv of torch tensors and JSON-serializable
    control messages. CUDA tensors are staged through pinned host memory.
    Framing is deterministic; partial reads and EOF are treated as errors.
    """

    # Dtype -> code mapping (1 byte)
    _DTYPE_TO_CODE = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
        torch.int32: 3,
        torch.int64: 4,
        torch.uint8: 5,
        torch.bool: 6,
    }

    _CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}

    # Message types
    _MSG_TENSOR = 0
    _MSG_OBJECT = 1

    # Fixed header layout:
    # magic (4 bytes): 0xDEADBEEF
    # msg_type (1 byte): 0=tensor, 1=object
    # dtype_code (1 byte): 0-6 for tensor types
    # ndim (1 byte): number of dimensions
    # shape (40 bytes): 5 uint64 slots for dims (up to 5D)
    # payload_len (8 bytes): byte count of payload
    # Total: 4 + 1 + 1 + 1 + 40 + 8 = 55 bytes
    _HEADER_FMT = "!I B B B 5Q Q"
    _HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _MAGIC = 0xDEADBEEF

    def __init__(self, sock: socket.socket, is_server: bool = False):
        """
        Initialize endpoint with an existing socket.

        Args:
            sock: Connected socket.socket instance.
            is_server: Whether this endpoint is the listening side (for logging/clarity).
        """
        self.sock = sock
        self.is_server = is_server
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Pinned staging buffer for CUDA -> host transfers (reused across calls)
        self._pinned_buffer = None
        self._pinned_size = 0

    @classmethod
    def listen(
        cls,
        host: str,
        port: int,
        timeout: float = 60.0,
    ) -> NetEndpoint:
        """
        Create a server endpoint that listens for and accepts one peer connection.

        Args:
            host: Interface to bind to (e.g., "0.0.0.0").
            port: Port to listen on.
            timeout: Accept timeout in seconds.

        Returns:
            NetEndpoint connected to the peer.

        Raises:
            NetTransportError: On socket/accept errors or timeout.
        """
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((host, port))
            server_sock.listen(1)
            server_sock.settimeout(timeout)

            try:
                peer_sock, peer_addr = server_sock.accept()
            finally:
                server_sock.close()

            return cls(peer_sock, is_server = True)
        except Exception as e:
            raise NetTransportError(f"listen on {host}:{port} failed: {e}") from e

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        timeout: float = 60.0,
    ) -> NetEndpoint:
        """
        Create a client endpoint that connects to a listening peer.

        Retries with exponential backoff until timeout expires.

        Args:
            host: Remote host to connect to.
            port: Remote port to connect to.
            timeout: Total time to retry before giving up (seconds).

        Returns:
            NetEndpoint connected to the peer.

        Raises:
            NetTransportError: If connection fails after timeout or on other errors.
        """
        import time

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        start_time = time.time()
        attempt = 0
        last_error = None

        while time.time() - start_time < timeout:
            try:
                sock.connect((host, port))
                return cls(sock, is_server = False)
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                last_error = e
                attempt += 1
                backoff = min(0.1 * (2 ** attempt), 1.0)
                remaining = timeout - (time.time() - start_time)
                if remaining > 0:
                    time.sleep(min(backoff, remaining))

        sock.close()
        raise NetTransportError(
            f"connect to {host}:{port} failed after {timeout}s: {last_error}"
        )

    def send_tensor(self, t: torch.Tensor) -> None:
        """
        Send a tensor to the peer.

        Non-contiguous tensors are made contiguous before sending.
        CUDA tensors are staged through pinned host memory.

        Args:
            t: torch.Tensor to send (any shape, any supported dtype/device).

        Raises:
            NetTransportError: If dtype is unsupported or send fails.
        """
        # Ensure contiguous and on correct device/dtype for send
        if not t.is_contiguous():
            t = t.contiguous()

        dtype = t.dtype
        if dtype not in self._DTYPE_TO_CODE:
            raise NetTransportError(f"Unsupported dtype: {dtype}")

        dtype_code = self._DTYPE_TO_CODE[dtype]
        ndim = len(t.shape)
        if ndim > 5:
            raise NetTransportError(f"Tensor has {ndim} dimensions; max 5 supported")

        shape_tuple = tuple(t.shape) + (0,) * (5 - ndim)
        payload_len = t.numel() * t.itemsize

        # Stage a CUDA tensor through the reused pinned buffer. The staged tensor stays on the host
        # and is sent straight from its storage; copying it again with .cpu() would undo the point
        # of pinning
        if t.is_cuda:
            staging = self._get_pinned_buffer(payload_len)
            send_data = staging[:payload_len].view(t.dtype).view(t.shape)
            send_data.copy_(t, non_blocking = False)
            send_data = send_data.view(-1)
        else:
            send_data = t.view(-1)

        # Build and send header
        header = struct.pack(
            self._HEADER_FMT,
            self._MAGIC,
            self._MSG_TENSOR,
            dtype_code,
            ndim,
            *shape_tuple,
            payload_len,
        )
        self._sendall(header)

        # TODO: this copies the payload once more than necessary. Sending from a memoryview of the
        # storage needs a dtype-safe way to get those bytes (torch has no public buffer protocol for
        # bfloat16/bool), so measure before optimizing: at 10 KB per decode hidden state the copy is
        # noise, and it only matters for prefill-sized chunks
        if send_data.numel() > 0:
            self._sendall(bytes(send_data.untyped_storage()))

    def recv_tensor(self, out: torch.Tensor | None = None) -> torch.Tensor:
        """
        Receive a tensor from the peer.

        If `out` is provided and matches the incoming shape/dtype/device,
        the data is written directly into `out` (zero-copy). Otherwise,
        a new tensor is allocated on CPU and, if necessary, moved to the
        inferred device.

        Args:
            out: Optional pre-allocated tensor to receive into. Must have
                 matching shape, dtype, and device; otherwise it is ignored
                 and a new tensor is allocated.

        Returns:
            The received tensor (either `out` reused, or a newly allocated tensor).

        Raises:
            NetTransportError: On framing/EOF/dtype errors.
        """
        header = self._recv_exact(self._HEADER_SIZE)
        (
            magic,
            msg_type,
            dtype_code,
            ndim,
            d0, d1, d2, d3, d4,
            payload_len,
        ) = struct.unpack(self._HEADER_FMT, header)

        if magic != self._MAGIC:
            raise NetTransportError(f"Bad magic: {hex(magic)}")
        if msg_type != self._MSG_TENSOR:
            raise NetTransportError(f"Expected tensor message, got {msg_type}")
        if dtype_code not in self._CODE_TO_DTYPE:
            raise NetTransportError(f"Unknown dtype code: {dtype_code}")

        dtype = self._CODE_TO_DTYPE[dtype_code]
        shape = (d0, d1, d2, d3, d4)[:ndim]

        # Determine device: prefer out's device, fall back to CPU
        target_device = torch.device("cpu")
        if out is not None:
            if out.shape != shape or out.dtype != dtype:
                # Mismatch: discard out and allocate fresh
                out = None
            else:
                target_device = out.device

        if out is None:
            out = torch.empty(shape, dtype = dtype, device = target_device)

        # Receive payload
        if out.numel() > 0:
            host_out = out if out.device.type == "cpu" else torch.empty(
                shape,
                dtype = dtype,
                device = torch.device("cpu"),
            )
            payload = self._recv_exact(payload_len)
            host_out.view(-1).copy_(
                torch.frombuffer(payload, dtype = dtype),
                non_blocking = False,
            )
            if out.device.type == "cuda":
                out.copy_(host_out, non_blocking = False)

        return out

    def send_obj(self, obj: Any) -> None:
        """
        Send a JSON-serializable object to the peer.

        Args:
            obj: Any JSON-serializable Python object (dict, list, int, str, etc.).

        Raises:
            NetTransportError: On serialization or send failure.
        """
        try:
            payload = json.dumps(obj).encode("utf-8")
        except (TypeError, ValueError) as e:
            raise NetTransportError(f"Object not JSON-serializable: {e}") from e

        payload_len = len(payload)

        # Build and send header (object message, no dtype/shape info)
        header = struct.pack(
            self._HEADER_FMT,
            self._MAGIC,
            self._MSG_OBJECT,
            0,
            0,
            0, 0, 0, 0, 0,
            payload_len,
        )
        self._sendall(header)
        if payload_len > 0:
            self._sendall(payload)

    def recv_obj(self) -> Any:
        """
        Receive a JSON-serializable object from the peer.

        Returns:
            Decoded Python object.

        Raises:
            NetTransportError: On framing/EOF/parse errors.
        """
        header = self._recv_exact(self._HEADER_SIZE)
        (
            magic,
            msg_type,
            _,
            _,
            _, _, _, _, _,
            payload_len,
        ) = struct.unpack(self._HEADER_FMT, header)

        if magic != self._MAGIC:
            raise NetTransportError(f"Bad magic: {hex(magic)}")
        if msg_type != self._MSG_OBJECT:
            raise NetTransportError(f"Expected object message, got {msg_type}")

        payload = self._recv_exact(payload_len) if payload_len > 0 else b""

        try:
            return json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise NetTransportError(f"Object parse failed: {e}") from e

    def close(self) -> None:
        """Close the underlying socket."""
        if self.sock:
            self.sock.close()

    def _sendall(self, data: memoryview | bytes) -> None:
        """
        Send all bytes of data over the socket, looping until complete.

        Args:
            data: Bytes or memoryview to send.

        Raises:
            NetTransportError: On socket errors.
        """
        if isinstance(data, memoryview):
            data = bytes(data)

        total_sent = 0
        while total_sent < len(data):
            try:
                sent = self.sock.send(data[total_sent:])
                if sent == 0:
                    raise NetTransportError("Socket send returned 0 (peer closed)")
                total_sent += sent
            except socket.error as e:
                raise NetTransportError(f"Send failed: {e}") from e

    def _recv_exact(self, n: int) -> bytes:
        """
        Receive exactly n bytes from the socket, looping until complete.

        Args:
            n: Exact number of bytes to receive.

        Returns:
            Bytes received.

        Raises:
            NetTransportError: On EOF (partial read) or socket errors.
        """
        data = b""
        while len(data) < n:
            try:
                chunk = self.sock.recv(n - len(data))
                if not chunk:
                    raise NetTransportError(
                        f"EOF while expecting {n} bytes (got {len(data)})"
                    )
                data += chunk
            except socket.error as e:
                raise NetTransportError(f"Recv failed: {e}") from e

        return data

    def _get_pinned_buffer(self, size: int) -> torch.Tensor:
        """
        Allocate or reuse a pinned host buffer for CUDA staging.

        Grows the buffer if needed; never shrinks.

        Args:
            size: Required size in bytes.

        Returns:
            torch.Tensor (pinned, uint8) of at least `size` bytes.
        """
        if self._pinned_buffer is None or self._pinned_size < size:
            # Allocate with some headroom to avoid frequent reallocations
            new_size = int(size * 1.5) + 1024
            self._pinned_buffer = torch.empty(
                new_size,
                dtype = torch.uint8,
                device = torch.device("cpu"),
            ).pin_memory()
            self._pinned_size = new_size

        return self._pinned_buffer[:size]
