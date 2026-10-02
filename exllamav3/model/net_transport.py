"""
Standalone tensor transport module for cross-host pipeline mode in exllamav3.

Moves torch tensors and small Python control messages between two processes on
DIFFERENT machines over TCP, as the payload path for a pipeline split where one
node runs layers 0..k and the other k+1..N.

The transport layer handles:
- Fixed-size framing with magic, msg type, dtype code, shape, payload length
- Reliable recv loops that handle partial reads and detect EOF
- Fast CUDA tensor staging via reused pinned host memory
- Direct zero-copy recv_into from socket into pinned buffers for GPU transfers
- Support for FP8 (float8_e4m3fn) alongside float16, bfloat16, float32, int32, int64, uint8, bool
- Non-contiguous tensor handling (make contiguous before send)
- Socket buffer tuning for high-throughput networks (QSFP/100GbE)
- Clear error handling with NetTransportError
"""

from __future__ import annotations

import socket
import struct
import json
import io
import sys
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

    # Register float8_e4m3fn if supported in this torch build
    if hasattr(torch, "float8_e4m3fn"):
        _DTYPE_TO_CODE[torch.float8_e4m3fn] = 7

    _CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}

    # Message types
    _MSG_TENSOR = 0
    _MSG_OBJECT = 1

    # Fixed header layout:
    # magic (4 bytes): 0xDEADBEEF
    # msg_type (1 byte): 0=tensor, 1=object
    # dtype_code (1 byte): 0-7 for tensor types
    # ndim (1 byte): number of dimensions
    # shape (40 bytes): 5 uint64 slots for dims (up to 5D)
    # payload_len (8 bytes): byte count of payload
    # Total: 4 + 1 + 1 + 1 + 40 + 8 = 55 bytes
    _HEADER_FMT = "!I B B B 5Q Q"
    _HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _MAGIC = 0xDEADBEEF

    def __init__(self, sock: socket.socket, is_server: bool = False, buffer_size: int = 4 * 1024 * 1024):
        """
        Initialize endpoint with an existing socket.

        Args:
            sock: Connected socket.socket instance.
            is_server: Whether this endpoint is the listening side (for logging/clarity).
            buffer_size: TCP send/recv buffer size in bytes (default 4MB for high-speed interconnects).
        """
        self.sock = sock
        self.is_server = is_server

        # Low latency socket tuning
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

        # Socket buffer sizing for large prefill chunks (e.g. 100GbE / QSFP)
        if buffer_size > 0:
            try:
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
            except OSError:
                pass

        # Quick-ACK on Linux if available to minimize handshake delays
        if hasattr(socket, "TCP_QUICKACK"):
            try:
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
            except OSError:
                pass

        # Pinned staging buffer for CUDA <-> host transfers (reused across calls)
        self._pinned_buffer = None
        self._pinned_size = 0
        # Event recorded after an asynchronous host-to-device copy out of the staging buffer
        # (recv_tensor with a stream). The buffer must not be overwritten until it completes
        self._staging_event = None

    @classmethod
    def listen(
        cls,
        host: str,
        port: int,
        timeout: float = 60.0,
        buffer_size: int = 4 * 1024 * 1024,
    ) -> NetEndpoint:
        """
        Create a server endpoint that listens for and accepts one peer connection.

        Args:
            host: Interface to bind to (e.g., "0.0.0.0").
            port: Port to listen on.
            timeout: Accept timeout in seconds.
            buffer_size: TCP socket buffer size.

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

            return cls(peer_sock, is_server = True, buffer_size = buffer_size)
        except Exception as e:
            raise NetTransportError(f"listen on {host}:{port} failed: {e}") from e

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        timeout: float = 60.0,
        buffer_size: int = 4 * 1024 * 1024,
    ) -> NetEndpoint:
        """
        Create a client endpoint that connects to a listening peer.

        Retries with exponential backoff until timeout expires.

        Args:
            host: Remote host to connect to.
            port: Remote port to connect to.
            timeout: Total time to retry before giving up (seconds).
            buffer_size: TCP socket buffer size.

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
                return cls(sock, is_server = False, buffer_size = buffer_size)
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

    def send_tensor(self, t: torch.Tensor, stream: torch.cuda.Stream | None = None) -> None:
        """
        Send a tensor to the peer.

        Non-contiguous tensors are made contiguous before sending.
        CUDA tensors are staged through pinned host memory.

        Args:
            t: torch.Tensor to send (any shape, any supported dtype/device).
            stream: Optional CUDA stream for asynchronous staging.

        Raises:
            NetTransportError: If dtype is unsupported or send fails.
        """
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

        # Stage a CUDA tensor through the reused pinned buffer
        if t.is_cuda:
            staging = self._get_pinned_buffer(payload_len)
            send_data = staging[:payload_len].view(t.dtype).view(t.shape)
            if stream is not None:
                with torch.cuda.stream(stream):
                    send_data.copy_(t, non_blocking = True)
                stream.synchronize()
            else:
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

        # Send payload
        if send_data.numel() > 0:
            # send_data viewed as uint8 avoids numpy conversion issues with bfloat16 / fp8
            u8_tensor = send_data.view(torch.uint8)
            # Use buffer interface directly
            self._sendall(memoryview(u8_tensor.numpy()))

    def recv_tensor(
        self,
        out: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """
        Receive a tensor from the peer.

        If `out` is provided and matches shape/dtype, data is written directly into `out`.
        If targeting CUDA, reads directly into pinned memory via recv_into, then copies to GPU.

        Args:
            out: Optional pre-allocated tensor to receive into.
            device: Target device if out is None (defaults to out.device or CPU).
            stream: Optional CUDA stream for async host-to-device copy.

        Returns:
            The received tensor.

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

        # Determine target device
        if out is not None:
            if out.shape != shape or out.dtype != dtype:
                out = None
                target_device = torch.device(device) if device else torch.device("cpu")
            else:
                target_device = out.device
        else:
            target_device = torch.device(device) if device else torch.device("cpu")

        if out is None:
            out = torch.empty(shape, dtype = dtype, device = target_device)

        if payload_len == 0:
            return out

        if target_device.type == "cuda":
            # Fast zero-copy path: recv directly into reusable pinned host buffer
            staging = self._get_pinned_buffer(payload_len)
            mv = memoryview(staging[:payload_len].view(torch.uint8).numpy()).cast("B")
            self._recv_into_exact(mv)

            host_view = staging[:payload_len].view(dtype).view(shape)
            if stream is not None:
                with torch.cuda.stream(stream):
                    out.copy_(host_view, non_blocking = True)
                # The copy may still be reading the staging buffer when this returns; the next
                # send/recv that reuses the buffer waits for this event first
                self._staging_event = torch.cuda.Event()
                self._staging_event.record(stream)
            else:
                out.copy_(host_view, non_blocking = False)
        else:
            # CPU target: read directly into destination tensor buffer
            mv = memoryview(out.view(torch.uint8).numpy()).cast("B")
            self._recv_into_exact(mv)

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
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()

    def _sendall(self, data: memoryview | bytes) -> None:
        """
        Send all bytes of data over the socket, looping until complete.

        Args:
            data: Bytes or memoryview to send.

        Raises:
            NetTransportError: On socket errors.
        """
        total_sent = 0
        total_len = len(data)
        while total_sent < total_len:
            try:
                sent = self.sock.send(data[total_sent:])
                if sent == 0:
                    raise NetTransportError("Socket send returned 0 (peer closed)")
                total_sent += sent
            except socket.error as e:
                raise NetTransportError(f"Send failed: {e}") from e

    def _recv_exact(self, n: int) -> bytes:
        """
        Receive exactly n bytes from the socket into a new bytes object.

        Args:
            n: Exact number of bytes to receive.

        Returns:
            Bytes received.

        Raises:
            NetTransportError: On EOF (partial read) or socket errors.
        """
        data = bytearray(n)
        mv = memoryview(data)
        self._recv_into_exact(mv)
        return bytes(data)

    def _recv_into_exact(self, buffer: memoryview) -> None:
        """
        Receive exactly len(buffer) bytes directly into an existing memoryview.
        Avoids extra memory allocations and copying.

        Args:
            buffer: Writable memoryview.

        Raises:
            NetTransportError: On EOF or socket errors.
        """
        n = len(buffer)
        pos = 0
        while pos < n:
            try:
                nbytes = self.sock.recv_into(buffer[pos:])
                if nbytes == 0:
                    raise NetTransportError(
                        f"EOF while expecting {n} bytes (got {pos})"
                    )
                pos += nbytes
            except socket.error as e:
                raise NetTransportError(f"Recv failed: {e}") from e

    def _get_pinned_buffer(self, size: int) -> torch.Tensor:
        """
        Allocate or reuse a pinned host buffer for CUDA staging.

        Grows the buffer if needed; never shrinks.

        Args:
            size: Required size in bytes.

        Returns:
            torch.Tensor (pinned, uint8) of at least `size` bytes.
        """
        if self._staging_event is not None:
            self._staging_event.synchronize()
            self._staging_event = None
        if self._pinned_buffer is None or self._pinned_size < size:
            new_size = int(size * 1.5) + 1024
            self._pinned_buffer = torch.empty(
                new_size,
                dtype = torch.uint8,
                device = torch.device("cpu"),
            ).pin_memory()
            self._pinned_size = new_size

        return self._pinned_buffer[:size]
