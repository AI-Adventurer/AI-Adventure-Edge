from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Protocol


class EdgePublisher(Protocol):
    def publish(self, packet: dict[str, Any]) -> None:
        """Emit a packet to the configured sink."""

    def close(self) -> None:
        """Release sink resources."""


class JsonlPublisher:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path or "-"
        self._owns_stream = self.output_path != "-"
        if self._owns_stream:
            path = Path(self.output_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("a", encoding="utf-8")
        else:
            self._stream = sys.stdout

    def publish(self, packet: dict[str, Any]) -> None:
        self._stream.write(json.dumps(packet, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if self._owns_stream:
            self._stream.close()


class SocketIOPublisher:
    def __init__(
        self,
        url: str,
        event: str = "frame",
        namespace: str = "/edge/frames",
        socketio_path: str = "socket.io",
        transports: list[str] | None = None,
        timeout_sec: float = 5.0,
    ) -> None:
        self.url = self._normalize_url(url)
        self.event = event.strip() or "frame"
        self.namespace = self._normalize_namespace(namespace)
        self.socketio_path = socketio_path.strip("/") or "socket.io"
        self.transports = [item for item in (transports or ["polling", "websocket"]) if item]
        self.timeout_sec = float(timeout_sec)
        self._client = None

    @staticmethod
    def _normalize_url(url: str) -> str:
        trimmed = url.strip()
        if trimmed.startswith("ws://"):
            return "http://" + trimmed[len("ws://") :]
        if trimmed.startswith("wss://"):
            return "https://" + trimmed[len("wss://") :]
        return trimmed

    @staticmethod
    def _normalize_namespace(namespace: str) -> str:
        trimmed = namespace.strip()
        if not trimmed or trimmed == "/":
            return "/"
        return trimmed if trimmed.startswith("/") else f"/{trimmed}"

    def _connect(self):
        if self._client is not None and self._client.connected:
            return self._client
        try:
            import socketio
        except ImportError as exc:
            raise RuntimeError(
                "Socket.IO output requires python-socketio[client]. Install with: pip install '.[edge]'"
            ) from exc
        client = socketio.Client(reconnection=True, logger=False, engineio_logger=False)
        client.connect(
            self.url,
            socketio_path=self.socketio_path,
            transports=self.transports,
            wait=True,
            wait_timeout=self.timeout_sec,
            namespaces=[self.namespace],
        )
        self._client = client
        return client

    def publish(self, packet: dict[str, Any]) -> None:
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                client = self._connect()
                client.emit(self.event, packet, namespace=self.namespace)
                return
            except Exception as exc:
                last_error = exc
                self.close()
        raise RuntimeError(
            "Could not publish edge packet to Socket.IO "
            f"{self.url} namespace={self.namespace!r} event={self.event!r} "
            f"path={self.socketio_path!r} transports={self.transports!r}"
        ) from last_error

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            finally:
                self._client = None


class MultiPublisher:
    def __init__(self, publishers: list[EdgePublisher]) -> None:
        self.publishers = publishers

    def publish(self, packet: dict[str, Any]) -> None:
        for publisher in self.publishers:
            publisher.publish(packet)

    def close(self) -> None:
        for publisher in self.publishers:
            publisher.close()


def build_edge_publisher(
    *,
    output_path: str = "",
    sio_url: str = "",
    sio_event: str = "frame",
    sio_namespace: str = "/edge/frames",
    sio_path: str = "socket.io",
    sio_transports: list[str] | None = None,
) -> EdgePublisher:
    publishers: list[EdgePublisher] = []
    if output_path or not sio_url:
        publishers.append(JsonlPublisher(output_path or "-"))
    if sio_url:
        publishers.append(
            SocketIOPublisher(
                url=sio_url,
                event=sio_event,
                namespace=sio_namespace,
                socketio_path=sio_path,
                transports=sio_transports,
            )
        )
    if len(publishers) == 1:
        return publishers[0]
    return MultiPublisher(publishers)
