from __future__ import annotations

import asyncio
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray | None) -> None:
        if frame_bgr is None:
            return
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3:
            return
        with self._lock:
            self._latest_frame = (
                frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)
            )

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_frame


def _resolve_video_size(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    frame_h, frame_w = frame.shape[:2]
    if target_width > 0 and target_height > 0:
        return target_width, target_height
    if target_width > 0:
        scale = target_width / max(1, frame_w)
        return target_width, max(1, int(round(frame_h * scale)))
    if target_height > 0:
        scale = target_height / max(1, frame_h)
        return max(1, int(round(frame_w * scale))), target_height
    return frame_w, frame_h


def _build_shared_video_track_class():
    try:
        from aiortc import VideoStreamTrack
    except ImportError as exc:
        raise RuntimeError(
            "WebRTC video output requires aiortc. Install with: pip install '.[edge-video]'"
        ) from exc

    class SharedVideoTrack(VideoStreamTrack):
        def __init__(
            self,
            frame_buffer: LatestFrameBuffer,
            *,
            fps: float = 15.0,
            width: int = 0,
            height: int = 0,
        ) -> None:
            super().__init__()
            self.frame_buffer = frame_buffer
            self.fps = max(1.0, float(fps))
            self.target_width = max(0, int(width))
            self.target_height = max(0, int(height))
            self._next_frame_at = 0.0
            self._fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        async def recv(self):
            from av import VideoFrame

            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._next_frame_at <= 0.0:
                self._next_frame_at = now
            sleep_for = self._next_frame_at - now
            if sleep_for > 0.0:
                await asyncio.sleep(sleep_for)
            self._next_frame_at = max(self._next_frame_at + (1.0 / self.fps), loop.time())

            frame = self.frame_buffer.snapshot()
            if frame is None:
                frame = self._fallback_frame
            else:
                target_width, target_height = _resolve_video_size(
                    frame,
                    self.target_width,
                    self.target_height,
                )
                if (target_width, target_height) != (frame.shape[1], frame.shape[0]):
                    frame = cv2.resize(frame, (target_width, target_height))
                self._fallback_frame = frame

            pts, time_base = await self.next_timestamp()
            video_frame = VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

    return SharedVideoTrack


@dataclass(slots=True)
class WebRTCVideoConfig:
    url: str
    source_id: str
    namespace: str = "/edge/video"
    socketio_path: str = "socket.io"
    transports: list[str] | None = None
    offer_event: str = "offer"
    answer_event: str = "answer"
    candidate_event: str = "candidate"
    response_event: str = "response"
    fps: float = 15.0
    width: int = 0
    height: int = 0
    timeout_sec: float = 10.0
    ice_servers: list[str] | None = None


class WebRTCVideoStreamer:
    def __init__(self, config: WebRTCVideoConfig) -> None:
        self.config = config
        self.frame_buffer = LatestFrameBuffer()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._start_error: Exception | None = None
        self._sio = None
        self._pc = None
        self._negotiation_lock: asyncio.Lock | None = None
        self._renegotiation_task: asyncio.Task | None = None
        self._pending_remote_candidates: list[dict[str, Any]] = []
        self._remote_description_applied = False
        self._peer_ready = False
        self._submit_interval_sec = 1.0 / max(1.0, float(self.config.fps))
        self._last_submit_at = 0.0

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

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_thread, name="edge-webrtc-video", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=max(1.0, self.config.timeout_sec))
        if self._start_error is not None:
            raise RuntimeError("Could not start WebRTC video streamer") from self._start_error

    def submit_frame(self, frame_bgr: np.ndarray | None) -> None:
        if not self._peer_ready:
            return
        now = time.monotonic()
        if (now - self._last_submit_at) < self._submit_interval_sec:
            return
        self._last_submit_at = now
        self.frame_buffer.update(frame_bgr)

    def close(self) -> None:
        self._stop_requested.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _log(self, message: str) -> None:
        print(f"[edge-video] {message}", file=sys.stderr)

    def _run_thread(self) -> None:
        try:
            asyncio.run(self._async_main())
        except Exception as exc:
            self._start_error = exc
            self._ready.set()
            self._log(f"fatal_error={exc}")

    async def _async_main(self) -> None:
        try:
            import socketio
            from aiortc import RTCPeerConnection
            from aiortc import RTCConfiguration, RTCIceServer, RTCSessionDescription
            from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
        except ImportError as exc:
            raise RuntimeError(
                "WebRTC video output requires aiortc and python-socketio. "
                "Install with: pip install '.[edge-video]'"
            ) from exc

        self._loop = asyncio.get_running_loop()
        self._candidate_from_sdp = candidate_from_sdp
        self._candidate_to_sdp = candidate_to_sdp
        self._rtc_session_description = RTCSessionDescription
        self._rtc_peer_connection = RTCPeerConnection
        self._rtc_configuration = RTCConfiguration
        self._rtc_ice_server = RTCIceServer
        self._negotiation_lock = asyncio.Lock()
        self._pending_remote_candidates = []
        self._remote_description_applied = False

        namespace = self._normalize_namespace(self.config.namespace)
        url = self._normalize_url(self.config.url)
        transports = [item for item in (self.config.transports or ["polling", "websocket"]) if item]

        self._sio = socketio.AsyncClient(reconnection=True, logger=False, engineio_logger=False)

        @self._sio.on("connect", namespace=namespace)
        async def _on_connect():
            self._log(f"connected namespace={namespace} url={url}")
            await self._safe_negotiate("socket_connect")

        @self._sio.on("disconnect", namespace=namespace)
        async def _on_disconnect():
            self._peer_ready = False
            self._log(f"disconnected namespace={namespace}")

        @self._sio.on("connect_error", namespace=namespace)
        async def _on_connect_error(data):
            self._log(f"connect_error={data}")

        @self._sio.on(self.config.response_event, namespace=namespace)
        async def _on_response(data):
            if isinstance(data, dict):
                self._log(f"server_response={data}")

        @self._sio.on(self.config.answer_event, namespace=namespace)
        async def _on_answer(data):
            try:
                await self._apply_answer(data or {})
            except Exception as exc:
                self._log(f"answer_error={exc}")
                self._schedule_renegotiation("answer_error")

        @self._sio.on(self.config.candidate_event, namespace=namespace)
        async def _on_candidate(data):
            try:
                await self._apply_remote_candidate(data or {})
            except Exception as exc:
                self._log(f"candidate_error={exc}")

        await self._sio.connect(
            url,
            socketio_path=self.config.socketio_path.strip("/") or "socket.io",
            transports=transports,
            wait=True,
            wait_timeout=float(self.config.timeout_sec),
            namespaces=[namespace],
        )
        self._ready.set()

        try:
            while not self._stop_requested.is_set():
                await asyncio.sleep(0.2)
        finally:
            if self._renegotiation_task is not None:
                self._renegotiation_task.cancel()
                self._renegotiation_task = None
            if self._pc is not None:
                await self._pc.close()
                self._pc = None
            if self._sio.connected:
                await self._sio.disconnect()

    async def _create_peer_connection(self):
        if self._pc is not None:
            await self._pc.close()
        self._pending_remote_candidates = []
        self._remote_description_applied = False
        self._peer_ready = False

        ice_servers = [
            self._rtc_ice_server(urls=url)
            for url in (self.config.ice_servers or [])
            if str(url).strip()
        ]
        rtc_config = self._rtc_configuration(iceServers=ice_servers)
        pc = self._rtc_peer_connection(rtc_config)
        self._pc = pc

        track_class = _build_shared_video_track_class()
        track_wrapper = track_class(
            self.frame_buffer,
            fps=self.config.fps,
            width=self.config.width,
            height=self.config.height,
        )
        pc.addTrack(track_wrapper)

        @pc.on("connectionstatechange")
        async def _on_connectionstatechange():
            if self._pc is not pc:
                return
            state = pc.connectionState
            self._log(f"pc_state={state}")
            if state in {"failed", "closed"}:
                self._peer_ready = False
                self._schedule_renegotiation(f"pc_state_{state}")

        @pc.on("iceconnectionstatechange")
        async def _on_iceconnectionstatechange():
            if self._pc is not pc:
                return
            state = pc.iceConnectionState
            self._log(f"ice_state={state}")
            if state == "failed":
                self._peer_ready = False
                self._schedule_renegotiation("ice_failed")

        @pc.on("icegatheringstatechange")
        async def _on_icegatheringstatechange():
            if self._pc is not pc:
                return
            self._log(f"ice_gathering_state={pc.iceGatheringState}")

        @pc.on("signalingstatechange")
        async def _on_signalingstatechange():
            if self._pc is not pc:
                return
            self._log(f"signaling_state={pc.signalingState}")

        return pc

    async def _safe_negotiate(self, reason: str) -> None:
        if self._stop_requested.is_set() or self._sio is None or self._negotiation_lock is None:
            return
        async with self._negotiation_lock:
            if self._stop_requested.is_set() or self._sio is None:
                return
            self._log(f"negotiate_start reason={reason}")
            await self._negotiate()

    def _schedule_renegotiation(self, reason: str) -> None:
        if self._stop_requested.is_set() or self._loop is None:
            return
        if self._renegotiation_task is not None and not self._renegotiation_task.done():
            return

        async def _runner() -> None:
            try:
                await asyncio.sleep(0.5)
                await self._safe_negotiate(reason)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._log(f"renegotiate_error reason={reason} error={exc}")
            finally:
                self._renegotiation_task = None

        self._renegotiation_task = asyncio.create_task(_runner())

    async def _negotiate(self) -> None:
        if self._sio is None:
            return
        pc = await self._create_peer_connection()
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        payload = {
            "type": "offer",
            "source": self.config.source_id,
            "sdp": pc.localDescription.sdp,
        }
        await self._sio.emit(
            self.config.offer_event,
            payload,
            namespace=self._normalize_namespace(self.config.namespace),
        )
        self._log("offer_sent")

    async def _apply_answer(self, data: dict[str, Any]) -> None:
        if self._pc is None:
            return
        source = str(data.get("source", "")).strip()
        if source and source == self.config.source_id:
            return
        target = str(data.get("target", "")).strip()
        if target and target != self.config.source_id:
            return
        sdp = str(data.get("sdp", "")).strip()
        if not sdp:
            return
        answer_type = str(data.get("type") or "answer")
        if self._pc.signalingState == "stable":
            self._log("answer_ignored reason=already_stable")
            return
        await self._pc.setRemoteDescription(
            self._rtc_session_description(sdp=sdp, type=answer_type)
        )
        self._remote_description_applied = True
        self._peer_ready = True
        self._log("answer_applied")
        await self._flush_pending_remote_candidates()

    def _candidate_from_payload(self, data: dict[str, Any]):
        candidate_sdp = str(data.get("candidate", "")).strip()
        if not candidate_sdp:
            return None
        if candidate_sdp.startswith("candidate:"):
            candidate_sdp = candidate_sdp[len("candidate:") :]
        candidate = self._candidate_from_sdp(candidate_sdp)
        candidate.sdpMid = data.get("sdpMid")
        if data.get("sdpMLineIndex") is not None:
            candidate.sdpMLineIndex = int(data["sdpMLineIndex"])
        return candidate

    async def _flush_pending_remote_candidates(self) -> None:
        if self._pc is None or not self._remote_description_applied:
            return
        pending = self._pending_remote_candidates
        if not pending:
            return
        self._pending_remote_candidates = []
        for candidate_data in pending:
            candidate = self._candidate_from_payload(candidate_data)
            if candidate is None:
                continue
            await self._pc.addIceCandidate(candidate)
            self._log("candidate_applied")

    async def _apply_remote_candidate(self, data: dict[str, Any]) -> None:
        if self._pc is None:
            return
        source = str(data.get("source", "")).strip()
        if source and source == self.config.source_id:
            return
        target = str(data.get("target", "")).strip()
        if target and target != self.config.source_id:
            return
        if not str(data.get("candidate", "")).strip():
            return
        if not self._remote_description_applied:
            self._pending_remote_candidates.append(dict(data))
            self._log(f"candidate_queued pending={len(self._pending_remote_candidates)}")
            return
        candidate = self._candidate_from_payload(data)
        if candidate is None:
            return
        await self._pc.addIceCandidate(candidate)
        self._log("candidate_applied")


def build_edge_video_streamer(
    *,
    url: str,
    source_id: str,
    namespace: str = "/edge/video",
    socketio_path: str = "socket.io",
    transports: list[str] | None = None,
    offer_event: str = "offer",
    answer_event: str = "answer",
    candidate_event: str = "candidate",
    response_event: str = "response",
    fps: float = 15.0,
    width: int = 0,
    height: int = 0,
    timeout_sec: float = 10.0,
    ice_servers: list[str] | None = None,
) -> WebRTCVideoStreamer:
    config = WebRTCVideoConfig(
        url=url,
        source_id=source_id,
        namespace=namespace,
        socketio_path=socketio_path,
        transports=transports,
        offer_event=offer_event,
        answer_event=answer_event,
        candidate_event=candidate_event,
        response_event=response_event,
        fps=fps,
        width=width,
        height=height,
        timeout_sec=timeout_sec,
        ice_servers=ice_servers,
    )
    return WebRTCVideoStreamer(config)
