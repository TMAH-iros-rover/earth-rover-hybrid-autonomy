from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


class AutonomyStatusStore:
    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._status = dict(initial or {})

    def publish(self, status: dict[str, Any]) -> None:
        with self._lock:
            self._status = dict(status)

    def get(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)


class AutonomyStatusServer:
    def __init__(
        self,
        host: str,
        port: int,
        store: AutonomyStatusStore,
        actions: dict[str, Callable[[], dict[str, Any]]] | None = None,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("port must be in [0, 65535]")
        self.store = store
        self._server = ThreadingHTTPServer(
            (host, port), _handler_for(store, actions or {})
        )
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="mission1-autonomy-status",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def _handler_for(
    store: AutonomyStatusStore,
    actions: dict[str, Callable[[], dict[str, Any]]],
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/status":
                self._send(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
                return
            self._send(HTTPStatus.OK, store.get())

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self._headers()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            action = actions.get(path)
            if action is None:
                self._send(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
                return
            try:
                status = action()
                store.publish(status)
                self._send(HTTPStatus.OK, status)
            except Exception as exc:
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"detail": f"{type(exc).__name__}: {exc}"},
                )

        def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._headers()
            self.end_headers()
            self.wfile.write(body)

        def _headers(self) -> None:
            origin = self.headers.get("Origin")
            if origin in {"http://127.0.0.1:8000", "http://localhost:8000"}:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Cache-Control", "no-store, max-age=0")

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler
