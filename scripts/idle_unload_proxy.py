#!/usr/bin/env python3
import argparse
import http.client
import json
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class BackendManager:
    def __init__(
        self,
        backend_cmd: list[str],
        backend_host: str,
        backend_port: int,
        idle_timeout_seconds: float,
        startup_timeout_seconds: float,
        shutdown_grace_seconds: float,
        idle_poll_seconds: float,
        request_timeout_seconds: float | None,
    ) -> None:
        self.backend_cmd = backend_cmd
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.idle_timeout_seconds = idle_timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self.shutdown_grace_seconds = shutdown_grace_seconds
        self.idle_poll_seconds = idle_poll_seconds
        self.request_timeout_seconds = request_timeout_seconds

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._process: subprocess.Popen[bytes] | None = None
        self._starting = False
        self._active_requests = 0
        self._last_activity = time.monotonic()
        self._shutting_down = False

    def status(self) -> dict[str, object]:
        with self._lock:
            process = self._process
            running = bool(process and process.poll() is None)
            return {
                "backend_running": running,
                "active_requests": self._active_requests,
                "idle_timeout_seconds": self.idle_timeout_seconds,
                "seconds_since_last_activity": round(
                    max(0.0, time.monotonic() - self._last_activity), 3
                ),
            }

    def mark_request_start(self) -> None:
        with self._lock:
            self._active_requests += 1
            self._last_activity = time.monotonic()

    def mark_request_end(self) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._last_activity = time.monotonic()

    def ensure_backend(self) -> None:
        with self._cond:
            self._last_activity = time.monotonic()
            while True:
                if self._shutting_down:
                    raise RuntimeError("proxy is shutting down")

                process = self._process
                if process and process.poll() is None:
                    return

                if not self._starting:
                    self._starting = True
                    break

                self._cond.wait()

        process: subprocess.Popen[bytes] | None = None
        try:
            print(
                "Starting vLLM backend: {}".format(" ".join(self.backend_cmd)),
                flush=True,
            )
            process = subprocess.Popen(self.backend_cmd)
            self._wait_until_ready(process)
        except Exception:
            if process and process.poll() is None:
                self._terminate_process(process, reason="failed startup")
            with self._cond:
                self._process = None
                self._starting = False
                self._cond.notify_all()
            raise

        with self._cond:
            self._process = process
            self._starting = False
            self._last_activity = time.monotonic()
            self._cond.notify_all()

    def stop_backend(self, reason: str) -> None:
        with self._cond:
            process = self._process
            if not process:
                return
            self._process = None
            self._cond.notify_all()

        if process.poll() is None:
            print(f"Stopping vLLM backend: {reason}", flush=True)
            self._terminate_process(process, reason=reason)

    def shutdown(self) -> None:
        with self._cond:
            self._shutting_down = True
            self._cond.notify_all()
        self.stop_backend("proxy shutdown")

    def reap_if_idle(self) -> None:
        if self.idle_timeout_seconds <= 0:
            return

        while True:
            with self._lock:
                if self._shutting_down:
                    return
                process = self._process
                active_requests = self._active_requests
                idle_for = time.monotonic() - self._last_activity

            if process and process.poll() is not None:
                with self._cond:
                    if self._process is process:
                        self._process = None
                        self._cond.notify_all()
            elif process and active_requests == 0 and idle_for >= self.idle_timeout_seconds:
                self.stop_backend(
                    f"idle for {int(idle_for)}s (timeout={int(self.idle_timeout_seconds)}s)"
                )

            time.sleep(self.idle_poll_seconds)

    def _wait_until_ready(self, process: subprocess.Popen[bytes]) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_error = "backend did not accept connections"

        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"backend exited before becoming ready with code {process.returncode}"
                )

            try:
                connection = http.client.HTTPConnection(
                    self.backend_host,
                    self.backend_port,
                    timeout=3,
                )
                connection.request("GET", "/")
                response = connection.getresponse()
                response.read()
                print(
                    f"vLLM backend is ready on {self.backend_host}:{self.backend_port}",
                    flush=True,
                )
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(1)
            finally:
                try:
                    connection.close()
                except Exception:
                    pass

        raise TimeoutError(
            "timed out waiting for backend startup after "
            f"{int(self.startup_timeout_seconds)}s: {last_error}"
        )

    def _terminate_process(
        self,
        process: subprocess.Popen[bytes],
        reason: str,
    ) -> None:
        process.terminate()
        try:
            process.wait(timeout=self.shutdown_grace_seconds)
        except subprocess.TimeoutExpired:
            print(
                f"Backend did not exit after SIGTERM during {reason}; sending SIGKILL",
                flush=True,
            )
            process.kill()
            process.wait()


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_PATCH(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def do_OPTIONS(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        self._handle()

    def log_message(self, format: str, *args: object) -> None:
        sys.stdout.write(
            "%s - - [%s] %s\n"
            % (self.client_address[0], self.log_date_time_string(), format % args)
        )
        sys.stdout.flush()

    def _handle(self) -> None:
        if self.path == "/healthz":
            self._write_json(200, {"status": "ok", **self.server.backend_manager.status()})
            return

        self.server.backend_manager.mark_request_start()
        try:
            self.server.backend_manager.ensure_backend()
            self._proxy_request()
        except Exception as exc:
            self._write_json(
                503,
                {
                    "error": "backend unavailable",
                    "detail": str(exc),
                },
            )
        finally:
            self.server.backend_manager.mark_request_end()

    def _proxy_request(self) -> None:
        content_length_header = self.headers.get("Content-Length")
        body = None
        if content_length_header:
            body = self.rfile.read(int(content_length_header))

        connection = http.client.HTTPConnection(
            self.server.backend_manager.backend_host,
            self.server.backend_manager.backend_port,
            timeout=self.server.backend_manager.request_timeout_seconds,
        )
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        headers["Host"] = (
            f"{self.server.backend_manager.backend_host}:"
            f"{self.server.backend_manager.backend_port}"
        )

        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()

            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() in HOP_BY_HOP_HEADERS:
                    continue
                self.send_header(key, value)
            self.end_headers()

            if self.command != "HEAD":
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            connection.close()

    def _write_json(self, status: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)
            self.wfile.flush()


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler_class, backend_manager):
        super().__init__(server_address, handler_class)
        self.backend_manager = backend_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lazy-start proxy for vLLM that unloads the backend after an idle timeout."
    )
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--backend-host", required=True)
    parser.add_argument("--backend-port", type=int, required=True)
    parser.add_argument("--idle-timeout-seconds", type=float, required=True)
    parser.add_argument("--startup-timeout-seconds", type=float, default=900)
    parser.add_argument("--shutdown-grace-seconds", type=float, default=30)
    parser.add_argument("--idle-poll-seconds", type=float, default=5)
    parser.add_argument("--request-timeout-seconds", type=float, default=0)
    parser.add_argument("backend_cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.backend_cmd and args.backend_cmd[0] == "--":
        args.backend_cmd = args.backend_cmd[1:]
    if not args.backend_cmd:
        parser.error("backend command is required after '--'")
    return args


def main() -> int:
    args = parse_args()

    backend_manager = BackendManager(
        backend_cmd=args.backend_cmd,
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        idle_timeout_seconds=args.idle_timeout_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
        shutdown_grace_seconds=args.shutdown_grace_seconds,
        idle_poll_seconds=args.idle_poll_seconds,
        request_timeout_seconds=(
            None if args.request_timeout_seconds <= 0 else args.request_timeout_seconds
        ),
    )

    watcher = threading.Thread(target=backend_manager.reap_if_idle, daemon=True)
    watcher.start()

    server = ProxyServer((args.listen_host, args.listen_port), ProxyHandler, backend_manager)

    def handle_signal(signum, _frame) -> None:
        print(f"Received signal {signum}; shutting down idle-unload proxy", flush=True)
        backend_manager.shutdown()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print(
        "Idle-unload proxy listening on "
        f"{args.listen_host}:{args.listen_port}; backend on {args.backend_host}:{args.backend_port}; "
        f"timeout={int(args.idle_timeout_seconds)}s",
        flush=True,
    )

    try:
        server.serve_forever()
    finally:
        server.server_close()
        backend_manager.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
