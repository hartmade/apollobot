"""No-network execution server used only inside a Cloudflare sandbox container."""

from __future__ import annotations

import io
import json
import os
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8090
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_OUTPUT_BYTES = 2_000_000
WRITABLE_ROOTS = (Path("data/processed"), Path("figures"))


def safe_extract(payload: bytes, destination: Path) -> None:
    total = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Archive path escapes the workspace")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("Links and devices are not accepted")
            target = (destination / relative).resolve()
            if destination not in target.parents:
                raise ValueError("Archive path escapes the workspace")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError("Unsupported archive member")
            total += member.size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds the extraction limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source:
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def execution_limits(timeout: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (timeout + 5, timeout + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024 * 1024, 128 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))


def response_archive(workspace: Path, metadata: dict[str, object]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        encoded = json.dumps(metadata, separators=(",", ":")).encode()
        info = tarfile.TarInfo("_apollobot_result.json")
        info.size = len(encoded)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(encoded))
        total = 0
        for relative_root in WRITABLE_ROOTS:
            root = workspace / relative_root
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_symlink() or path.is_dir():
                    continue
                total += path.stat().st_size
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError("Sandbox outputs exceed the response limit")
                archive.add(path, arcname=path.relative_to(workspace).as_posix(), recursive=False)
    return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "ApolloSandbox/1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/ready":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ready"}')

    def do_PUT(self) -> None:  # noqa: N802
        if not self.path.startswith("/run/"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0 or length > MAX_ARCHIVE_BYTES:
                raise ValueError("Invalid archive size")
            script_name = self.headers.get("x-apollo-script", "")
            timeout = max(1, min(3600, int(self.headers.get("x-apollo-timeout", "300"))))
            with tempfile.TemporaryDirectory(prefix="apollo-run-") as directory:
                workspace = Path(directory).resolve()
                safe_extract(self.rfile.read(length), workspace)
                script = (workspace / script_name).resolve()
                if workspace not in script.parents or not script.is_file():
                    raise ValueError("Analysis script is outside the workspace")
                for relative in WRITABLE_ROOTS:
                    (workspace / relative).mkdir(parents=True, exist_ok=True)
                environment = {
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONUNBUFFERED": "1",
                    "HOME": "/tmp",
                    "XDG_CACHE_HOME": "/tmp/.cache",
                    "MPLCONFIGDIR": "/tmp/matplotlib",
                }
                timed_out = False
                try:
                    process = subprocess.run(
                        [sys.executable, str(script)],
                        cwd=workspace,
                        env=environment,
                        capture_output=True,
                        timeout=timeout,
                        check=False,
                        preexec_fn=lambda: execution_limits(timeout),
                    )
                    returncode = process.returncode
                    stdout = process.stdout
                    stderr = process.stderr
                except subprocess.TimeoutExpired as error:
                    timed_out = True
                    returncode = -1
                    stdout = error.stdout or b""
                    stderr = error.stderr or b""
                truncated = len(stdout) > MAX_OUTPUT_BYTES or len(stderr) > MAX_OUTPUT_BYTES
                metadata = {
                    "returncode": returncode,
                    "stdout": stdout[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
                    "stderr": stderr[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
                    "timed_out": timed_out,
                    "truncated": truncated,
                }
                payload = response_archive(workspace, metadata)
            self.send_response(200)
            self.send_header("content-type", "application/gzip")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as error:
            payload = json.dumps({"error": str(error)}).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        print(format % args, flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
