"""Resource-bounded execution for generated scientific Python.

The web service requests container mode by default. Local mode exists for the
open-source CLI and tests, but it still strips inherited secrets and uses an
async process with wall-time and output ceilings.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import signal
import sys
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path


class SandboxError(RuntimeError):
    """Raised when an execution environment cannot be created safely."""


@dataclass(frozen=True)
class SandboxPolicy:
    mode: str = "local"
    image: str = "frontier-science/apollobot-sandbox:py312"
    memory_mb: int = 2048
    cpus: float = 2.0
    pids: int = 256
    output_bytes: int = 2_000_000
    network: bool = False
    allow_local: bool = True
    endpoint: str = "http://sandbox.internal"

    @classmethod
    def from_environment(cls, *, mode: str | None = None) -> SandboxPolicy:
        selected = mode or os.getenv("APOLLOBOT_SANDBOX_MODE", "local")
        return cls(
            mode=selected,
            image=os.getenv(
                "APOLLOBOT_SANDBOX_IMAGE",
                "frontier-science/apollobot-sandbox:py312",
            ),
            memory_mb=int(os.getenv("APOLLOBOT_SANDBOX_MEMORY_MB", "2048")),
            cpus=float(os.getenv("APOLLOBOT_SANDBOX_CPUS", "2")),
            pids=int(os.getenv("APOLLOBOT_SANDBOX_PIDS", "256")),
            output_bytes=int(os.getenv("APOLLOBOT_SANDBOX_OUTPUT_BYTES", "2000000")),
            network=os.getenv("APOLLOBOT_SANDBOX_NETWORK", "off").lower() == "on",
            allow_local=os.getenv("APOLLOBOT_ALLOW_LOCAL_EXECUTION", "1") == "1",
            endpoint=os.getenv("APOLLOBOT_SANDBOX_URL", "http://sandbox.internal"),
        )


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False


class ExecutionSandbox:
    def __init__(self, policy: SandboxPolicy) -> None:
        self.policy = policy

    async def run_python(
        self,
        script: str | Path,
        *,
        workspace: str | Path,
        timeout_seconds: int,
    ) -> SandboxResult:
        root = Path(workspace).resolve()
        script_path = Path(script).resolve()
        if root not in script_path.parents or not script_path.is_file():
            raise SandboxError("Analysis script must be a file inside the session workspace")
        relative_script = script_path.relative_to(root)

        if self.policy.mode == "cloudflare":
            return await self._run_cloudflare(root, relative_script, timeout_seconds)
        if self.policy.mode == "container":
            command = self.container_command(root, relative_script)
            environment = minimal_environment()
        elif self.policy.mode == "local":
            if not self.policy.allow_local:
                raise SandboxError("Local generated-code execution is disabled")
            command = [sys.executable, str(script_path)]
            environment = minimal_environment()
        else:
            raise SandboxError(f"Unsupported sandbox mode: {self.policy.mode}")

        return await self._run(
            command,
            workspace=root,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )

    async def _run_cloudflare(
        self,
        workspace: Path,
        relative_script: Path,
        timeout_seconds: int,
    ) -> SandboxResult:
        import httpx

        archive = workspace_archive(workspace)
        run_id = uuid.uuid4().hex
        endpoint = f"{self.policy.endpoint.rstrip('/')}/run/{run_id}"
        headers = {
            "content-type": "application/gzip",
            "x-apollo-script": relative_script.as_posix(),
            "x-apollo-timeout": str(timeout_seconds),
        }
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds + 45) as client:
                response = await client.put(endpoint, content=archive, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise SandboxError(f"Cloudflare sandbox request failed: {error}") from error
        result = extract_sandbox_response(response.content, workspace)
        limit = self.policy.output_bytes
        stdout_bytes = result["stdout"].encode()
        stderr_bytes = result["stderr"].encode()
        truncated = (
            bool(result.get("truncated")) or len(stdout_bytes) > limit or len(stderr_bytes) > limit
        )
        stdout = stdout_bytes[:limit].decode("utf-8", errors="replace")
        stderr = stderr_bytes[:limit].decode("utf-8", errors="replace")
        if truncated:
            stderr = f"{stderr}\n[output truncated by sandbox]".strip()
        return SandboxResult(
            returncode=int(result["returncode"]),
            stdout=stdout,
            stderr=stderr,
            timed_out=bool(result.get("timed_out")),
            truncated=truncated,
        )

    def container_command(self, workspace: Path, relative_script: Path) -> list[str]:
        docker = shutil.which("docker")
        if not docker:
            raise SandboxError(
                "Container execution is required but Docker is unavailable on this worker"
            )
        network = "bridge" if self.policy.network else "none"
        processed_dir = workspace / "data" / "processed"
        figures_dir = workspace / "figures"
        processed_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        command = [
            docker,
            "run",
            "--rm",
            "--network",
            network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.policy.pids),
            "--memory",
            f"{self.policy.memory_mb}m",
            "--cpus",
            str(self.policy.cpus),
            "--tmpfs",
            " /tmp:rw,noexec,nosuid,size=256m".strip(),
            "--env",
            "HOME=/tmp",
            "--env",
            "XDG_CACHE_HOME=/tmp/.cache",
            "--env",
            "MPLCONFIGDIR=/tmp/matplotlib",
            "--mount",
            f"type=bind,src={workspace},dst=/workspace,readonly",
            "--mount",
            f"type=bind,src={processed_dir},dst=/workspace/data/processed,rw",
            "--mount",
            f"type=bind,src={figures_dir},dst=/workspace/figures,rw",
            "--workdir",
            "/workspace",
        ]
        if hasattr(os, "getuid"):
            command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        command.extend([self.policy.image, "python", f"/workspace/{relative_script.as_posix()}"])
        return command

    async def _run(
        self,
        command: list[str],
        *,
        workspace: Path,
        environment: dict[str, str],
        timeout_seconds: int,
    ) -> SandboxResult:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(workspace),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
        except TimeoutError:
            timed_out = True
            stdout_bytes, stderr_bytes = await terminate_process(process)
        except asyncio.CancelledError:
            await terminate_process(process)
            raise

        limit = self.policy.output_bytes
        truncated = len(stdout_bytes) > limit or len(stderr_bytes) > limit
        stdout = stdout_bytes[:limit].decode("utf-8", errors="replace")
        stderr = stderr_bytes[:limit].decode("utf-8", errors="replace")
        if truncated:
            stderr = f"{stderr}\n[output truncated by sandbox]".strip()
        return SandboxResult(
            returncode=process.returncode if process.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            truncated=truncated,
        )


async def terminate_process(
    process: asyncio.subprocess.Process,
) -> tuple[bytes, bytes]:
    """Stop the whole generated-code process group and drain its pipes."""
    if process.returncode is None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except TimeoutError:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
    try:
        return await asyncio.wait_for(process.communicate(), timeout=3)
    except TimeoutError:
        return b"", b"Process output could not be drained after termination."


def minimal_environment() -> dict[str, str]:
    """Return a non-secret execution environment.

    Provider keys, database credentials, webhook secrets, and the host Python
    path are intentionally absent.
    """
    return {
        "PATH": os.getenv("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "HOME": tempfile.gettempdir(),
    }


def workspace_archive(workspace: Path, limit: int = 96 * 1024 * 1024) -> bytes:
    """Create a symlink-free archive with a bounded uncompressed size."""
    total = 0
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(workspace.rglob("*")):
            relative = path.relative_to(workspace)
            if path.is_symlink() or any(part in {".git", "__pycache__"} for part in relative.parts):
                continue
            if path.is_dir():
                continue
            total += path.stat().st_size
            if total > limit:
                raise SandboxError("Sandbox workspace exceeds the 96 MiB safety limit")
            archive.add(path, arcname=relative.as_posix(), recursive=False)
    return buffer.getvalue()


def extract_sandbox_response(payload: bytes, workspace: Path) -> dict[str, object]:
    """Safely merge the sandbox's writable output directories into a workspace."""
    metadata: dict[str, object] | None = None
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise SandboxError("Sandbox returned an unsafe archive path")
            if member.name == "_apollobot_result.json":
                extracted = archive.extractfile(member)
                if not extracted:
                    raise SandboxError("Sandbox response metadata is unreadable")
                value = json.loads(extracted.read())
                if not isinstance(value, dict):
                    raise SandboxError("Sandbox response metadata is invalid")
                metadata = value
                continue
            if not path.parts or path.parts[0] not in {"figures", "data"}:
                continue
            if path.parts[0] == "data" and (len(path.parts) < 2 or path.parts[1] != "processed"):
                continue
            destination = (workspace / path).resolve()
            if workspace not in destination.parents:
                raise SandboxError("Sandbox output escaped the workspace")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise SandboxError("Sandbox returned an unsupported archive member")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source:
                with destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
    if metadata is None:
        raise SandboxError("Sandbox response did not include execution metadata")
    return metadata
