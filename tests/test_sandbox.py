from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from apollobot.compute.sandbox import (
    ExecutionSandbox,
    SandboxError,
    SandboxPolicy,
    minimal_environment,
)


@pytest.mark.asyncio
async def test_local_sandbox_runs_without_inheriting_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    script = tmp_path / "check.py"
    script.write_text("import os\nprint('secret=' + str('OPENAI_API_KEY' in os.environ))\n")
    sandbox = ExecutionSandbox(SandboxPolicy(mode="local", allow_local=True))
    result = await sandbox.run_python(script, workspace=tmp_path, timeout_seconds=5)
    assert result.returncode == 0
    assert "secret=False" in result.stdout
    assert "must-not-leak" not in result.stdout


@pytest.mark.asyncio
async def test_sandbox_rejects_script_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("print('no')")
    sandbox = ExecutionSandbox(SandboxPolicy(mode="local"))
    with pytest.raises(SandboxError):
        await sandbox.run_python(outside, workspace=tmp_path, timeout_seconds=1)


def test_minimal_environment_excludes_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    environment = minimal_environment()
    assert "ANTHROPIC_API_KEY" not in environment
    assert "SUPABASE_SERVICE_ROLE_KEY" not in environment


def test_container_command_disables_network_and_mounts_inputs_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "analysis" / "scripts" / "run.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')")
    monkeypatch.setattr("apollobot.compute.sandbox.shutil.which", lambda _: "/usr/bin/docker")
    sandbox = ExecutionSandbox(SandboxPolicy(mode="container", network=False))
    command = sandbox.container_command(tmp_path, script.relative_to(tmp_path))
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert any("dst=/workspace,readonly" in item for item in command)
    assert any("dst=/workspace/data/processed,rw" in item for item in command)
    assert "no-new-privileges" in command
    assert "/tmp:rw,noexec,nosuid,size=256m" in command  # noqa: S108
    assert "HOME=/tmp" in command
    assert "XDG_CACHE_HOME=/tmp/.cache" in command
    assert "MPLCONFIGDIR=/tmp/matplotlib" in command


@pytest.mark.asyncio
async def test_cancelling_local_sandbox_terminates_process_group(tmp_path: Path) -> None:
    script = tmp_path / "long_run.py"
    pid_file = tmp_path / "child.pid"
    script.write_text(
        "import os, pathlib, time\n"
        "pathlib.Path('child.pid').write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    sandbox = ExecutionSandbox(SandboxPolicy(mode="local", allow_local=True))
    task = asyncio.create_task(sandbox.run_python(script, workspace=tmp_path, timeout_seconds=60))
    for _ in range(100):
        if pid_file.exists():
            break
        await asyncio.sleep(0.01)
    assert pid_file.exists()
    child_pid = int(pid_file.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
