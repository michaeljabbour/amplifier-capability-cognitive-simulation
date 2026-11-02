"""Real, short-lived CLI fixtures for cleanup on exceptional exits.

These scripts are fixtures, not a Centaur model. No network is used.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shlex
import signal
import sys
import tempfile

import pytest

from amplifier_module_tool_centaur.backends import CliBackend


def track_temp_files(monkeypatch, folder: Path, fail_second: bool = False):
    original = tempfile.NamedTemporaryFile
    created: list[Path] = []
    calls = 0

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        if fail_second and calls == 2:
            raise OSError("fixture output-file creation failure")
        kwargs["dir"] = folder
        result = original(*args, **kwargs)
        created.append(Path(result.name))
        return result

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracked)
    return created


@pytest.mark.asyncio
async def test_bad_template_removes_prompt_and_output_files(tmp_path, monkeypatch):
    files = track_temp_files(monkeypatch, tmp_path)
    with pytest.raises(KeyError, match="missing_placeholder"):
        await CliBackend("python {missing_placeholder}").predict("Fictional prompt <<")
    assert len(files) == 2
    assert all(not path.exists() for path in files)


@pytest.mark.asyncio
async def test_output_file_creation_failure_removes_prompt(tmp_path, monkeypatch):
    files = track_temp_files(monkeypatch, tmp_path, fail_second=True)
    with pytest.raises(OSError, match="fixture output-file"):
        await CliBackend("unused").predict("Fictional prompt <<")
    assert len(files) == 1
    assert not files[0].exists()


def waiting_fixture(tmp_path):
    script = tmp_path / "controlled_child.py"
    started = tmp_path / "started.txt"
    marker = tmp_path / "continued.txt"
    release = tmp_path / "release.txt"
    script.write_text(
        "from pathlib import Path\nimport time\n"
        f"Path({str(started)!r}).write_text('started')\n"
        f"while not Path({str(release)!r}).exists():\n    time.sleep(0.01)\n"
        f"Path({str(marker)!r}).write_text('continued')\n"
    )
    command = shlex.join([sys.executable, str(script), "{prompt_file}", "{output_file}"])
    return command, started, marker, release


def synchronize_startup(monkeypatch, started):
    """Start the backend's timeout only after this controlled fixture is ready."""
    original = asyncio.create_subprocess_shell
    ready = asyncio.Event()

    async def ready_shell(*args, **kwargs):
        proc = await original(*args, **kwargs)
        for _ in range(200):
            if started.exists():
                ready.set()
                return proc
            if proc.returncode is not None:
                raise AssertionError("controlled fixture exited before readiness")
            await asyncio.sleep(0.01)
        if kwargs.get("start_new_session"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        await proc.communicate()
        raise AssertionError("controlled fixture did not become ready within 2 seconds")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", ready_shell)
    return ready


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract only")
@pytest.mark.asyncio
async def test_timeout_stops_child_before_later_side_effect(tmp_path, monkeypatch):
    files = track_temp_files(monkeypatch, tmp_path)
    command, started, marker, release = waiting_fixture(tmp_path)
    synchronize_startup(monkeypatch, started)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await CliBackend(command, timeout=0.05).predict("Fictional prompt <<")
        assert started.exists()
        release.write_text("caller finished")
        await asyncio.sleep(0.1)
        assert not marker.exists()
        assert all(not path.exists() for path in files)
    finally:
        release.write_text("fixture cleanup")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract only")
@pytest.mark.asyncio
async def test_cancellation_stops_child_before_later_side_effect(tmp_path, monkeypatch):
    files = track_temp_files(monkeypatch, tmp_path)
    command, started, marker, release = waiting_fixture(tmp_path)
    ready = synchronize_startup(monkeypatch, started)
    task = asyncio.create_task(CliBackend(command, timeout=5).predict("Fictional prompt <<"))
    try:
        await asyncio.wait_for(ready.wait(), timeout=2.5)
        assert started.exists(), "controlled fixture must start before cancellation"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.write_text("caller finished")
        await asyncio.sleep(0.1)
        assert not marker.exists()
        assert all(not path.exists() for path in files)
    finally:
        release.write_text("fixture cleanup")
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
