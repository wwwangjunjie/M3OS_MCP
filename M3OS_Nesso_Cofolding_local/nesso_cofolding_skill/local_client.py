"""Lifecycle and file-queue client for the host-local Nesso worker."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any


class LocalWorkerError(RuntimeError):
    """Raised when the local worker cannot start or complete a job."""


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _remove_queued_job_files(queue_dir: Path, job_id: str) -> None:
    for state in ("pending", "running"):
        (queue_dir / state / f"{job_id}.json").unlink(missing_ok=True)


def _quarantine_stale_jobs(queue_dir: Path) -> list[Path]:
    abandoned_dir = queue_dir / "abandoned"
    abandoned_dir.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for state in ("pending", "running"):
        for source in sorted((queue_dir / state).glob("*.json")):
            destination = abandoned_dir / f"{state}_{source.name}"
            if destination.exists():
                destination = abandoned_dir / f"{state}_{uuid.uuid4().hex}_{source.name}"
            os.replace(source, destination)
            moved.append(destination)
    return moved


def local_queue_dir(runs_root: Path, gpu_id: str) -> Path:
    return runs_root / ".nesso_local" / f"gpu_{gpu_id}"


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    """Make a path absolute while preserving virtualenv interpreter links."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _read_pid(pid_path: Path) -> int | None:
    try:
        value = pid_path.read_text(encoding="utf-8").strip()
        pid = int(value)
    except (FileNotFoundError, ValueError):
        return None
    return pid if pid > 1 else None


def _worker_running(pid: int | None, worker_script: Path) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
    except OSError:
        return True
    return str(worker_script) in cmdline


def build_worker_command(
    *, python: Path, worker_script: Path, queue_dir: Path, cache_dir: Path
) -> list[str]:
    command = [
        str(python),
        str(worker_script),
        "--queue-dir",
        str(queue_dir),
        "--cache-dir",
        str(cache_dir),
    ]
    if os.environ.get("NESSO_LOCAL_NO_KERNELS", "0").lower() in {
        "1",
        "true",
        "yes",
    }:
        command.append("--no-kernels")
    return command


def ensure_local_worker(
    *,
    gpu_id: str,
    skill_dir: Path,
    runs_root: Path,
    python: Path,
    cache_dir: Path,
) -> tuple[Path, int, list[str]]:
    """Start one detached host worker when needed and wait until it is ready."""
    skill_dir = skill_dir.resolve()
    runs_root = runs_root.resolve()
    # Resolving .venv/bin/python would bypass pyvenv.cfg and launch the base
    # interpreter without the Nesso packages installed in the virtualenv.
    python = _absolute_without_resolving_symlinks(python)
    cache_dir = cache_dir.resolve()
    worker_script = skill_dir / "local_worker.py"
    if not python.is_file():
        raise FileNotFoundError(f"Local Nesso Python does not exist: {python}")
    if not worker_script.is_file():
        raise FileNotFoundError(worker_script)
    runs_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    queue_dir = local_queue_dir(runs_root, gpu_id)
    for child in ("pending", "running", "results", "completed", "abandoned"):
        (queue_dir / child).mkdir(parents=True, exist_ok=True)
    pid_path = queue_dir / "worker.pid"
    command = build_worker_command(
        python=python,
        worker_script=worker_script,
        queue_dir=queue_dir,
        cache_dir=cache_dir,
    )

    lock_path = queue_dir / "lifecycle.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        pid = _read_pid(pid_path)
        if not _worker_running(pid, worker_script):
            abandoned = _quarantine_stale_jobs(queue_dir)
            if abandoned:
                print(
                    f"Quarantined {len(abandoned)} stale Nesso queue job(s) before worker startup",
                    flush=True,
                )
            for marker in ("ready.json", "fatal.json"):
                (queue_dir / marker).unlink(missing_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            environment["NESSO_CACHE"] = str(cache_dir)
            log_path = queue_dir / "worker.log"
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    command,
                    cwd=skill_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
            pid = process.pid
            _atomic_write_text(pid_path, f"{pid}\n")

        timeout = _positive_float_env("NESSO_LOCAL_START_TIMEOUT_SECONDS", 600)
        deadline = time.monotonic() + timeout
        ready_path = queue_dir / "ready.json"
        fatal_path = queue_dir / "fatal.json"
        while time.monotonic() < deadline:
            if ready_path.is_file():
                ready = json.loads(ready_path.read_text(encoding="utf-8"))
                if int(ready.get("pid", -1)) != pid:
                    raise LocalWorkerError(
                        f"Ready marker belongs to another worker: {ready}"
                    )
                return queue_dir, pid, command
            if fatal_path.is_file():
                fatal = json.loads(fatal_path.read_text(encoding="utf-8"))
                raise LocalWorkerError(f"Local Nesso startup failed: {fatal}")
            if not _worker_running(pid, worker_script):
                raise LocalWorkerError(
                    f"Local Nesso worker {pid} exited; see {queue_dir / 'worker.log'}"
                )
            time.sleep(0.2)
        raise LocalWorkerError(
            f"Local Nesso worker was not ready within {timeout:.1f}s"
        )


def submit_local_job(
    *, queue_dir: Path, pid: int, worker_script: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    job_id = str(payload.get("job_id") or uuid.uuid4().hex)
    payload = {**payload, "job_id": job_id}
    request_path = queue_dir / "pending" / f"{job_id}.json"
    result_path = queue_dir / "results" / f"{job_id}.json"
    result_path.unlink(missing_ok=True)
    _atomic_write_json(request_path, payload)

    timeout = _positive_float_env("NESSO_LOCAL_JOB_TIMEOUT_SECONDS", 3600)
    poll_seconds = _positive_float_env("NESSO_LOCAL_POLL_SECONDS", 0.2)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if not result.get("success"):
                    raise LocalWorkerError(
                        f"Local Nesso job {job_id} failed: "
                        f"{result.get('error_code')}: {result.get('error_message')}"
                    )
                return result
            if not _worker_running(pid, worker_script):
                raise LocalWorkerError(
                    f"Local Nesso worker {pid} exited while running job {job_id}"
                )
            time.sleep(poll_seconds)
        raise LocalWorkerError(
            f"Local Nesso job {job_id} exceeded timeout {timeout:.1f}s"
        )
    except BaseException:
        _remove_queued_job_files(queue_dir, job_id)
        raise


def run_local_job(
    *,
    gpu_id: str,
    skill_dir: Path,
    runs_root: Path,
    python: Path,
    cache_dir: Path,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    queue_dir, pid, command = ensure_local_worker(
        gpu_id=gpu_id,
        skill_dir=skill_dir,
        runs_root=runs_root,
        python=python,
        cache_dir=cache_dir,
    )
    result = submit_local_job(
        queue_dir=queue_dir,
        pid=pid,
        worker_script=skill_dir.resolve() / "local_worker.py",
        payload=payload,
    )
    return result, command


def stop_local_worker(*, runs_root: Path, gpu_id: str) -> bool:
    queue_dir = local_queue_dir(runs_root.resolve(), gpu_id)
    pid = _read_pid(queue_dir / "worker.pid")
    worker_script = Path(__file__).resolve().with_name("local_worker.py")
    if not _worker_running(pid, worker_script):
        return False
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and _worker_running(pid, worker_script):
        time.sleep(0.1)
    return not _worker_running(pid, worker_script)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "stop"))
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--gpu-id", required=True)
    args = parser.parse_args()
    queue_dir = local_queue_dir(args.runs_root.resolve(), args.gpu_id)
    pid = _read_pid(queue_dir / "worker.pid")
    worker_script = Path(__file__).resolve().with_name("local_worker.py")
    if args.action == "status":
        print(
            json.dumps(
                {"running": _worker_running(pid, worker_script), "pid": pid},
                ensure_ascii=False,
            )
        )
        return
    stopped = stop_local_worker(runs_root=args.runs_root, gpu_id=args.gpu_id)
    print(json.dumps({"stopped": stopped, "pid": pid}, ensure_ascii=False))


if __name__ == "__main__":
    main()
