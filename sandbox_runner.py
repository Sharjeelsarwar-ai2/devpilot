#!/usr/bin/env python3
"""Controlled subprocess runner for DevPilot.

This is a resource/time-bounded execution layer, not a kernel-isolated sandbox.
It is designed for Streamlit Cloud prototypes: it scrubs secrets, uses a
separate process group, captures bounded output, and always tears down children.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MAX_OUTPUT = 16_000


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def safe_path(workspace: Path, relative: str) -> Path:
    root = workspace.resolve()
    target = (workspace / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Target path escapes the sandbox workspace.")
    return target


def scrub_env(workspace: Path) -> dict[str, str]:
    """Pass only a minimal environment to child processes."""
    path_value = os.environ.get("PATH", "")
    env = {
        "PATH": path_value,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "HOME": str(workspace / ".sandbox_home"),
        "TMPDIR": str(workspace / ".sandbox_tmp"),
    }
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def terminate_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass
    deadline = time.time() + 2
    while process.poll() is None and time.time() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def run_process(
    command: list[str], workspace: Path, timeout: int, mode: str
) -> dict[str, Any]:
    process = None
    started = time.time()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            env=scrub_env(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            bufsize=1,
        )
        output, _ = process.communicate(timeout=timeout)
        return {
            "ok": process.returncode == 0,
            "mode": mode,
            "returncode": process.returncode,
            "duration_s": round(time.time() - started, 2),
            "output": (output or "")[-MAX_OUTPUT:],
        }
    except subprocess.TimeoutExpired:
        if process is not None:
            terminate_tree(process)
        return {
            "ok": False,
            "mode": mode,
            "error_type": "timeout",
            "error": f"Process timed out after {timeout}s.",
        }
    except OSError as exc:
        if process is not None:
            terminate_tree(process)
        return {
            "ok": False,
            "mode": mode,
            "error_type": "infrastructure",
            "error": str(exc),
        }
    except Exception as exc:
        if process is not None:
            terminate_tree(process)
        return {
            "ok": False,
            "mode": mode,
            "error_type": "runner",
            "error": str(exc),
        }


def compile_target(workspace: Path, target: str, timeout: int) -> dict[str, Any]:
    path = safe_path(workspace, target)
    if not path.exists() or path.suffix != ".py":
        return {"ok": False, "error": "compile requires an existing .py file."}
    return run_process(
        [sys.executable, "-m", "py_compile", str(path)],
        workspace,
        timeout,
        "sandbox_py_compile",
    )


def pytest_project(workspace: Path, timeout: int) -> dict[str, Any]:
    tests = list(workspace.rglob("test_*.py")) + list(workspace.rglob("*_test.py"))
    if not tests:
        return {
            "ok": True,
            "mode": "sandbox_pytest",
            "tests_found": 0,
            "message": "No pytest files were present.",
        }
    return run_process(
        [sys.executable, "-m", "pytest", "-q"],
        workspace,
        timeout,
        "sandbox_pytest",
    ) | {"tests_found": len(tests)}


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def smoke_streamlit(workspace: Path, target: str, timeout: int) -> dict[str, Any]:
    entry = safe_path(workspace, target)
    if not entry.exists() or entry.suffix != ".py":
        return {
            "ok": False,
            "error": "Streamlit smoke test requires an existing .py entry file.",
        }

    port = free_port()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(entry),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]

    process = None
    started = time.time()
    url = f"http://127.0.0.1:{port}/_stcore/health"
    output_lines: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            env=scrub_env(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            bufsize=1,
        )

        deadline = started + timeout
        while time.time() < deadline:
            if process.stdout is not None:
                line = process.stdout.readline()
                if line:
                    output_lines.append(line.rstrip())
                    if sum(len(x) for x in output_lines) > MAX_OUTPUT:
                        output_lines = output_lines[-80:]

            if process.poll() is not None:
                remaining = ""
                try:
                    remaining, _ = process.communicate(timeout=1)
                except Exception:
                    pass
                if remaining:
                    output_lines.append(remaining)
                return {
                    "ok": False,
                    "mode": "sandbox_streamlit_smoke",
                    "returncode": process.returncode,
                    "error_type": "application_startup",
                    "error": "Streamlit exited before its HTTP health endpoint became ready.",
                    "startup_log": "\n".join(output_lines)[-MAX_OUTPUT:],
                }

            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    status = int(response.status)
                    body = response.read(256).decode("utf-8", errors="ignore")
                    if status == 200:
                        return {
                            "ok": True,
                            "mode": "sandbox_streamlit_smoke",
                            "http_status": status,
                            "health_body": body,
                            "duration_s": round(time.time() - started, 2),
                            "message": "Streamlit HTTP health endpoint responded successfully.",
                        }
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(0.25)

        return {
            "ok": False,
            "mode": "sandbox_streamlit_smoke",
            "error_type": "timeout",
            "error": f"Streamlit HTTP health endpoint was not ready within {timeout}s.",
            "startup_log": "\n".join(output_lines)[-MAX_OUTPUT:],
        }
    except OSError as exc:
        return {
            "ok": False,
            "mode": "sandbox_streamlit_smoke",
            "error_type": "infrastructure",
            "error": str(exc),
        }
    finally:
        if process is not None:
            terminate_tree(process)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["compile", "pytest", "streamlit"])
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--target", default="")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    if not workspace.exists() or not workspace.is_dir():
        emit({"ok": False, "error_type": "infrastructure", "error": "Workspace does not exist."})
        return 2

    try:
        if args.mode == "compile":
            result = compile_target(workspace, args.target, min(args.timeout, 30))
        elif args.mode == "pytest":
            result = pytest_project(workspace, min(args.timeout, 60))
        else:
            result = smoke_streamlit(workspace, args.target, min(args.timeout, 30))
        emit(result)
        return 0 if result.get("ok") else 1
    except Exception as exc:
        emit({"ok": False, "error_type": "runner", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
