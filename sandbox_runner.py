from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def safe_path(workspace: Path, relative: str) -> Path:
    root = workspace.resolve()
    target = (workspace / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Path escapes workspace.")
    return target


def compile_file(workspace: Path, target: str) -> dict:
    path = safe_path(workspace, target)
    if not path.exists() or path.suffix != ".py":
        return {"ok": False, "mode": "compile", "error": "Invalid Python file."}

    source = path.read_text(encoding="utf-8")
    try:
        compile(source, target, "exec")
        return {"ok": True, "mode": "compile"}
    except SyntaxError as exc:
        return {
            "ok": False,
            "mode": "compile",
            "error": exc.msg,
            "line": exc.lineno,
        }


def run_pytest(workspace: Path) -> dict:
    tests = list(workspace.rglob("test_*.py")) + list(workspace.rglob("*_test.py"))
    if not tests:
        return {
            "ok": True,
            "mode": "pytest",
            "message": "No tests found.",
        }

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=40,
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONUNBUFFERED": "1",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            },
        )
    except Exception as exc:
        return {
            "ok": False,
            "mode": "pytest",
            "error_type": "infrastructure",
            "error": str(exc),
        }

    return {
        "ok": proc.returncode == 0,
        "mode": "pytest",
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-5000:],
        "stderr": (proc.stderr or "")[-5000:],
    }


def smoke_streamlit(workspace: Path, target: str) -> dict:
    entry = safe_path(workspace, target)
    if not entry.exists():
        return {"ok": False, "mode": "sandbox_streamlit_smoke", "error": "Entry file not found."}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }

    proc = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                target,
                "--server.headless",
                "true",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(port),
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        deadline = time.time() + 12

        while time.time() < deadline:
            if proc.poll() is not None:
                log = ""
                if proc.stdout:
                    log = proc.stdout.read()[-5000:]
                return {
                    "ok": False,
                    "mode": "sandbox_streamlit_smoke",
                    "returncode": proc.returncode,
                    "error": "Streamlit exited before its HTTP server became ready.",
                    "startup_log": log,
                }

            try:
                import requests

                response = requests.get(
                    f"http://127.0.0.1:{port}",
                    timeout=1,
                )
                if response.status_code == 200:
                    return {
                        "ok": True,
                        "mode": "sandbox_streamlit_smoke",
                        "http_status": 200,
                        "message": "Streamlit HTTP server responded successfully.",
                    }
            except Exception:
                pass

            time.sleep(0.4)

        return {
            "ok": False,
            "mode": "sandbox_streamlit_smoke",
            "error": "Streamlit did not become ready before the timeout.",
        }

    except Exception as exc:
        return {
            "ok": False,
            "mode": "sandbox_streamlit_smoke",
            "error_type": "infrastructure",
            "error": str(exc),
        }

    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compile", "pytest", "streamlit"], required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--target", default="")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()

    try:
        if args.mode == "compile":
            result = compile_file(workspace, args.target)
        elif args.mode == "pytest":
            result = run_pytest(workspace)
        else:
            result = smoke_streamlit(workspace, args.target)

        emit(result)

    except Exception as exc:
        emit({
            "ok": False,
            "error_type": "infrastructure",
            "error": str(exc),
        })
        raise


if __name__ == "__main__":
    main()
