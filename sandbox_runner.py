#!/usr/bin/env python3
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None

MAX_OUTPUT = 12000


def safe_path(workspace: Path, relative: str) -> Path:
    root = workspace.resolve()
    target = (workspace / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Target path escapes the sandbox workspace.")
    return target


def limits(cpu_seconds: int, memory_mb: int, file_mb: int):
    if resource is None:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except Exception:
        pass
    try:
        memory = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    except Exception:
        pass
    try:
        size = file_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except Exception:
        pass


def kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        time.sleep(0.5)
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def hardened_env(workspace: Path) -> dict:
    # Give the child a minimal, non-secret environment.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "HOME": str(workspace / ".sandbox_home"),
        "TMPDIR": str(workspace / ".sandbox_tmp"),
    }
    Path(env["HOME"]).mkdir(exist_ok=True)
    Path(env["TMPDIR"]).mkdir(exist_ok=True)
    # Do not expose common LLM/cloud credentials or Streamlit secrets.
    return env


def start_process(command, workspace: Path, cpu=20, memory_mb=768, file_mb=10):
    env = hardened_env(workspace)
    return subprocess.Popen(
        command,
        cwd=str(workspace),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        preexec_fn=(lambda: limits(cpu, memory_mb, file_mb)) if os.name == "posix" else None,
    )


def tail_output(process: subprocess.Popen, limit=MAX_OUTPUT) -> str:
    try:
        output, _ = process.communicate(timeout=1)
        return (output or "")[-limit:]
    except Exception:
        return ""


def run_python(workspace: Path, target: Path, timeout: int):
    process = None
    try:
        process = start_process([sys.executable, "-I", str(target)], workspace)
        output, _ = process.communicate(timeout=timeout)
        return {
            "ok": process.returncode == 0,
            "mode": "sandbox_python",
            "returncode": process.returncode,
            "output": (output or "")[-MAX_OUTPUT:],
        }
    except subprocess.TimeoutExpired:
        if process:
            kill_process_tree(process)
        return {"ok": False, "mode": "sandbox_python", "error": f"Timed out after {timeout}s."}
    except Exception as exc:
        if process:
            kill_process_tree(process)
        return {"ok": False, "mode": "sandbox_python", "error": str(exc)}


def compile_python(workspace: Path, target: Path, timeout: int):
    process = None
    try:
        process = start_process(
            [sys.executable, "-I", "-m", "py_compile", str(target)],
            workspace,
            cpu=10,
            memory_mb=512,
            file_mb=5,
        )
        output, _ = process.communicate(timeout=timeout)
        return {
            "ok": process.returncode == 0,
            "mode": "sandbox_py_compile",
            "returncode": process.returncode,
            "output": (output or "")[-MAX_OUTPUT:],
        }
    except subprocess.TimeoutExpired:
        if process:
            kill_process_tree(process)
        return {"ok": False, "mode": "sandbox_py_compile", "error": f"Timed out after {timeout}s."}
    except Exception as exc:
        if process:
            kill_process_tree(process)
        return {"ok": False, "mode": "sandbox_py_compile", "error": str(exc)}


def run_pytest(workspace: Path, timeout: int):
    process = None
    try:
        process = start_process(
            [sys.executable, "-I", "-m", "pytest", "-q"],
            workspace,
            cpu=30,
            memory_mb=768,
            file_mb=15,
        )
        output, _ = process.communicate(timeout=timeout)
        return {
            "ok": process.returncode == 0,
            "mode": "sandbox_pytest",
            "returncode": process.returncode,
            "output": (output or "")[-MAX_OUTPUT:],
        }
    except subprocess.TimeoutExpired:
        if process:
            kill_process_tree(process)
        return {"ok": False, "mode": "sandbox_pytest", "error": f"Timed out after {timeout}s."}
    except Exception as exc:
        if process:
            kill_process_tree(process)
        return {"ok": False, "mode": "sandbox_pytest", "error": str(exc)}


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    try:
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def smoke_streamlit(workspace: Path, target: Path, timeout: int):
    port = find_free_port()
    command = [
        sys.executable,
        "-I",
        "-m",
        "streamlit",
        "run",
        str(target),
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
    start = time.time()
    try:
        process = start_process(command, workspace, cpu=30, memory_mb=768, file_mb=20)
        deadline = start + timeout
        url = f"http://127.0.0.1:{port}/"

        while time.time() < deadline:
            if process.poll() is not None:
                output = tail_output(process)
                return {
                    "ok": False,
                    "mode": "sandbox_streamlit_smoke",
                    "returncode": process.returncode,
                    "error": "Streamlit exited before its HTTP server became ready.",
                    "startup_log": output,
                }
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    body = response.read(2000).decode("utf-8", errors="ignore")
                    return {
                        "ok": 200 <= response.status < 400,
                        "mode": "sandbox_streamlit_smoke",
                        "status_code": response.status,
                        "startup_seconds": round(time.time() - start, 2),
                        "contains_streamlit": "streamlit" in body.lower(),
                        "message": "Streamlit started and returned an HTTP response inside the sandbox runner.",
                    }
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(0.3)

        return {
            "ok": False,
            "mode": "sandbox_streamlit_smoke",
            "error": f"Timed out after {timeout}s waiting for HTTP response.",
            "startup_log": tail_output(process),
        }
    finally:
        if process:
            kill_process_tree(process)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["python", "compile", "pytest", "streamlit"])
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--target", default="")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        raise ValueError("Sandbox workspace does not exist.")

    if args.mode in {"python", "compile", "streamlit"}:
        target = safe_path(workspace, args.target)
        if not target.is_file():
            raise ValueError("Target file does not exist.")

    if args.mode == "python":
        result = run_python(workspace, target, args.timeout)
    elif args.mode == "compile":
        result = compile_python(workspace, target, args.timeout)
    elif args.mode == "pytest":
        result = run_pytest(workspace, args.timeout)
    else:
        result = smoke_streamlit(workspace, target, args.timeout)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(1)
