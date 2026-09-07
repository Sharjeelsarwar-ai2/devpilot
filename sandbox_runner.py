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

try:
    import resource
except ImportError:
    resource = None

MAX_OUTPUT = 12000
DEFAULT_TIMEOUT = 20


def scrubbed_env(workspace: Path) -> dict:
    # Do not pass application/API secrets to uploaded code.
    allowed = {}
    for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT"):
        if key in os.environ:
            allowed[key] = os.environ[key]
    allowed["PYTHONUNBUFFERED"] = "1"
    allowed["PYTHONNOUSERSITE"] = "1"
    allowed["DEV_PILOT_WORKSPACE"] = str(workspace)
    return allowed


def apply_limits(cpu_seconds: int = 18, max_file_bytes: int = 8 * 1024 * 1024):
    if resource is None:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))
    except Exception:
        pass
    # Keep child/process creation bounded where supported.
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except Exception:
        pass


def cleanup_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass


def run_command(command, workspace: Path, timeout: int):
    preexec = None
    if os.name == "posix":
        def _limits():
            os.setsid()
            apply_limits(cpu_seconds=max(8, timeout - 2))
        preexec = _limits

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=scrubbed_env(workspace),
            start_new_session=(os.name == "posix"),
            preexec_fn=preexec if os.name == "posix" else None,
        )
    except Exception as exc:
        return {"ok": False, "returncode": -1, "error": str(exc)}

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout[-MAX_OUTPUT:],
            "stderr": stderr[-MAX_OUTPUT:],
        }
    except subprocess.TimeoutExpired:
        cleanup_process(proc)
        return {
            "ok": False,
            "returncode": -9,
            "error": f"Process exceeded the {timeout}s timeout and was terminated.",
        }
    finally:
        cleanup_process(proc)


def compile_file(workspace: Path, target: str, timeout: int):
    path = (workspace / target).resolve()
    root = workspace.resolve()
    if path != root and root not in path.parents:
        return {"ok": False, "error": "Target escapes workspace."}
    if not path.exists() or path.suffix != ".py":
        return {"ok": False, "error": f"Python file not found: {target}"}
    result = run_command([sys.executable, "-m", "py_compile", str(path)], workspace, timeout)
    result["mode"] = "compile"
    return result


def run_python(workspace: Path, target: str, timeout: int):
    path = (workspace / target).resolve()
    root = workspace.resolve()
    if path != root and root not in path.parents:
        return {"ok": False, "error": "Target escapes workspace."}
    if not path.exists() or path.suffix != ".py":
        return {"ok": False, "error": f"Python file not found: {target}"}
    result = run_command([sys.executable, str(path)], workspace, timeout)
    result["mode"] = "python"
    return result


def run_pytest(workspace: Path, timeout: int):
    result = run_command([sys.executable, "-m", "pytest", "-q"], workspace, timeout)
    result["mode"] = "pytest"
    return result


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def http_ok(url: str) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            return response.status
    except Exception:
        return None


def run_streamlit(workspace: Path, target: str, timeout: int):
    port = free_port()
    entry = (workspace / target).resolve()
    root = workspace.resolve()
    if entry != root and root not in entry.parents:
        return {"ok": False, "error": "Target escapes workspace."}
    if not entry.exists() or entry.suffix != ".py":
        return {"ok": False, "error": f"Streamlit entry file not found: {target}"}

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(entry),
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
    ]

    preexec = None
    if os.name == "posix":
        def _limits():
            os.setsid()
            apply_limits(cpu_seconds=max(10, timeout - 2), max_file_bytes=8 * 1024 * 1024)
        preexec = _limits

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=scrubbed_env(workspace),
            start_new_session=(os.name == "posix"),
            preexec_fn=preexec if os.name == "posix" else None,
        )
    except Exception as exc:
        return {"ok": False, "returncode": -1, "error": str(exc), "mode": "sandbox_streamlit_smoke"}

    deadline = time.monotonic() + timeout
    status = None
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate(timeout=2)
                return {
                    "ok": False,
                    "mode": "sandbox_streamlit_smoke",
                    "returncode": proc.returncode,
                    "error": "Streamlit exited before its HTTP server became ready.",
                    "startup_log": (stdout + "\n" + stderr)[-MAX_OUTPUT:],
                    "hint": (
                        "The sandbox uses the same Python interpreter as the main app. "
                        "If this says 'No module named streamlit', add streamlit to the "
                        "main deployment requirements."
                    ),
                }
            status = http_ok(f"http://127.0.0.1:{port}/")
            if status is not None:
                return {
                    "ok": status == 200,
                    "mode": "sandbox_streamlit_smoke",
                    "returncode": None,
                    "http_status": status,
                    "message": "Streamlit HTTP server responded successfully.",
                }
            time.sleep(0.35)

        stdout, stderr = proc.communicate(timeout=2)
        return {
            "ok": False,
            "mode": "sandbox_streamlit_smoke",
            "returncode": proc.returncode,
            "error": "Streamlit HTTP server did not become ready before the timeout.",
            "startup_log": (stdout + "\n" + stderr)[-MAX_OUTPUT:],
        }
    finally:
        cleanup_process(proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["python", "compile", "pytest", "streamlit"])
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--target", default="")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if args.mode == "compile":
        result = compile_file(workspace, args.target, args.timeout)
    elif args.mode == "python":
        result = run_python(workspace, args.target, args.timeout)
    elif args.mode == "pytest":
        result = run_pytest(workspace, args.timeout)
    else:
        result = run_streamlit(workspace, args.target, args.timeout)

    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
