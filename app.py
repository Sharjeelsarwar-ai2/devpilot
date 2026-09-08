import os
import json
import zipfile
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="DevPilot AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_FILE_BYTES = 300_000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TOOL_OUTPUT_CHARS = 12_000
MAX_AGENT_STEPS = 14
SANDBOX_RUNNER = Path(__file__).with_name("sandbox_runner.py")

# -----------------------------
# UI styling
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(105,85,255,.18), transparent 30%),
        radial-gradient(circle at 85% 18%, rgba(0,207,255,.12), transparent 28%),
        radial-gradient(circle at 50% 100%, rgba(160,80,255,.10), transparent 34%),
        #070a12;
    color:#f5f7fb;
}
.block-container { max-width:1450px; padding-top:5.5rem; padding-bottom:3rem; }
[data-testid="stHeader"] { background:transparent !important; box-shadow:none !important; }
[data-testid="stDecoration"] { display:none !important; }
[data-testid="stSidebar"] {
    background:rgba(10,14,25,.74);
    border-right:1px solid rgba(255,255,255,.08);
    backdrop-filter:blur(20px);
}
[data-testid="stSidebar"] > div:first-child { padding-top:1.6rem; }
.glass-card {
    background:linear-gradient(135deg,rgba(255,255,255,.075),rgba(255,255,255,.025));
    border:1px solid rgba(255,255,255,.11);
    box-shadow:0 24px 70px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.05);
    border-radius:22px; padding:1.25rem; backdrop-filter:blur(22px);
}
.hero {
    position:relative; z-index:2; padding:1.8rem; border-radius:28px;
    border:1px solid rgba(255,255,255,.11);
    background:linear-gradient(135deg,rgba(111,90,255,.16),rgba(0,210,255,.06) 55%,rgba(255,255,255,.025));
    box-shadow:0 24px 80px rgba(0,0,0,.3); backdrop-filter:blur(24px); margin-bottom:1rem;
}
.hero-kicker { font-size:.76rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:rgba(255,255,255,.60); margin-bottom:.55rem; }
.hero-title { font-size:clamp(2rem,4vw,3.4rem); font-weight:800; line-height:1.02; margin:0; letter-spacing:-.045em; }
.hero-gradient { background:linear-gradient(90deg,#fff 0%,#bfc9ff 48%,#77dfff 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-sub { color:rgba(255,255,255,.66); margin-top:.72rem; font-size:.96rem; max-width:900px; line-height:1.6; }
.pill-row { margin-top:1rem; }
.pill { display:inline-block; margin-right:.45rem; margin-bottom:.35rem; padding:.38rem .72rem; border-radius:999px; border:1px solid rgba(255,255,255,.10); background:rgba(255,255,255,.045); color:rgba(255,255,255,.72); font-size:.76rem; }
.section-title { font-size:1.02rem; font-weight:700; letter-spacing:-.02em; margin:.1rem 0 .55rem; }
.section-sub { font-size:.82rem; color:rgba(255,255,255,.48); margin-bottom:1rem; }
.metric { text-align:center; padding:.8rem .4rem; }
.metric-value { font-size:1.35rem; font-weight:800; }
.metric-label { font-size:.72rem; color:rgba(255,255,255,.48); margin-top:.2rem; }
.agent-step { display:flex; align-items:center; gap:.75rem; padding:.62rem .7rem; margin:.28rem 0; border-radius:13px; border:1px solid rgba(255,255,255,.07); background:rgba(255,255,255,.028); }
.agent-step.active { background:rgba(118,100,255,.12); border-color:rgba(145,132,255,.24); }
.agent-step.done { background:rgba(70,190,130,.055); }
.agent-step.error { background:rgba(255,90,90,.07); border-color:rgba(255,100,100,.20); }
.agent-step.skipped { opacity:.48; }
.step-dot { width:10px; height:10px; border-radius:50%; flex:0 0 10px; background:rgba(255,255,255,.25); }
.step-dot.active { background:#8e82ff; box-shadow:0 0 16px rgba(142,130,255,.75); }
.step-dot.done { background:#50d296; }
.step-dot.error { background:#ff7777; }
.step-dot.skipped { background:rgba(255,255,255,.16); }
.step-text { font-size:.82rem; color:rgba(255,255,255,.76); }
.step-detail { font-size:.70rem; color:rgba(255,255,255,.40); margin-top:.1rem; }
[data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,.028) !important; border:1px dashed rgba(255,255,255,.16) !important; border-radius:18px !important; }
textarea, input { border-radius:14px !important; }
.stButton > button { border-radius:14px; min-height:46px; font-weight:700; border:1px solid rgba(255,255,255,.11); box-shadow:0 10px 30px rgba(0,0,0,.16); }
footer { visibility:hidden; }
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_STEPS = [
    ("Understand requirement", "Interpret the requested change"),
    ("Inspect project", "Map the codebase and identify relevant files"),
    ("Design solution", "Choose the smallest coherent implementation"),
    ("Implement changes", "Read, edit and create files"),
    ("Run verification", "Sandboxed compile, tests and Streamlit smoke test"),
    ("Fix issues", "Inspect failures and rework the implementation"),
    ("Finalize", "Summarize verified changes"),
]

for key, value in {
    "workspace": None,
    "final_report": "",
    "agent_events": [],
    "agent_step": 0,
    "running": False,
    "run_success": False,
    "verification_passed": False,
    "verification_failed": False,
    "verification_blocked": False,
    "streamlit_verification_required": False,
    "streamlit_smoke_passed": False,
    "step_state": {name: "pending" for name, _ in DEFAULT_STEPS},
    "step_detail": {name: detail for name, detail in DEFAULT_STEPS},
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def safe_workspace_path(workspace: Path, relative_path: str) -> Path:
    candidate = (workspace / relative_path).resolve()
    root = workspace.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path escapes the project workspace.")
    return candidate


def clean_project_root(workspace: Path) -> None:
    entries = [p for p in workspace.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        top = entries[0]
        for child in list(top.iterdir()):
            shutil.move(str(child), str(workspace / child.name))
        top.rmdir()


def extract_project(uploaded_file) -> Path:
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Project ZIP exceeds the 20 MB upload limit.")
    temp_root = Path(tempfile.mkdtemp(prefix="devpilot_"))
    workspace = temp_root / "workspace"
    workspace.mkdir()
    zip_path = temp_root / "project.zip"
    zip_path.write_bytes(raw)

    with zipfile.ZipFile(zip_path, "r") as zf:
        total_uncompressed = 0
        for member in zf.infolist():
            path = Path(member.filename)
            total_uncompressed += member.file_size
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe ZIP path detected.")
            if total_uncompressed > 100 * 1024 * 1024:
                raise ValueError("Uncompressed project exceeds the 100 MB safety limit.")
        zf.extractall(workspace)

    clean_project_root(workspace)
    return workspace


def list_files(workspace: Path) -> Dict[str, Any]:
    files = []
    ignored = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}
    for p in workspace.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(workspace).parts
        if any(part in ignored for part in rel_parts):
            continue
        rel = p.relative_to(workspace).as_posix()
        if rel == ".streamlit/secrets.toml":
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        files.append({"path": rel, "size": size})
    files.sort(key=lambda x: x["path"])
    return {"files": files[:500], "count": len(files)}


def read_file(workspace: Path, path: str) -> Dict[str, Any]:
    target = safe_workspace_path(workspace, path)
    if not target.exists():
        return {"ok": False, "error": f"File not found: {path}"}
    if not target.is_file():
        return {"ok": False, "error": f"Not a file: {path}"}
    size = target.stat().st_size
    if size > MAX_FILE_BYTES:
        return {"ok": False, "error": f"{path} is too large ({size} bytes)."}
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "error": f"{path} is not UTF-8 text."}
    return {"ok": True, "path": path, "content": text}


def write_file(workspace: Path, path: str, content: str) -> Dict[str, Any]:
    target = safe_workspace_path(workspace, path)
    if target.name in {".env", "secrets.toml"} or "secrets" in target.parts:
        return {"ok": False, "error": "Writing secret files is blocked."}
    # Prevent accidental dependency-shadowing files that can hide the real
    # package/environment failure (for example streamlit.py).
    blocked_shadow_modules = {
        "streamlit.py", "groq.py", "pytest.py", "subprocess.py",
        "socket.py", "ssl.py", "json.py", "os.py", "sys.py",
        "pathlib.py", "typing.py", "tempfile.py", "zipfile.py"
    }
    if target.name in blocked_shadow_modules:
        return {
            "ok": False,
            "error": f"Refusing to create dependency-shadowing module: {path}"
        }
    if target.name == "__init__.py" and any(part in {"streamlit", "groq", "pytest"} for part in target.parts[:-1]):
        return {
            "ok": False,
            "error": f"Refusing to create a dependency-shadowing package: {path}"
        }

    blocked_shadow_names = {
        "streamlit.py", "groq.py", "pytest.py", "subprocess.py", "os.py",
        "sys.py", "json.py", "socket.py", "requests.py", "typing.py",
        "pathlib.py", "tempfile.py", "shutil.py", "zipfile.py"
    }
    if target.name in blocked_shadow_names:
        return {"ok": False, "error": f"Refusing to create a module that shadows a runtime dependency: {target.name}"}
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        return {"ok": False, "error": f"Refusing to write more than {MAX_FILE_BYTES} bytes to one file."}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "bytes_written": len(content.encode("utf-8"))}


def sandbox_call(mode: str, workspace: Path, target: str = "", timeout: int = 20) -> Dict[str, Any]:
    if not SANDBOX_RUNNER.exists():
        return {"ok": False, "error": "sandbox_runner.py is missing from the deployment."}

    command = [
        sys.executable,
        str(SANDBOX_RUNNER),
        "--mode", mode,
        "--workspace", str(workspace),
        "--timeout", str(timeout),
    ]
    if target:
        command += ["--target", target]

    try:
        result = subprocess.run(
            command,
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            timeout=timeout + 8,
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONUNBUFFERED": "1",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            },
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"Sandbox runner exceeded its {timeout + 8}s controller timeout.",
            "category": "sandbox_infrastructure",
            "fatal": True,
        }
    except OSError as exc:
        message = str(exc)
        if exc.errno == 11 or "resource temporarily unavailable" in message.lower():
            return {
                "ok": False,
                "error": message,
                "category": "sandbox_infrastructure",
                "fatal": True,
            }
        return {"ok": False, "error": message, "category": "sandbox_infrastructure", "fatal": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "category": "sandbox_infrastructure", "fatal": True}

    output = (result.stdout or "").strip().splitlines()
    payload = None
    if output:
        try:
            payload = json.loads(output[-1])
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        return {
            "ok": False,
            "error": "Sandbox runner returned invalid output.",
            "runner_stdout": (result.stdout or "")[-MAX_TOOL_OUTPUT_CHARS:],
            "runner_stderr": (result.stderr or "")[-MAX_TOOL_OUTPUT_CHARS:],
        }
    if result.returncode != 0 and payload.get("ok", False):
        payload["ok"] = False
    error_text = str(payload.get("error", ""))
    infrastructure_markers = (
        "preexec_fn",
        "permission denied",
        "operation not permitted",
        "resource temporarily unavailable",
        "[errno 11]",
        "cannot start process",
        "no module named streamlit",
    )
    if any(marker in error_text.lower() for marker in infrastructure_markers):
        payload["ok"] = False
        payload["category"] = "sandbox_infrastructure"
        payload["fatal"] = True
    return payload


def run_python(workspace: Path, path: str) -> Dict[str, Any]:
    target = safe_workspace_path(workspace, path)
    if not target.exists() or target.suffix != ".py":
        return {"ok": False, "error": "run_python requires an existing .py file."}
    source = target.read_text(encoding="utf-8", errors="ignore")
    if "import streamlit" in source or "from streamlit" in source or "st." in source:
        result = sandbox_call("compile", workspace, path, timeout=15)
        result["note"] = "Detected Streamlit code; syntax was checked in the sandbox. Use smoke_test_streamlit for runtime verification."
        return result
    return sandbox_call("python", workspace, path, timeout=20)


def run_tests(workspace: Path) -> Dict[str, Any]:
    tests = list(workspace.rglob("test_*.py")) + list(workspace.rglob("*_test.py"))
    if tests:
        return sandbox_call("pytest", workspace, timeout=35)

    py_files = list(workspace.rglob("*.py"))[:100]
    errors = []
    for py in py_files:
        rel = py.relative_to(workspace).as_posix()
        result = sandbox_call("compile", workspace, rel, timeout=10)
        if not result.get("ok"):
            errors.append({"file": rel, "result": result})
    return {
        "ok": not errors,
        "mode": "sandbox_py_compile_batch",
        "files_checked": len(py_files),
        "compile_errors": errors,
    }


def smoke_test_streamlit(workspace: Path, path: str) -> Dict[str, Any]:
    target = safe_workspace_path(workspace, path)
    if not target.exists() or target.suffix != ".py":
        return {"ok": False, "error": "Streamlit smoke test requires an existing .py entry file."}
    return sandbox_call("streamlit", workspace, path, timeout=18)


TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "List project files.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file before editing it.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or replace a UTF-8 text file in the project workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "run_python", "description": "Run a normal Python script inside the controlled sandbox. For Streamlit code it performs a sandboxed syntax check instead.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run pytest in the controlled sandbox when tests exist, otherwise compile-check Python files.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "smoke_test_streamlit", "description": "Launch a Streamlit app in a short-lived controlled sandbox process, verify that localhost HTTP responds, collect startup diagnostics, then terminate it.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative path to the Streamlit entry file, e.g. app.py"}}, "required": ["path"], "additionalProperties": False}}},
]

SYSTEM_PROMPT = """
You are DevPilot, an autonomous software development agent working inside a temporary project workspace.

Your job is to implement the user's requirement, not merely explain it.

Available tools:
- list_files
- read_file
- write_file
- run_python
- run_tests
- smoke_test_streamlit

Execution and safety rules:
1. Start by inspecting the project. Usually call list_files first.
2. Read relevant files before changing them.
3. Make the smallest coherent set of changes needed.
4. Preserve the project's existing architecture and UI style when practical.
5. Never invent unseen file contents when replacing existing files.
6. Never write secrets, credentials, .env files or secret configuration.
7. Never delete the whole project.
8. Use run_tests after implementation when possible.
9. If the project is a Streamlit app, use smoke_test_streamlit on its entry point before claiming the app starts correctly.
10. If verification fails, inspect the diagnostic output, fix the relevant code, and verify again.
11. Remember that the execution tools run inside a controlled, resource-limited subprocess environment. Do not attempt to escape it or weaken the controls.
12. Never claim that a change works unless verification actually passed.
13. Keep working until the requirement is implemented and verified or there is a genuine blocker.
14. Final response must contain: files changed, changes made, verification performed, remaining issues.
15. Never create fake/shim modules to bypass a missing dependency. In particular, never create files such as streamlit.py, groq.py, pytest.py, or other modules that shadow installed libraries.
16. A tool call returning ok=false is a real failure. Do not describe it as successful; diagnose it and retry or report the blocker.

write_file replaces the entire file. Read an existing file first.
"""


def set_stage(stage: str, state: str = "active", detail: str | None = None) -> None:
    if stage not in st.session_state.step_state:
        return
    st.session_state.step_state[stage] = state
    if detail:
        st.session_state.step_detail[stage] = detail


def infer_stage(name: str) -> str:
    if name in {"list_files", "read_file"}:
        return "Inspect project"
    if name == "write_file":
        return "Implement changes"
    if name in {"run_python", "run_tests", "smoke_test_streamlit"}:
        return "Run verification"
    return "Design solution"


def execute_tool(workspace: Path, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    stage = infer_stage(name)
    if stage not in {"Inspect project", "Design solution"}:
        set_stage(stage, "active")

    if name == "list_files": result = list_files(workspace)
    elif name == "read_file": result = read_file(workspace, args["path"])
    elif name == "write_file": result = write_file(workspace, args["path"], args["content"])
    elif name == "run_python": result = run_python(workspace, args["path"])
    elif name == "run_tests": result = run_tests(workspace)
    elif name == "smoke_test_streamlit": result = smoke_test_streamlit(workspace, args["path"])
    else: result = {"ok": False, "error": f"Unknown tool: {name}"}

    detail = name.replace("_", " ").title()
    if "path" in args:
        detail += f" · {args['path']}"
    if result.get("ok") is False and result.get("error"):
        detail += f" · {str(result['error'])[:100]}"

    st.session_state.agent_events.append({"tool": name, "detail": detail, "ok": bool(result.get("ok", True)), "result": result})

    if name in {"list_files", "read_file"}:
        set_stage("Inspect project", "done" if result.get("ok", True) else "error", detail)
    elif name == "write_file":
        if result.get("ok"):
            set_stage("Design solution", "done", "Implementation plan executed")
            set_stage("Implement changes", "done", detail)
        else:
            set_stage("Implement changes", "error", detail)
    elif name in {"run_python", "run_tests", "smoke_test_streamlit"}:
        st.session_state.verification_seen = True
        st.session_state.verification_attempts = st.session_state.get("verification_attempts", 0) + 1

        if name == "smoke_test_streamlit":
            st.session_state.streamlit_verification_required = True
            st.session_state.streamlit_smoke_passed = bool(result.get("ok"))

        if result.get("fatal") or result.get("category") == "sandbox_infrastructure":
            st.session_state.verification_passed = False
            st.session_state.verification_failed = True
            st.session_state.verification_blocked = True
            set_stage("Run verification", "error", "Sandbox infrastructure failure")
            set_stage("Fix issues", "error", "Cannot safely repair an execution-environment failure")
            return result

        # A general verification result cannot overwrite a failed required
        # Streamlit smoke test. The required smoke test must pass separately.
        st.session_state.verification_failed = not bool(result.get("ok"))
        st.session_state.verification_passed = (
            bool(result.get("ok"))
            and (not st.session_state.streamlit_verification_required
                 or st.session_state.streamlit_smoke_passed)
        )

        if st.session_state.verification_passed:
            set_stage("Run verification", "done", "All required sandbox verification passed")
            if st.session_state.step_state["Fix issues"] == "active":
                set_stage("Fix issues", "done", "Verification passed after fixes")
            elif st.session_state.step_state["Fix issues"] == "pending":
                set_stage("Fix issues", "skipped", "No verification failure")
        else:
            set_stage("Run verification", "error", "Sandbox verification failed")
            set_stage("Fix issues", "active", "Analyzing verification failure")
    return result


def render_progress(timeline, progress, status) -> None:
    states = st.session_state.step_state
    completed = sum(v in {"done", "skipped"} for v in states.values())
    pct = completed / len(states)
    progress.progress(pct, text=f"Agent progress · {int(pct * 100)}%")

    html = []
    for name, _ in DEFAULT_STEPS:
        state = states.get(name, "pending")
        icon = {"pending": "○", "active": "◉", "done": "✓", "error": "!", "skipped": "—"}[state]
        detail = st.session_state.step_detail.get(name, "")
        html.append(
            f'<div class="agent-step {state}"><div class="step-dot {state}"></div>'
            f'<div><div class="step-text">{icon} {name}</div><div class="step-detail">{detail}</div></div></div>'
        )
    timeline.markdown("".join(html), unsafe_allow_html=True)
    active = next((n for n, s in states.items() if s == "active"), None)
    if active:
        status.markdown(f"**Currently:** {active}")
    elif all(s in {"done", "skipped"} for s in states.values()):
        status.markdown("**Status:** Development run verified")
    else:
        status.markdown("**Status:** Verification stopped with an issue")


def run_agent(workspace: Path, requirement: str, client: Groq, model: str, progress_ui) -> str:
    st.session_state.agent_events = []
    st.session_state.agent_step = 0
    st.session_state.verification_passed = False
    st.session_state.verification_seen = False
    st.session_state.step_state = {name: "pending" for name, _ in DEFAULT_STEPS}
    st.session_state.step_detail = {name: detail for name, detail in DEFAULT_STEPS}
    st.session_state.verification_passed = False
    st.session_state.verification_failed = False
    st.session_state.verification_attempts = 0
    set_stage("Understand requirement", "active", "Parsing requested change")
    render_progress(*progress_ui)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Implement this requirement in the uploaded project:\n\n{requirement}\n\nBegin by inspecting the project."},
    ]

    response_text = ""
    for step in range(1, MAX_AGENT_STEPS + 1):
        if st.session_state.get("verification_blocked"):
            set_stage("Finalize", "error", "Execution environment blocked verification")
            st.session_state.run_success = False
            render_progress(*progress_ui)
            return (
                "Development stopped because the sandbox execution environment is unavailable. "
                "The agent will not modify code to work around an infrastructure failure."
            )

        st.session_state.agent_step = step
        set_stage("Understand requirement", "done", "Requirement loaded")
        states = st.session_state.step_state
        if not any(v == "active" for v in states.values()):
            if states["Inspect project"] == "pending": set_stage("Inspect project", "active", "Inspecting project structure")
            elif states["Design solution"] == "pending": set_stage("Design solution", "active", "Planning implementation")
            elif states["Implement changes"] == "pending": set_stage("Implement changes", "active", "Applying code changes")
            elif states["Run verification"] == "pending": set_stage("Run verification", "active", "Running sandbox verification")
            elif states["Fix issues"] == "pending": set_stage("Fix issues", "active", "Analyzing failures")
        render_progress(*progress_ui)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )
        message = response.choices[0].message
        assistant: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]
        messages.append(assistant)

        if not message.tool_calls:
            # Do not allow a green final state without successful verification.
            if not st.session_state.verification_passed:
                if st.session_state.get("verification_attempts", 0) >= 3:
                    set_stage("Finalize", "error", "Verification failed three times; stopping safely")
                    st.session_state.run_success = False
                    render_progress(*progress_ui)
                    return (
                        "The agent stopped after three unsuccessful verification attempts. "
                        "Review the detailed verification diagnostics before retrying."
                    )
                messages.append({
                    "role": "user",
                    "content": (
                        "Do not finalize yet. Verification has not passed. "
                        "Use the appropriate verification tool now. If verification "
                        "failed, inspect the diagnostics, fix the code, and verify again "
                        "before giving the final report."
                    ),
                })
                set_stage("Run verification", "active", "Verification required before finalizing")
                render_progress(*progress_ui)
                continue

            response_text = message.content or "Agent finished without a final report."
            if not st.session_state.verification_passed:
                set_stage("Finalize", "error", "Final report withheld because verification has not passed")
                st.session_state.run_success = False
                render_progress(*progress_ui)
                return (
                    "The agent stopped without a verified result. No successful finalization was recorded.\n\n"
                    + response_text
                )
            st.session_state.run_success = True
            for stage in ["Design solution", "Implement changes", "Run verification"]:
                if st.session_state.step_state[stage] == "active":
                    set_stage(stage, "done", "Completed")
            if st.session_state.step_state["Fix issues"] == "pending":
                set_stage("Fix issues", "skipped", "No verification failures")
            elif st.session_state.step_state["Fix issues"] == "active":
                set_stage("Fix issues", "done", "Verification passed after fixes")
            set_stage("Finalize", "active", "Generating final report")
            render_progress(*progress_ui)
            set_stage("Finalize", "done", "Final report generated")
            render_progress(*progress_ui)
            break

        for tool_call in message.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
                result = execute_tool(workspace, tool_call.function.name, args)
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, ensure_ascii=False)})
            render_progress(*progress_ui)
            if result.get("fatal") or result.get("category") == "sandbox_infrastructure":
                st.session_state.run_success = False
                set_stage("Finalize", "error", "Execution environment blocked verification")
                render_progress(*progress_ui)
                return (
                    "Development blocked: the sandbox execution environment failed before the project could be verified. "
                    f"Diagnostic: {result.get('error', 'unknown sandbox error')}\n\n"
                    "No code changes should be made to work around this environment problem. "
                    "Check that the deployed sandbox_runner.py matches the current version and that the runtime permits subprocess creation."
                )
    else:
        response_text = "The agent reached its maximum step limit before producing a verified final report. Review the activity log and verification results."
        st.session_state.run_success = False
        set_stage("Finalize", "error", "Step limit reached")
        render_progress(*progress_ui)
    return response_text


with st.sidebar:
    st.markdown("<div class='section-title'>✦ DevPilot AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Autonomous software development workspace</div>", unsafe_allow_html=True)
    api_key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    model = st.text_input("Model", value=DEFAULT_MODEL)
    st.markdown("<div class='glass-card'><div class='section-title'>Execution layer</div><div class='section-sub'>Uploaded code is tested through a controlled subprocess runner with CPU, memory, file-size and timeout limits, plus a scrubbed environment.</div><span class='pill'>Sandbox runner</span><span class='pill'>Streamlit smoke test</span><span class='pill'>Resource limits</span></div>", unsafe_allow_html=True)
    st.caption("For Streamlit Cloud, put GROQ_API_KEY in App settings → Secrets.")

st.markdown(
    """
<div class="hero">
  <div class="hero-kicker">AI SOFTWARE ENGINEERING AGENT</div>
  <div class="hero-title">Build. Test. <span class="hero-gradient">Fix.</span></div>
  <div class="hero-sub">Upload a project and let the agent inspect the codebase, implement the requirement, run sandboxed verification, recover from failures, and return a development report.</div>
  <div class="pill-row"><span class="pill">Groq LLM</span><span class="pill">Tool calling</span><span class="pill">Sandboxed execution</span><span class="pill">Streamlit smoke tests</span><span class="pill">Streamlit Cloud ready</span></div>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1.25, 1], gap="large")
with left:
    st.markdown("<div class='glass-card'><div class='section-title'>1 · Project</div><div class='section-sub'>Upload a ZIP of the application you want the agent to modify.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Project ZIP", type=["zip"], label_visibility="collapsed")
    if uploaded: st.success(f"Ready · {uploaded.name}")
    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown("<div class='glass-card'><div class='section-title'>2 · Requirement</div><div class='section-sub'>Describe the outcome, not the implementation details.</div>", unsafe_allow_html=True)
    requirement = st.text_area("Requirement", placeholder="Example: Add authentication with session state, environment-based secrets and logout support.", height=145, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

run_clicked = st.button("✦  Run Development Agent", type="primary", use_container_width=True)

if run_clicked:
    if not api_key:
        st.error("Add your Groq API key in the sidebar or Streamlit Cloud Secrets.")
        st.stop()
    if uploaded is None:
        st.error("Upload a project ZIP first.")
        st.stop()
    if not requirement.strip():
        st.error("Describe the software requirement first.")
        st.stop()

    try:
        workspace = extract_project(uploaded)
        st.session_state.workspace = str(workspace)
        st.session_state.final_report = ""
        st.markdown("### Agent execution")
        progress = st.empty(); status = st.empty(); timeline = st.empty()
        client = Groq(api_key=api_key)
        with st.spinner("Agent is reasoning and executing tools..."):
            final_report = run_agent(workspace, requirement.strip(), client, model.strip() or DEFAULT_MODEL, (timeline, progress, status))
        st.session_state.final_report = final_report
        if st.session_state.verification_passed and not st.session_state.verification_failed:
            st.success("Development run completed and verified.")
        else:
            st.warning("Development run completed with an unresolved verification issue.")
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP archive.")
    except Exception as exc:
        st.exception(exc)

if st.session_state.final_report:
    st.divider()
    events = st.session_state.agent_events
    successful = sum(1 for e in events if e["ok"])
    failures = len(events) - successful
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='glass-card'><div class='metric'><div class='metric-value'>{len(events)}</div><div class='metric-label'>Agent actions</div></div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='glass-card'><div class='metric'><div class='metric-value'>{successful}</div><div class='metric-label'>Successful actions</div></div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='glass-card'><div class='metric'><div class='metric-value'>{failures}</div><div class='metric-label'>Failed actions</div></div></div>", unsafe_allow_html=True)
    st.markdown("### Development report")
    st.markdown(st.session_state.final_report)

    workspace_path = Path(st.session_state.workspace)
    if workspace_path.exists():
        st.markdown("### Modified project")
        rows = []
        for p in sorted(workspace_path.rglob("*")):
            if p.is_file():
                try: rows.append({"file": p.relative_to(workspace_path).as_posix(), "size": p.stat().st_size})
                except OSError: pass
        st.dataframe(rows, use_container_width=True, hide_index=True)

        zip_output = workspace_path.parent / "modified_project.zip"
        with zipfile.ZipFile(zip_output, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in workspace_path.rglob("*"):
                if p.is_file() and p.name not in {"modified_project.zip"}:
                    zf.write(p, p.relative_to(workspace_path))
        with open(zip_output, "rb") as f:
            st.download_button("Download modified project", data=f, file_name="modified_project.zip", mime="application/zip", use_container_width=True)

    with st.expander("View detailed agent activity"):
        for i, event in enumerate(events, 1):
            icon = "✅" if event["ok"] else "❌"
            st.markdown(f"**{icon} {i}. {event['detail']}**")
            st.json(event["result"])
