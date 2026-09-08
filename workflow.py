from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from groq import Groq
try:
    from groq import BadRequestError, RateLimitError
except ImportError:  # pragma: no cover
    BadRequestError = Exception
    RateLimitError = Exception


DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_FILE_BYTES = 300_000
MAX_READ_LINES = 140
MAX_PATCH_BYTES = 20_000
MAX_CREATE_BYTES = 18_000
MAX_AGENT_TURNS = 8
MAX_REPAIR_ATTEMPTS = 2
MAX_RATE_LIMIT_RETRIES = 1
MAX_LLM_OUTPUT = 1500
MAX_TOOL_CONTEXT_CHARS = 7_000
MAX_ERROR_CONTEXT_CHARS = 6_000
GENERATED_TEST_PATH = "tests/test_devpilot_requirements.py"
SANDBOX_RUNNER = Path(__file__).with_name("sandbox_runner.py")

STAGES = [
    ("requirements", "Requirements", "Turn the request into explicit acceptance criteria."),
    ("inspection", "Inspection", "Map the codebase and identify relevant files."),
    ("design", "Solution design", "Choose a minimal, coherent implementation strategy."),
    ("implementation", "Implementation", "Apply localized code changes with patches."),
    ("test_generation", "Test generation", "Create requirement-level functional tests."),
    ("testing", "Testing", "Run deterministic tests and syntax checks."),
    ("failure_analysis", "Failure analysis", "Understand failed tests without changing code."),
    ("error_handling", "Error handling", "Classify the failure and decide whether repair is safe."),
    ("repair", "Repair", "Fix only the code implicated by the failure."),
    ("sandbox", "Sandbox verification", "Launch the app and verify its HTTP health endpoint."),
    ("final_report", "Final report", "Summarize verified changes and remaining issues."),
]

STAGE_KEYS = [x[0] for x in STAGES]


class FatalAgentError(RuntimeError):
    pass


class InfrastructureFailure(FatalAgentError):
    pass


class AgentRateLimitError(FatalAgentError):
    pass


class WorkflowState:
    def __init__(self) -> None:
        self.stage_state = {key: "pending" for key in STAGE_KEYS}
        self.stage_detail = {key: detail for key, _, detail in STAGES}
        self.events: List[Dict[str, Any]] = []
        self.workspace: Optional[Path] = None
        self.requirements: str = ""
        self.requirement_plan: str = ""
        self.inspection: str = ""
        self.design: str = ""
        self.test_path: Optional[str] = None
        self.verification: Dict[str, Any] = {
            "syntax_ok": False,
            "tests_ok": False,
            "smoke_ok": False,
            "infrastructure_error": False,
            "test_failure": False,
        }
        self.repair_attempts = 0
        self.final_report = ""
        self.aborted = False
        self.abort_reason = ""


def set_stage(state: WorkflowState, stage: str, value: str, detail: Optional[str] = None) -> None:
    state.stage_state[stage] = value
    if detail:
        state.stage_detail[stage] = detail


def stage_index(stage: str) -> int:
    return STAGE_KEYS.index(stage)


def callback_event(state: WorkflowState, callback: Optional[Callable[[WorkflowState], None]]) -> None:
    if callback:
        callback(state)


def log_event(state: WorkflowState, tool: str, ok: bool, detail: str, result: Any) -> None:
    state.events.append({"tool": tool, "ok": ok, "detail": detail, "result": result})


def safe_workspace_path(workspace: Path, relative_path: str) -> Path:
    candidate = (workspace / relative_path).resolve()
    root = workspace.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path escapes project workspace.")
    return candidate


def extract_project(uploaded_file) -> Path:
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Project ZIP exceeds the 20 MB upload limit.")

    temp_root = Path(tempfile.mkdtemp(prefix="devpilot_workflow_"))
    workspace = temp_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    archive = temp_root / "project.zip"
    archive.write_bytes(raw)

    with zipfile.ZipFile(archive, "r") as zf:
        total = 0
        for member in zf.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe ZIP path detected.")
            total += member.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Uncompressed project exceeds the 100 MB safety limit.")
        zf.extractall(workspace)

    entries = list(workspace.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        top = entries[0]
        for child in list(top.iterdir()):
            shutil.move(str(child), str(workspace / child.name))
        top.rmdir()

    return workspace


def list_files(workspace: Path) -> Dict[str, Any]:
    ignored = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}
    files = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(workspace).parts
        if any(part in ignored for part in parts):
            continue
        rel = path.relative_to(workspace).as_posix()
        if rel == ".streamlit/secrets.toml":
            continue
        try:
            files.append({"path": rel, "size": path.stat().st_size})
        except OSError:
            pass
    files.sort(key=lambda x: x["path"])
    return {"ok": True, "files": files[:500], "count": len(files)}


def search_file(workspace: Path, path: str, query: str) -> Dict[str, Any]:
    target = safe_workspace_path(workspace, path)
    if not target.exists() or not target.is_file():
        return {"ok": False, "error": f"File not found: {path}"}
    if target.stat().st_size > MAX_FILE_BYTES:
        return {"ok": False, "error": f"{path} is too large to search."}
    text = target.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    hits = []
    q = query.lower()
    for i, line in enumerate(lines, 1):
        if q in line.lower():
            lo = max(1, i - 2)
            hi = min(len(lines), i + 2)
            context = "\n".join(f"{n}: {lines[n-1]}" for n in range(lo, hi + 1))
            hits.append({"line": i, "context": context})
            if len(hits) >= 20:
                break
    return {"ok": True, "path": path, "query": query, "hits": hits}


def read_file(
    workspace: Path,
    path: str,
    start_line: int = 1,
    max_lines: int = MAX_READ_LINES,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
) -> Dict[str, Any]:
    """Read a bounded file window; tolerate multiple common range-key conventions."""
    if line_start is not None:
        start_line = line_start
    if line_end is not None:
        try:
            s_line = int(start_line)
            e_line = int(line_end)
            if e_line >= s_line:
                max_lines = e_line - s_line + 1
        except (TypeError, ValueError):
            pass

    target = safe_workspace_path(workspace, path)
    if not target.exists() or not target.is_file():
        return {"ok": False, "error": f"File not found: {path}"}
    if target.stat().st_size > MAX_FILE_BYTES:
        return {"ok": False, "error": f"{path} exceeds the read size limit."}

    try:
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        return {"ok": False, "error": f"{path} is not UTF-8 text."}

    try:
        requested_start = int(start_line)
    except (TypeError, ValueError):
        requested_start = 1
    try:
        requested_max = int(max_lines)
    except (TypeError, ValueError):
        requested_max = MAX_READ_LINES

    total_lines = len(lines)
    normalized_max = max(1, min(requested_max, MAX_READ_LINES))
    if total_lines == 0:
        return {"ok": True, "path": path, "start_line": 1, "end_line": 0, "total_lines": 0, "content": ""}

    normalized_start = max(1, requested_start)
    if normalized_start > total_lines:
        normalized_start = max(1, total_lines - normalized_max + 1)

    selected = lines[normalized_start - 1 : normalized_start - 1 + normalized_max]
    end_line = normalized_start + len(selected) - 1
    numbered = "".join(f"{i}: {line}" for i, line in enumerate(selected, normalized_start))

    result: Dict[str, Any] = {
        "ok": True,
        "path": path,
        "start_line": normalized_start,
        "end_line": end_line,
        "total_lines": total_lines,
        "content": numbered[:MAX_TOOL_CONTEXT_CHARS],
    }
    if normalized_start != requested_start or normalized_max != requested_max:
        result["warning"] = f"Requested range ({requested_start}, {requested_max}) was normalized to ({normalized_start}, {normalized_max})."
    return result


def apply_patch(workspace: Path, path: str, old_text: str, new_text: str) -> Dict[str, Any]:
    target = safe_workspace_path(workspace, path)
    if not target.exists() or not target.is_file():
        return {"ok": False, "error": f"File not found: {path}"}
    if len(old_text.encode()) > MAX_PATCH_BYTES or len(new_text.encode()) > MAX_PATCH_BYTES:
        return {"ok": False, "error": "Patch block is too large; use smaller chunks."}
    if any(token in Path(path).parts for token in [".git", ".venv", "venv"]):
        return {"ok": False, "error": "Cannot patch environment directories."}
    if Path(path).name in {"streamlit.py", "groq.py", "pytest.py", "subprocess.py", "os.py"}:
        return {"ok": False, "error": "Dependency-shadowing filenames are blocked."}
    text = target.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        return {
            "ok": False,
            "error": f"Patch anchor must occur exactly once; found {count} occurrences in {path}.",
        }
    target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return {"ok": True, "path": path, "bytes_written": len(new_text.encode())}


def create_file(workspace: Path, path: str, content: str) -> Dict[str, Any]:
    target = safe_workspace_path(workspace, path)
    if target.exists():
        return {"ok": False, "error": f"File already exists: {path}. Use apply_patch or append_file."}
    if path.endswith(".py") and Path(path).name in {"streamlit.py", "groq.py", "pytest.py", "subprocess.py", "os.py"}:
        return {"ok": False, "error": "Dependency-shadowing filenames are blocked."}
    if len(content.encode()) > MAX_CREATE_BYTES:
        return {"ok": False, "error": "New file is too large; create it with append_file chunks."}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "bytes_written": len(content.encode())}


def append_file(workspace: Path, path: str, content: str) -> Dict[str, Any]:
    target = safe_workspace_path(workspace, path)
    if path.endswith(".py") and Path(path).name in {"streamlit.py", "groq.py", "pytest.py", "subprocess.py", "os.py"}:
        return {"ok": False, "error": "Dependency-shadowing filenames are blocked."}
    if len(content.encode()) > MAX_CREATE_BYTES:
        return {"ok": False, "error": "Append chunk is too large."}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(content)
    return {"ok": True, "path": path, "bytes_appended": len(content.encode())}


def sandbox_call(mode: str, workspace: Path, target: str = "", timeout: int = 20) -> Dict[str, Any]:
    if not SANDBOX_RUNNER.exists():
        return {
            "ok": False,
            "error_type": "infrastructure",
            "error": "sandbox_runner.py is missing from deployment.",
        }
    command = [
        os.environ.get("PYTHON", "python"),
        str(SANDBOX_RUNNER),
        "--mode",
        mode,
        "--workspace",
        str(workspace),
        "--timeout",
        str(timeout),
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
            env={"PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error_type": "infrastructure", "error": "Sandbox controller timed out."}
    except OSError as exc:
        return {"ok": False, "error_type": "infrastructure", "error": str(exc)}
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    payload: Dict[str, Any] | None = None
    if lines:
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        return {
            "ok": False,
            "error_type": "infrastructure" if result.returncode != 0 else "runner",
            "error": "Sandbox runner returned invalid JSON.",
            "runner_stdout": (result.stdout or "")[-6000:],
            "runner_stderr": (result.stderr or "")[-4000:],
        }
    if result.returncode != 0 and payload.get("ok", False):
        payload["ok"] = False
    return payload


def run_compile(workspace: Path, path: str) -> Dict[str, Any]:
    return sandbox_call("compile", workspace, path, timeout=12)


def run_tests(workspace: Path) -> Dict[str, Any]:
    return sandbox_call("pytest", workspace, timeout=45)


def smoke_test_streamlit(workspace: Path, path: str) -> Dict[str, Any]:
    return sandbox_call("streamlit", workspace, path, timeout=20)


def determine_entrypoint(workspace: Path) -> Optional[str]:
    candidates = ["app.py", "streamlit_app.py", "main.py"]
    for candidate in candidates:
        if (workspace / candidate).exists():
            try:
                text = (workspace / candidate).read_text(encoding="utf-8", errors="ignore")
                if "streamlit" in text.lower() or candidate != "main.py":
                    return candidate
            except OSError:
                pass
    for path in workspace.rglob("*.py"):
        if any(p in {".venv", "venv", ".git", "__pycache__"} for p in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "import streamlit" in text or "from streamlit" in text:
                return path.relative_to(workspace).as_posix()
        except OSError:
            continue
    return None


TOOL_LIST = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List project files. Use {} as arguments.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a bounded UTF-8 file window. Preferred keys: path, start_line, max_lines. "
                "Aliases line_start and line_end are accepted. Do not send full-file rewrites."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "max_lines": {"type": "integer"},
                    "line_start": {"type": "integer"},
                    "line_end": {"type": "integer"},
                },
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_file",
            "description": "Search one project file. Keys: path and query.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "query": {"type": "string"}},
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Replace exactly one existing text block. Keys: path, old_text, new_text. Use small localized edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new UTF-8 file. Keys: path and content. Keep content reasonably small.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append one small UTF-8 chunk. Keys: path and content.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "additionalProperties": True,
            },
        },
    },
]


IMPLEMENT_TOOLS = [TOOL_LIST[0], TOOL_LIST[1], TOOL_LIST[2], TOOL_LIST[3], TOOL_LIST[4], TOOL_LIST[5]]
READ_TOOLS = [TOOL_LIST[0], TOOL_LIST[1], TOOL_LIST[2]]
TEST_TOOLS = [TOOL_LIST[0], TOOL_LIST[1], TOOL_LIST[2], TOOL_LIST[4], TOOL_LIST[5]]

IMPLEMENTER_PROMPT = """
You are DevPilot's implementation engineer.

Implement the requested change in the supplied workspace.

Rules:
- Use list_files/search_file/read_file before editing existing code.
- Existing files MUST be changed with apply_patch, not whole-file rewrites.
- Use small localized patches. Prefer 1-3 patches over rewriting a file.
- New test files may use create_file/append_file in chunks.
- Never create files that impersonate dependencies (for example streamlit.py, groq.py, pytest.py).
- Never modify secrets, .env files, or credentials.
- Do not change unrelated behavior.
- Do not claim completion until you have made the requested change.
"""

INSPECTOR_PROMPT = """
You are DevPilot's codebase inspector. Inspect only what is needed to understand the requested change.
Use list_files first, then search/read small relevant sections. Do not edit anything.
Finish with a concise inspection summary naming the relevant files and why they matter.
"""

TESTER_PROMPT = """
You are DevPilot's test engineer. Generate requirement-level functional tests for the requested change.
Use the existing implementation and inspection context to avoid guessing labels or APIs.
For Streamlit apps, prefer streamlit.testing.v1.AppTest when practical.
Create only focused tests that prove the acceptance criteria. Do not modify application code while creating tests.
"""

REPAIR_PROMPT = """
You are DevPilot's repair engineer.
A verification failure has occurred. Read the failure output and inspect only the code needed to fix the root cause.
Use localized apply_patch edits. Do not rewrite whole files. Do not alter tests merely to make them pass unless the test itself is objectively wrong relative to the requirement.
Do not invent dependency-shadowing modules or modify secrets.
After the repair, stop; the controller will run verification again.
"""


class WorkflowEngine:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, callback: Optional[Callable[[WorkflowState], None]] = None):
        self.client = Groq(api_key=api_key)
        self.model = model or DEFAULT_MODEL
        self.callback = callback
        self.state = WorkflowState()

    def _set(self, stage: str, value: str, detail: Optional[str] = None) -> None:
        set_stage(self.state, stage, value, detail)
        callback_event(self.state, self.callback)

    def _event(self, tool: str, ok: bool, detail: str, result: Any) -> None:
        log_event(self.state, tool, ok, detail, result)
        callback_event(self.state, self.callback)

    def _request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = MAX_LLM_OUTPUT,
    ):
        common: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": min(int(max_tokens), MAX_LLM_OUTPUT),
        }
        if tools is not None:
            common["tools"] = tools
            common["tool_choice"] = "auto"
            common["parallel_tool_calls"] = False

        rate_retries = 0
        malformed_retries = 0
        while True:
            try:
                return self.client.chat.completions.create(**common)
            except RateLimitError as exc:
                if rate_retries >= MAX_RATE_LIMIT_RETRIES:
                    raise AgentRateLimitError(
                        "Groq token rate limit reached. The workflow stopped after one bounded retry."
                    ) from exc
                rate_retries += 1
                import re, time
                match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", str(exc), re.I)
                wait_s = min(15.0, max(2.0, float(match.group(1)) if match else 8.0))
                time.sleep(wait_s + 0.25)
            except BadRequestError as exc:
                message = str(exc)
                lower = message.lower()
                is_schema = "invalid json schema" in lower or ("schema" in lower and "parameters" in lower and "tool" in lower)
                is_tool_bad = (
                    "failed to parse tool call arguments" in lower
                    or "tool_use_failed" in lower
                    or "invalid tool call" in lower
                    or "tool call validation failed" in lower
                )
                if tools is not None and is_tool_bad and not is_schema and malformed_retries < 1:
                    malformed_retries += 1
                    common["max_tokens"] = min(int(common["max_tokens"]), 900)
                    continue
                raise FatalAgentError(f"Groq rejected the request: {message[:900]}") from exc
            except Exception as exc:
                raise FatalAgentError(f"Groq request failed: {exc}") from exc

    def _tool_error(self, name: str, message: str) -> Dict[str, Any]:
        result = {"ok": False, "error_type": "tool_arguments", "error": message}
        self._event(name, False, f"{name.replace('_', ' ').title()} · {message[:120]}", result)
        return result

    def _tool_execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        workspace = self.state.workspace
        assert workspace is not None
        try:
            if name == "list_files":
                result = list_files(workspace)
            elif name == "read_file":
                path = args.get("path")
                if not isinstance(path, str) or not path.strip():
                    return self._tool_error(name, "read_file requires a path string.")
                result = read_file(
                    workspace,
                    path,
                    start_line=args.get("start_line", args.get("line_start", 1)),
                    max_lines=args.get("max_lines", MAX_READ_LINES),
                    line_start=args.get("line_start"),
                    line_end=args.get("line_end"),
                )
            elif name == "search_file":
                path, query = args.get("path"), args.get("query")
                if not isinstance(path, str) or not isinstance(query, str):
                    return self._tool_error(name, "search_file requires path and query strings.")
                result = search_file(workspace, path, query)
            elif name == "apply_patch":
                path = args.get("path") or args.get("file_path")
                old_text = args.get("old_text")
                if old_text is None:
                    old_text = args.get("old", args.get("find_text"))
                new_text = args.get("new_text")
                if new_text is None:
                    new_text = args.get("new", args.get("replace_with"))
                if not all(isinstance(v, str) for v in (path, old_text, new_text)):
                    return self._tool_error(name, "apply_patch requires path, old_text and new_text strings.")
                result = apply_patch(workspace, path, old_text, new_text)
            elif name == "create_file":
                path = args.get("path") or args.get("file_path")
                content = args.get("content")
                if not isinstance(path, str) or not isinstance(content, str):
                    return self._tool_error(name, "create_file requires path and content strings.")
                result = create_file(workspace, path, content)
            elif name == "append_file":
                path = args.get("path") or args.get("file_path")
                content = args.get("content")
                if not isinstance(path, str) or not isinstance(content, str):
                    return self._tool_error(name, "append_file requires path and content strings.")
                result = append_file(workspace, path, content)
            else:
                result = {"ok": False, "error": f"Unknown tool: {name}"}
        except Exception as exc:
            result = {"ok": False, "error_type": "tool_execution", "error": str(exc)}
        detail = name.replace("_", " ").title()
        path_value = args.get("path") or args.get("file_path")
        if isinstance(path_value, str):
            detail += f" · {path_value}"
        if not result.get("ok", False):
            detail += f" · {str(result.get('error', 'failed'))[:120]}"
        self._event(name, bool(result.get("ok", False)), detail, result)
        return result

    def _tool_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        max_turns: int = MAX_AGENT_TURNS,
        max_tokens: int = MAX_LLM_OUTPUT,
    ) -> str:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        final_text = ""
        repeated_failure_signature = None
        repeated_failure_count = 0
        for _ in range(max_turns):
            response = self._request(messages, tools=tools, max_tokens=max_tokens)
            message = response.choices[0].message
            assistant: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                assistant["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            messages.append(assistant)
            if not message.tool_calls:
                final_text = message.content or ""
                break
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                    result = self._tool_execute(tc.function.name, args)
                    if not result.get("ok", False):
                        signature = (tc.function.name, str(result.get("error", ""))[:300])
                        if signature == repeated_failure_signature:
                            repeated_failure_count += 1
                        else:
                            repeated_failure_signature = signature
                            repeated_failure_count = 1
                        if repeated_failure_count >= 2:
                            raise FatalAgentError(
                                f"Repeated tool failure stopped the stage: {tc.function.name} → {str(result.get('error', 'failed'))[:300]}"
                            )
                except json.JSONDecodeError as exc:
                    result = {"ok": False, "error_type": "tool_arguments", "fatal": True, "error": f"Invalid JSON arguments: {exc}"}
                    self._event(tc.function.name, False, "Malformed local tool arguments", result)
                    raise FatalAgentError(f"Malformed arguments for tool {tc.function.name}: {exc}") from exc
                except Exception as exc:
                    result = {"ok": False, "error_type": "tool_execution", "error": str(exc)}
                    self._event(tc.function.name, False, "Tool execution error", result)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)[:MAX_TOOL_CONTEXT_CHARS]})
        if not final_text:
            raise FatalAgentError("Agent reached its bounded tool-call limit without finishing the stage.")
        return final_text

    def _plain(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
        response = self._request(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=None,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _run_stage_requirements(self) -> None:
        self._set("requirements", "active", "Extracting acceptance criteria")
        self.state.requirement_plan = self._plain(
            """You are a requirements analyst. Convert the user's software request into a compact engineering brief.
Return exactly these headings: Goal, Acceptance criteria, Constraints, Assumptions.
Acceptance criteria must be observable and testable. Do not design implementation details.""",
            self.state.requirements,
            max_tokens=900,
        )
        self._set("requirements", "done", "Acceptance criteria extracted")

    def _run_stage_inspection(self) -> None:
        self._set("inspection", "active", "Mapping relevant files")
        result_text = self._tool_loop(
            INSPECTOR_PROMPT,
            f"REQUEST:\n{self.state.requirements}\n\nDo not modify files. Return a concise inspection summary. Begin with list_files.\n",
            READ_TOOLS,
            max_turns=7,
            max_tokens=900,
        )
        self.state.inspection = result_text
        self._set("inspection", "done", "Relevant files identified")

    def _run_stage_design(self) -> None:
        self._set("design", "active", "Building implementation plan")
        self.state.design = self._plain(
            """You are a senior software architect. Design the smallest implementation that satisfies the acceptance criteria.
Use the inspection summary. Prefer existing patterns. Mention files to change and verification strategy. Stay concise.""",
            f"REQUIREMENTS:\n{self.state.requirement_plan}\n\nINSPECTION:\n{self.state.inspection}",
            max_tokens=1000,
        )
        self._set("design", "done", "Implementation plan ready")

    def _run_stage_implementation(self) -> None:
        self._set("implementation", "active", "Applying localized patches")
        self._tool_loop(
            IMPLEMENTER_PROMPT,
            f"REQUEST:\n{self.state.requirements}\n\nREQUIREMENT BRIEF:\n{self.state.requirement_plan}\n\nINSPECTION:\n{self.state.inspection}\n\nDESIGN:\n{self.state.design}\n\nImplement now. End with a concise summary.",
            IMPLEMENT_TOOLS,
            max_turns=8,
            max_tokens=1200,
        )
        self._set("implementation", "done", "Requested changes applied")

    def _run_stage_test_generation(self) -> None:
        self._set("test_generation", "active", "Generating focused requirement tests")
        workspace = self.state.workspace
        assert workspace is not None
        if (workspace / GENERATED_TEST_PATH).exists():
            (workspace / GENERATED_TEST_PATH).unlink()
        test_prompt = f"""
REQUEST:
{self.state.requirements}

ACCEPTANCE CRITERIA:
{self.state.requirement_plan}

INSPECTION:
{self.state.inspection}

IMPLEMENTATION SUMMARY:
Review the current files as needed. Create {GENERATED_TEST_PATH}.
Rules: generate a small number of requirement-focused tests; use pytest; for Streamlit use AppTest when practical; do not assume features not supported by the requirement; don't modify app code.
"""
        self._tool_loop(TESTER_PROMPT, test_prompt, TEST_TOOLS, max_turns=8, max_tokens=1100)
        if not (workspace / GENERATED_TEST_PATH).exists():
            raise FatalAgentError("Test generation stage finished without creating requirement tests.")
        self.state.test_path = GENERATED_TEST_PATH
        self._set("test_generation", "done", GENERATED_TEST_PATH)

    def _compile_all(self) -> Dict[str, Any]:
        workspace = self.state.workspace
        assert workspace is not None
        python_files = []
        for path in workspace.rglob("*.py"):
            if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
                continue
            python_files.append(path.relative_to(workspace).as_posix())
        errors = []
        for rel in python_files[:120]:
            result = run_compile(workspace, rel)
            self._event("compile", bool(result.get("ok")), f"Compile · {rel}", result)
            if not result.get("ok"):
                errors.append({"file": rel, "result": result})
                if result.get("error_type") == "infrastructure":
                    break
        return {"ok": not errors, "files_checked": len(python_files), "errors": errors}

    def _run_stage_testing(self) -> None:
        self._set("testing", "active", "Running syntax and requirement tests")
        compile_result = self._compile_all()
        if not compile_result["ok"]:
            self.state.verification["syntax_ok"] = False
            if any(e["result"].get("error_type") == "infrastructure" for e in compile_result["errors"]):
                self.state.verification["infrastructure_error"] = True
                raise InfrastructureFailure("Sandbox infrastructure prevented syntax verification.")
            raise FatalAgentError("Python syntax verification failed.\n" + json.dumps(compile_result["errors"], ensure_ascii=False)[:8000])
        self.state.verification["syntax_ok"] = True

        test_result = run_tests(self.state.workspace)  # type: ignore[arg-type]
        self._event("run_tests", bool(test_result.get("ok")), "Requirement tests", test_result)
        self.state.verification["tests_ok"] = bool(test_result.get("ok"))
        self.state.verification["test_failure"] = not self.state.verification["tests_ok"]
        if test_result.get("error_type") == "infrastructure":
            self.state.verification["infrastructure_error"] = True
            raise InfrastructureFailure(str(test_result.get("error", "Sandbox infrastructure failure")))
        self._set(
            "testing",
            "done" if self.state.verification["tests_ok"] and self.state.verification["syntax_ok"] else "error",
            "Syntax and requirement tests passed" if self.state.verification["tests_ok"] and self.state.verification["syntax_ok"] else "Verification failed",
        )

    def _failure_context(self) -> str:
        failed = [e for e in self.state.events if not e["ok"]][-6:]
        compact = []
        for event in failed:
            result = event.get("result", {})
            compact.append({
                "tool": event.get("tool"),
                "detail": event.get("detail"),
                "result": result,
            })
        return json.dumps(compact, ensure_ascii=False)[:12_000]

    def _run_failure_stages(self) -> None:
        self._set("failure_analysis", "active", "Reading failed verification output")
        analysis = self._plain(
            """You are a failure analyst. Analyze the supplied verification failures and identify the most likely root cause.
Do not propose code yet. Distinguish application/test failures from sandbox or infrastructure failures. Return a concise diagnosis.""",
            f"REQUIREMENTS:\n{self.state.requirement_plan}\n\nFAILURES:\n{self._failure_context()}",
            max_tokens=900,
        )
        self._set("failure_analysis", "done", analysis[:180])

        self._set("error_handling", "active", "Classifying failure and deciding repairability")
        decision = self._plain(
            """You are DevPilot's error-handling controller. Decide whether the failure is safe for an automated repair attempt.
Return exactly: Classification, Repairable (yes/no), Reason.
Infrastructure/resource/provider failures are not repairable by modifying user code.""",
            f"ANALYSIS:\n{analysis}\n\nFAILURES:\n{self._failure_context()}",
            max_tokens=600,
        )
        self._set("error_handling", "done", decision[:180])
        if "repairable (yes)" not in decision.lower():
            raise FatalAgentError(f"Automatic repair stopped: {decision}")

        self._set("repair", "active", "Applying a bounded repair")
        repair_prompt = f"""
REQUIREMENT:
{self.state.requirements}

ANALYSIS:
{analysis}

ERROR-HANDLING DECISION:
{decision}

VERIFICATION FAILURE:
{self._failure_context()}

Repair only the application code necessary to satisfy the requirement and fix the root cause. Use localized patches.
"""
        self._tool_loop(REPAIR_PROMPT, repair_prompt, IMPLEMENT_TOOLS, max_turns=8, max_tokens=1100)
        self.state.repair_attempts += 1
        self._set("repair", "done", f"Repair attempt {self.state.repair_attempts} applied")

    def _run_stage_sandbox(self) -> None:
        self._set("sandbox", "active", "Starting Streamlit health check")
        workspace = self.state.workspace
        assert workspace is not None
        entry = determine_entrypoint(workspace)
        if entry is None:
            self._set("sandbox", "skipped", "No Streamlit entrypoint detected")
            self.state.verification["smoke_ok"] = True
            return
        result = smoke_test_streamlit(workspace, entry)
        self._event("smoke_test_streamlit", bool(result.get("ok")), f"Streamlit smoke test · {entry}", result)
        self.state.verification["smoke_ok"] = bool(result.get("ok"))
        if result.get("error_type") == "infrastructure":
            self.state.verification["infrastructure_error"] = True
            raise InfrastructureFailure(str(result.get("error", "Streamlit sandbox infrastructure failure")))
        if not result.get("ok"):
            raise FatalAgentError(str(result.get("error", "Streamlit smoke test failed.")))
        self._set("sandbox", "done", "HTTP health endpoint returned 200")

    def _run_final_report(self) -> None:
        self._set("final_report", "active", "Preparing verified development report")
        verification = json.dumps(self.state.verification, ensure_ascii=False)
        self.state.final_report = self._plain(
            """You are DevPilot's release reporter. Produce a concise engineering report.
Use only the supplied facts. Never claim verification that did not pass.
Sections: Result, Changes, Verification, Remaining issues.""",
            f"REQUIREMENTS:\n{self.state.requirement_plan}\n\nDESIGN:\n{self.state.design}\n\nVERIFICATION:\n{verification}\n\nEVENTS:\n{json.dumps(self.state.events[-12:], ensure_ascii=False)[:9000]}",
            max_tokens=1000,
        )
        self._set("final_report", "done", "Report generated from verified state")

    def run(self, uploaded_file, requirement: str) -> WorkflowState:
        self.state = WorkflowState()
        self.state.requirements = requirement.strip()
        try:
            self.state.workspace = extract_project(uploaded_file)
            self._run_stage_requirements()
            self._run_stage_inspection()
            self._run_stage_design()
            self._run_stage_implementation()
            self._run_stage_test_generation()

            repair_loops = 0
            while True:
                try:
                    self._run_stage_testing()
                    break
                except InfrastructureFailure:
                    self.state.aborted = True
                    self.state.abort_reason = "Verification infrastructure failed. Code repair was intentionally skipped."
                    self._set("repair", "skipped", "Sandbox infrastructure failure; no code repair attempted")
                    break
                except FatalAgentError:
                    if self.state.verification.get("infrastructure_error"):
                        self.state.aborted = True
                        self.state.abort_reason = "Verification infrastructure failed."
                        self._set("repair", "skipped", "Infrastructure failure; no code repair attempted")
                        break
                    if repair_loops >= MAX_REPAIR_ATTEMPTS:
                        self.state.aborted = True
                        self.state.abort_reason = "Maximum repair attempts reached."
                        self._set("repair", "error", "Maximum repair attempts reached")
                        break
                    self._run_failure_stages()
                    repair_loops += 1

            if not self.state.aborted and self.state.verification["tests_ok"] and self.state.verification["syntax_ok"]:
                try:
                    self._run_stage_sandbox()
                except FatalAgentError as exc:
                    self.state.aborted = True
                    self.state.abort_reason = str(exc)
                    self._set("sandbox", "error", self.state.abort_reason[:180])

            if self.state.aborted:
                self._set("final_report", "error", self.state.abort_reason[:180])
                self.state.final_report = (
                    "## Result\n\nDevelopment run stopped before final verification.\n\n"
                    f"**Reason:** {self.state.abort_reason}\n\n"
                    "The agent did not claim the change was verified."
                )
            else:
                self._run_final_report()
        except (AgentRateLimitError, FatalAgentError, InfrastructureFailure) as exc:
            self.state.aborted = True
            self.state.abort_reason = str(exc)
            self._set("final_report", "error", self.state.abort_reason[:180])
            self.state.final_report = (
                "## Result\n\nDevelopment run stopped safely.\n\n"
                f"**Reason:** {self.state.abort_reason}\n"
            )
        except Exception as exc:
            self.state.aborted = True
            self.state.abort_reason = f"Unexpected workflow error: {exc}"
            self._set("final_report", "error", self.state.abort_reason[:180])
            self.state.final_report = f"## Result\n\nDevelopment run stopped safely.\n\n**Reason:** {self.state.abort_reason}"
        finally:
            callback_event(self.state, self.callback)
        return self.state
