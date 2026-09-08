from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from groq import Groq, RateLimitError


DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_FILE_BYTES = 300_000
MAX_CONTEXT_CHARS = 16_000

MAX_STAGE_CALLS = 1
MAX_REPAIR_ATTEMPTS = 2
MAX_RATE_LIMIT_RETRIES = 1
MAX_OUTPUT_TOKENS = 1800

STAGES = [
    ("requirements", "Requirements", "Turn the user request into explicit acceptance criteria."),
    ("inspection", "Inspection", "Understand project structure and identify relevant files."),
    ("design", "Solution design", "Choose a minimal implementation plan."),
    ("implementation", "Implementation", "Apply focused edits to the project."),
    ("test_generation", "Test generation", "Create or update focused tests for the requirement."),
    ("testing", "Testing", "Run deterministic project checks and requirement tests."),
    ("failure_analysis", "Failure analysis", "Classify real project failures."),
    ("repair", "Repair", "Fix project failures with bounded edits."),
    ("sandbox", "Sandbox verification", "Verify the application in a controlled subprocess."),
    ("final_report", "Final report", "Summarize only what was actually verified."),
]


@dataclass
class WorkflowState:
    workspace: Path | None = None
    stage_state: dict[str, str] = field(
        default_factory=lambda: {k: "pending" for k, _, _ in STAGES}
    )
    stage_detail: dict[str, str] = field(
        default_factory=lambda: {k: d for k, _, d in STAGES}
    )
    events: list[dict[str, Any]] = field(default_factory=list)
    repair_attempts: int = 0
    aborted: bool = False
    abort_reason: str = ""
    final_report: str = ""
    verification: dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        callback: Callable[[WorkflowState], None] | None = None,
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.callback = callback

    # -------------------------
    # State/UI helpers
    # -------------------------
    def update(
        self,
        state: WorkflowState,
        stage: str,
        status: str,
        detail: str | None = None,
    ) -> None:
        state.stage_state[stage] = status
        if detail:
            state.stage_detail[stage] = detail
        if self.callback:
            self.callback(state)

    def event(
        self,
        state: WorkflowState,
        detail: str,
        ok: bool,
        result: dict[str, Any],
    ) -> None:
        state.events.append({
            "tool": "workflow",
            "detail": detail,
            "ok": ok,
            "result": result,
        })
        if self.callback:
            self.callback(state)

    def fail(
        self,
        state: WorkflowState,
        reason: str,
        stage: str,
    ) -> None:
        state.aborted = True
        state.abort_reason = reason
        self.update(state, stage, "error", reason)

    # -------------------------
    # Project handling
    # -------------------------
    def extract(self, uploaded_file) -> Path:
        raw = uploaded_file.getvalue()

        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("Project ZIP exceeds the 20 MB upload limit.")

        temp_root = Path(tempfile.mkdtemp(prefix="devpilot_"))
        workspace = temp_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        zip_path = temp_root / "project.zip"
        zip_path.write_bytes(raw)

        with zipfile.ZipFile(zip_path, "r") as zf:
            total = 0
            for member in zf.infolist():
                p = Path(member.filename)
                total += member.file_size

                if p.is_absolute() or ".." in p.parts:
                    raise ValueError("Unsafe ZIP path detected.")

                if total > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("Uncompressed project exceeds the safety limit.")

            zf.extractall(workspace)

        entries = list(workspace.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            top = entries[0]
            for child in list(top.iterdir()):
                shutil.move(str(child), str(workspace / child.name))
            top.rmdir()

        return workspace

    @staticmethod
    def safe_path(workspace: Path, relative: str) -> Path:
        candidate = (workspace / relative).resolve()
        root = workspace.resolve()

        if candidate != root and root not in candidate.parents:
            raise ValueError("Path escapes project workspace.")

        return candidate

    def project_map(self, workspace: Path) -> list[dict[str, Any]]:
        rows = []
        ignore = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}

        for p in workspace.rglob("*"):
            if not p.is_file():
                continue

            rel = p.relative_to(workspace)
            if any(part in ignore for part in rel.parts):
                continue

            try:
                size = p.stat().st_size
            except OSError:
                continue

            rows.append({"path": rel.as_posix(), "size": size})

        rows.sort(key=lambda x: x["path"])
        return rows[:500]

    def read_text(self, workspace: Path, relative: str) -> str:
        target = self.safe_path(workspace, relative)

        if not target.exists() or not target.is_file():
            raise ValueError(f"File not found: {relative}")

        if target.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{relative} exceeds the read-size limit.")

        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{relative} is not UTF-8 text.") from exc

    # -------------------------
    # LLM layer
    # No Groq function/tool calling.
    # This deliberately avoids the schema failures
    # that were causing repeated 400 errors.
    # -------------------------
    def llm_json(
        self,
        system: str,
        user: str,
        state: WorkflowState,
    ) -> dict[str, Any]:
        last_error = None

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                system
                                + "\\n\\nIMPORTANT: Return the response as valid JSON. "
                                  "The response must be a JSON object."
                            ),
                        },
                        {
                            "role": "user",
                            "content": user
                            + "\\n\\nReturn valid JSON only. Do not use Markdown fences.",
                        },
                    ],
                    temperature=0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    response_format={"type": "json_object"},
                )

                text = response.choices[0].message.content or "{}"
                # Be defensive with provider/model responses.
                text = text.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\\s*", "", text, flags=re.I)
                    text = re.sub(r"\\s*```$", "", text)
                data = json.loads(text)

                if not isinstance(data, dict):
                    raise ValueError("LLM returned JSON but not an object.")

                return data

            except RateLimitError as exc:
                last_error = exc
                if attempt >= MAX_RATE_LIMIT_RETRIES:
                    raise RuntimeError(
                        "Groq rate limit remained exceeded after one retry."
                    ) from exc

                time.sleep(10)

            except json.JSONDecodeError as exc:
                last_error = exc
                break

            except Exception as exc:
                last_error = exc
                break

        raise RuntimeError(f"Structured LLM response failed: {last_error}")

    # -------------------------
    # Editing
    # -------------------------
    @staticmethod
    def normalize_edits(data: dict[str, Any]) -> list[dict[str, str]]:
        edits = data.get("edits", [])
        if not isinstance(edits, list):
            return []

        normalized = []

        for edit in edits[:3]:
            if not isinstance(edit, dict):
                continue

            path = edit.get("path")
            search = edit.get("search")
            replace = edit.get("replace")

            if not all(isinstance(x, str) for x in (path, search, replace)):
                continue

            if not search:
                continue

            normalized.append({
                "path": path,
                "search": search,
                "replace": replace,
            })

        return normalized

    def apply_edits(
        self,
        workspace: Path,
        edits: list[dict[str, str]],
    ) -> dict[str, Any]:
        changed = []
        originals: dict[Path, str | None] = {}

        def find_normalized_match(current: str, search: str):
            current_normalized = current.replace("\r\n", "\n").replace("\r", "\n")
            search_normalized = search.replace("\r\n", "\n").replace("\r", "\n")

            if search_normalized in current_normalized:
                start = current_normalized.index(search_normalized)
                return start, start + len(search_normalized)

            current_lines = current_normalized.split("\n")
            search_lines = search_normalized.split("\n")

            while search_lines and not search_lines[0].strip():
                search_lines.pop(0)
            while search_lines and not search_lines[-1].strip():
                search_lines.pop()

            if not search_lines:
                return None

            wanted = [line.strip() for line in search_lines]

            for i in range(len(current_lines) - len(search_lines) + 1):
                candidate = [line.strip() for line in current_lines[i:i + len(search_lines)]]
                if candidate == wanted:
                    start = sum(len(line) + 1 for line in current_lines[:i])
                    end = sum(len(line) + 1 for line in current_lines[:i + len(search_lines)])
                    if end > start and current_normalized.endswith("\n"):
                        end -= 1
                    return start, end

            return None

        for edit in edits:
            target = self.safe_path(workspace, edit["path"])

            blocked_names = {
                "streamlit.py",
                "groq.py",
                "pytest.py",
                "subprocess.py",
                "os.py",
                "json.py",
            }

            if target.name in blocked_names:
                continue

            if "secrets" in target.parts or target.name in {".env", "secrets.toml"}:
                continue

            if target.exists():
                current = self.read_text(workspace, edit["path"])

                if edit["search"] in current:
                    updated = current.replace(edit["search"], edit["replace"], 1)
                else:
                    match = find_normalized_match(current, edit["search"])
                    if match is None:
                        continue

                    start, end = match
                    normalized_current = current.replace("\r\n", "\n").replace("\r", "\n")
                    updated = normalized_current[:start] + edit["replace"] + normalized_current[end:]
            else:
                originals.setdefault(target, None)
                target.parent.mkdir(parents=True, exist_ok=True)
                updated = edit["replace"]

            if target not in originals:
                originals[target] = current if target.exists() else None

            if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
                continue

            target.write_text(updated, encoding="utf-8")

            if edit["path"] not in changed:
                changed.append(edit["path"])

        if changed:
            compile_result = self.compile_project(workspace)

            if not compile_result["ok"]:
                for target, original in originals.items():
                    if original is None:
                        try:
                            target.unlink(missing_ok=True)
                        except OSError:
                            pass
                    else:
                        target.write_text(original, encoding="utf-8")

                return {
                    "ok": False,
                    "changed_files": [],
                    "count": 0,
                    "rolled_back": True,
                    "error": (
                        "Edits were rolled back because they introduced "
                        "a Python syntax error."
                    ),
                    "compile": compile_result,
                }

        return {
            "ok": True,
            "changed_files": changed,
            "count": len(changed),
        }
    # -------------------------
    # Deterministic verification
    # -------------------------
    def compile_project(self, workspace: Path) -> dict[str, Any]:
        errors = []

        for p in list(workspace.rglob("*.py"))[:150]:
            rel = p.relative_to(workspace).as_posix()

            try:
                source = p.read_text(encoding="utf-8")
            except Exception:
                continue

            if len(source.encode("utf-8")) > MAX_FILE_BYTES:
                continue

            try:
                compile(source, rel, "exec")
            except SyntaxError as exc:
                errors.append({
                    "file": rel,
                    "line": exc.lineno,
                    "message": exc.msg,
                })

        return {
            "ok": not errors,
            "mode": "python_compile",
            "errors": errors,
        }

    def run_pytest(self, workspace: Path) -> dict[str, Any]:
        tests = list(workspace.rglob("test_*.py")) + list(
            workspace.rglob("*_test.py")
        )

        if not tests:
            return {
                "ok": True,
                "mode": "pytest",
                "message": "No pytest tests present.",
            }

        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", "-q"],
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
            "stdout": (proc.stdout or "")[-6000:],
            "stderr": (proc.stderr or "")[-6000:],
        }

    def smoke_test_streamlit(
        self,
        workspace: Path,
        entry: str,
    ) -> dict[str, Any]:
        target = self.safe_path(workspace, entry)

        if not target.exists() or target.suffix != ".py":
            return {
                "ok": False,
                "error_type": "project",
                "error": "Streamlit entry file does not exist.",
            }

        import socket
        import requests

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        proc = None

        try:
            proc = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "streamlit",
                    "run",
                    entry,
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
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONUNBUFFERED": "1",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                },
            )

            deadline = time.time() + 12

            while time.time() < deadline:
                if proc.poll() is not None:
                    output = ""
                    if proc.stdout:
                        output = proc.stdout.read()[-6000:]

                    return {
                        "ok": False,
                        "error_type": "project_or_environment",
                        "returncode": proc.returncode,
                        "error": "Streamlit exited before becoming ready.",
                        "startup_log": output,
                    }

                try:
                    response = requests.get(
                        f"http://127.0.0.1:{port}",
                        timeout=1,
                    )

                    if response.status_code == 200:
                        return {
                            "ok": True,
                            "mode": "streamlit_smoke",
                            "http_status": response.status_code,
                        }
                except requests.RequestException:
                    pass

                time.sleep(0.4)

            return {
                "ok": False,
                "error_type": "project_or_environment",
                "error": "Streamlit did not become ready before timeout.",
            }

        except Exception as exc:
            return {
                "ok": False,
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

    # -------------------------
    # Workflow stages
    # -------------------------
    def run(self, uploaded_file, requirement: str) -> WorkflowState:
        state = WorkflowState()

        try:
            state.workspace = self.extract(uploaded_file)

            # 1. Requirements
            self.update(state, "requirements", "active", "Extracting acceptance criteria")
            criteria = self.llm_json(
                """
You are the requirements analyst.
Turn the user request into explicit, testable acceptance criteria.
Return JSON only:
{
  "criteria": ["criterion 1", "criterion 2"]
}
""",
                f"Requirement:\n{requirement[:5000]}",
                state,
            )
            state.verification["requirements"] = criteria
            self.event(state, "Requirements extracted", True, criteria)
            self.update(state, "requirements", "done", "Acceptance criteria extracted")

            # 2. Inspection
            self.update(state, "inspection", "active", "Inspecting project structure")
            project_files = self.project_map(state.workspace)
            inspection = self.llm_json(
                """
You are the project-inspection specialist.
Inspect the supplied project file list and identify the relevant files and entry point.
Return JSON only:
{
  "relevant_files": ["app.py"],
  "entry_point": "app.py",
  "notes": "brief"
}
Rules:
- Only use files present in the supplied file list.
- For Streamlit projects, prefer the main Streamlit .py file as entry_point.
""",
                f"Requirement:\n{requirement[:5000]}\n\nFiles:\n{json.dumps(project_files)[:MAX_CONTEXT_CHARS]}",
                state,
            )
            relevant = inspection.get("relevant_files", [])
            if not isinstance(relevant, list):
                relevant = []
            relevant = [str(x) for x in relevant if isinstance(x, str)]

            if not relevant:
                relevant = [row["path"] for row in project_files if row["path"].endswith((".py", ".js", ".ts", ".cs"))][:8]

            snippets = []
            for path in relevant[:8]:
                try:
                    snippets.append(f"FILE: {path}\n{self.read_text(state.workspace, path)[:5000]}")
                except Exception:
                    continue

            entry_point = inspection.get("entry_point")
            if not isinstance(entry_point, str) or not entry_point:
                entry_point = "app.py" if (state.workspace / "app.py").exists() else None
            inspection["relevant_files"] = relevant
            inspection["entry_point"] = entry_point
            state.verification["inspection"] = inspection
            self.event(state, "Project inspected", True, inspection)
            self.update(state, "inspection", "done", f"Relevant files: {len(relevant)}")

            # 3. Design
            self.update(state, "design", "active", "Planning minimal solution")
            design = self.llm_json(
                """
You are the solution-design specialist.
Create a minimal implementation plan based only on the requirement and inspected code.
Return JSON only:
{
  "plan": ["step 1", "step 2"],
  "target_files": ["app.py"],
  "notes": "brief"
}
""",
                f"Requirement:\n{requirement[:5000]}\n\nInspection:\n{json.dumps(inspection)[:6000]}\n\nCurrent code:\n{chr(10).join(snippets)[:MAX_CONTEXT_CHARS]}",
                state,
            )
            state.verification["design"] = design
            self.event(state, "Solution designed", True, design)
            self.update(state, "design", "done", "Minimal implementation plan prepared")

            # 4. Implementation
            self.update(state, "implementation", "active", "Applying focused edits")
            impl = self.llm_json(
                """
You are the implementation specialist.
You MUST make focused edits, not whole-file rewrites.
Return JSON only:
{
  "edits": [
    {"path":"file.py","search":"exact existing text","replace":"replacement text"}
  ],
  "notes": "brief"
}
Rules:
- Maximum 3 edits.
- search must be copied from the supplied code.
- Do not invent missing text.
- Never modify secret files.
- Do not create dependency-shadowing files such as streamlit.py, groq.py, pytest.py.
- Keep each edit small and preserve valid indentation.
""",
                f"Requirement:\n{requirement[:5000]}\n\nPlan:\n{json.dumps(design)[:6000]}\n\nCode:\n{chr(10).join(snippets)[:MAX_CONTEXT_CHARS]}",
                state,
            )

            edits = self.normalize_edits(impl)
            applied = self.apply_edits(state.workspace, edits)

            if not applied["changed_files"]:
                self.fail(state, "Implementation produced no valid matching edits.", "implementation")
                return self._finish(state)

            self.event(state, "Implementation applied", True, applied)
            self.update(state, "implementation", "done", f"Changed: {', '.join(applied['changed_files'])}")

            # 5. Test generation
            self.update(state, "test_generation", "active", "Preparing focused requirement tests")
            test_plan = self.llm_json(
                """
You are the test-generation specialist.
Return JSON only:
{
  "test_file": "tests/test_requirement.py",
  "tests": [
    {"name":"test_name","code":"small pytest test"}
  ]
}
Rules:
- Keep tests small and deterministic.
- Prefer pure-Python tests.
- Do not add dependencies.
""",
                f"Requirement:\n{requirement[:5000]}\n\nCurrent relevant code:\n{chr(10).join(snippets)[:MAX_CONTEXT_CHARS]}",
                state,
            )

            test_file = str(test_plan.get("test_file", "tests/test_requirement.py"))
            tests = test_plan.get("tests", [])
            if isinstance(tests, list) and tests:
                test_code = "import pytest\n\n"
                for item in tests[:4]:
                    if not isinstance(item, dict):
                        continue
                    code = item.get("code")
                    if isinstance(code, str) and code.strip():
                        test_code += code.rstrip() + "\n\n"

                if test_code.strip() != "import pytest":
                    target = self.safe_path(state.workspace, test_file)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(test_code, encoding="utf-8")
                    state.verification["test_file"] = test_file
                    self.event(state, "Requirement tests generated", True, {"test_file": test_file})

            self.update(state, "test_generation", "done", "Focused tests prepared")

            # 6-8 Verification / failure analysis / repair loop
            for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
                self.update(state, "testing", "active", "Running compile and tests")
                compile_result = self.compile_project(state.workspace)
                pytest_result = self.run_pytest(state.workspace)
                testing_ok = compile_result["ok"] and pytest_result["ok"]

                self.event(
                    state,
                    f"Testing attempt {attempt + 1}",
                    testing_ok,
                    {"compile": compile_result, "pytest": pytest_result},
                )

                if testing_ok:
                    self.update(state, "testing", "done", "Compile and tests passed")
                    self.update(state, "failure_analysis", "skipped", "No test failures")
                    self.update(state, "repair", "skipped", "No repair required")
                    break

                self.update(state, "testing", "error", "Verification tests failed")
                self.update(state, "failure_analysis", "active", "Classifying test failure")

                failure_payload = {"compile": compile_result, "pytest": pytest_result}
                analysis = self.llm_json(
                    """
You are the failure-analysis specialist.
Classify the failure without editing code.
Return JSON only:
{
  "type": "project" or "infrastructure",
  "cause": "brief cause",
  "repair_needed": true or false
}
""",
                    json.dumps(failure_payload)[:MAX_CONTEXT_CHARS],
                    state,
                )

                self.event(state, "Failure analyzed", True, analysis)

                if analysis.get("type") == "infrastructure":
                    self.fail(state, str(analysis.get("cause", "Verification environment failure")), "failure_analysis")
                    return self._finish(state)

                self.update(state, "failure_analysis", "done", str(analysis.get("cause", "Project failure"))[:120])

                if attempt >= MAX_REPAIR_ATTEMPTS:
                    self.fail(state, "Maximum repair attempts reached.", "repair")
                    return self._finish(state)

                self.update(state, "repair", "active", f"Repair attempt {attempt + 1} of {MAX_REPAIR_ATTEMPTS}")

                current_code = []
                for path in relevant[:8]:
                    try:
                        current_code.append(f"FILE: {path}\n{self.read_text(state.workspace, path)[:7000]}")
                    except Exception:
                        pass

                repair = self.llm_json(
                    """
You are the repair specialist.
Return focused edits only:
{
  "edits": [
    {"path":"file.py","search":"exact current text","replace":"replacement"}
  ],
  "notes":"brief"
}
Rules:
- Maximum 2 edits.
- Use exact current text from the supplied code.
- Do not rewrite whole files.
- Do not modify secret files.
- Do not create dependency-shadowing files.
""",
                    f"Failure:\n{json.dumps(failure_payload)[:8000]}\n\nCurrent code:\n{chr(10).join(current_code)[:MAX_CONTEXT_CHARS]}",
                    state,
                )

                repair_result = self.apply_edits(
                    state.workspace,
                    self.normalize_edits(repair),
                )
                state.repair_attempts += 1

                if not repair_result["changed_files"]:
                    self.fail(state, "Repair stage produced no valid edits.", "repair")
                    return self._finish(state)

                self.event(state, f"Repair attempt {attempt + 1}", True, repair_result)
                self.update(state, "repair", "done", f"Changed: {', '.join(repair_result['changed_files'])}")

            # 9. Sandbox verification
            self.update(state, "sandbox", "active", "Running Streamlit smoke verification")
            entry = inspection.get("entry_point") or "app.py"
            entry_path = state.workspace / str(entry)

            if entry_path.exists():
                smoke = self.smoke_test_streamlit(state.workspace, str(entry))
            else:
                smoke = {
                    "ok": True,
                    "mode": "streamlit_smoke",
                    "message": "No Streamlit entry point detected.",
                }

            state.verification["sandbox"] = smoke
            self.event(state, "Sandbox verification", smoke.get("ok", False), smoke)

            if not smoke.get("ok", False):
                self.fail(state, f"Sandbox verification failed: {smoke.get('error', 'unknown error')}", "sandbox")
                return self._finish(state)

            self.update(state, "sandbox", "done", "Application responded successfully")

            # 10. Final report
            self.update(state, "final_report", "active", "Preparing verified report")
            final = self.llm_json(
                """
You are the final-report specialist.
Return JSON only:
{
  "report": "markdown report",
  "verified": true or false
}
Never claim verification that is not present in the supplied results.
Mention changed files, tests and sandbox status.
""",
                json.dumps({
                    "requirement": requirement[:4000],
                    "verification": state.verification,
                    "events": state.events[-8:],
                })[:MAX_CONTEXT_CHARS],
                state,
            )

            if final.get("verified") is True:
                state.final_report = str(final.get("report", "Development run verified."))
                self.update(state, "final_report", "done", "Verified final report generated")
            else:
                self.fail(state, "Final report could not establish verified completion.", "final_report")

        except Exception as exc:
            stage = self._current_stage(state)
            self.fail(state, f"Workflow error: {exc}", stage)

        return self._finish(state)

    @staticmethod
    def _current_stage(state: WorkflowState) -> str:
        for key, _, _ in STAGES:
            if state.stage_state.get(key) == "active":
                return key
        return "final_report"

    def _finish(self, state: WorkflowState) -> WorkflowState:
        if state.aborted:
            if not state.final_report:
                state.final_report = (
                    "## Result\n\n"
                    "Development run stopped safely.\n\n"
                    f"**Reason:** {state.abort_reason}"
                )
        elif not state.final_report:
            state.final_report = "## Result\n\nDevelopment run completed."

        if self.callback:
            self.callback(state)

        return state
