from __future__ import annotations

import ast
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
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
MAX_RATE_LIMIT_RETRIES = 2
MAX_JSON_RETRIES = 2
MAX_OUTPUT_TOKENS = 6000

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
    requirement: str = ""
    stage_state: dict[str, str] = field(
        default_factory=lambda: {k: "pending" for k, _, _ in STAGES}
    )
    stage_detail: dict[str, str] = field(
        default_factory=lambda: {k: d for k, _, d in STAGES}
    )
    events: list[dict[str, Any]] = field(default_factory=list)
    repair_attempts: int = 0
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    verification_history: list[dict[str, Any]] = field(default_factory=list)
    engineering_report: dict[str, Any] = field(default_factory=dict)
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
        # Observable execution trace: record actions/results, not private model reasoning.
        state.events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        """Parse a JSON object defensively, including fenced/model-prefixed output."""
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        try:
            value = json.loads(cleaned)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

        # Recover the first complete JSON object when a provider adds prose.
        decoder = json.JSONDecoder()
        start = cleaned.find("{")
        if start >= 0:
            value, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(value, dict):
                return value

        raise ValueError("LLM did not return a valid JSON object.")

    def llm_json(
        self,
        system: str,
        user: str,
        state: WorkflowState,
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for rate_attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            for json_attempt in range(MAX_JSON_RETRIES + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    system
                                    + "\n\nIMPORTANT: Return valid JSON. The response must be one JSON object."
                                ),
                            },
                            {
                                "role": "user",
                                "content": user
                                + "\n\nReturn JSON only. No Markdown fences or commentary.",
                            },
                        ],
                        temperature=0,
                        max_tokens=MAX_OUTPUT_TOKENS,
                    )

                    # Do NOT use Groq response_format=json_object here.
                    # Some Groq model/provider combinations can reject the request
                    # before generation with `json_validate_failed`, even when the
                    # prompt explicitly requests JSON. We enforce JSON in the prompt
                    # and parse/validate it locally instead.
                    choice = response.choices[0]
                    text = choice.message.content or ""
                    if not text.strip():
                        finish_reason = getattr(choice, "finish_reason", None)
                        # Some reasoning-heavy models can consume a small output
                        # budget without emitting visible content. Retry once with
                        # a larger budget and a compact JSON-only instruction.
                        if json_attempt < MAX_JSON_RETRIES:
                            compact_system = (
                                "Return the requested JSON object immediately. "
                                "Do not explain, reason aloud, or use Markdown. "
                                "Output only valid JSON."
                            )
                            compact_user = (
                                user
                                + "\n\nURGENT: Output the JSON object now. Keep values concise."
                            )
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": compact_system},
                                    {"role": "user", "content": compact_user},
                                ],
                                temperature=0,
                                max_tokens=MAX_OUTPUT_TOKENS,
                            )
                            text = response.choices[0].message.content or ""
                            if text.strip():
                                return self._parse_json_object(text)
                        raise ValueError(
                            "LLM returned an empty response"
                            + (f" (finish_reason={finish_reason})" if finish_reason else "")
                        )
                    return self._parse_json_object(text)

                except RateLimitError as exc:
                    last_error = exc
                    break
                except (json.JSONDecodeError, ValueError) as exc:
                    last_error = exc
                    if json_attempt < MAX_JSON_RETRIES:
                        time.sleep(0.5)
                        continue
                    break
                except Exception as exc:
                    last_error = exc
                    break

            if isinstance(last_error, RateLimitError):
                if rate_attempt >= MAX_RATE_LIMIT_RETRIES:
                    raise RuntimeError(
                        "Groq rate limit remained exceeded after retries."
                    ) from last_error
                time.sleep(8)
                continue

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
            "rollback_state": [
                {
                    "path": target.relative_to(workspace).as_posix(),
                    "original": original,
                }
                for target, original in originals.items()
            ],
        }

    def rollback_changes(
        self,
        workspace: Path,
        rollback_state: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Restore exactly the files changed by one repair attempt."""
        restored = []
        errors = []

        for item in rollback_state:
            if not isinstance(item, dict):
                continue

            path = item.get("path")
            original = item.get("original")
            if not isinstance(path, str):
                continue

            try:
                target = self.safe_path(workspace, path)
                if original is None:
                    target.unlink(missing_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(str(original), encoding="utf-8")
                restored.append(path)
            except Exception as exc:
                errors.append({"path": path, "error": str(exc)})

        return {
            "ok": not errors,
            "restored_files": restored,
            "errors": errors,
        }

    @staticmethod
    def verification_quality(
        compile_result: dict[str, Any],
        pytest_result: dict[str, Any],
    ) -> tuple[int, int, int, int]:
        """Return a conservative quality tuple for comparing repair attempts."""
        compile_ok = 1 if compile_result.get("ok") else 0
        pytest_ok = 1 if pytest_result.get("ok") else 0

        output = "\n".join(
            [
                str(pytest_result.get("stdout", "") or ""),
                str(pytest_result.get("stderr", "") or ""),
            ]
        )

        def count(pattern: str) -> int:
            matches = re.findall(pattern, output, flags=re.I)
            return int(matches[-1]) if matches else 0

        passed = count(r"(\d+)\s+passed")
        failed = count(r"(\d+)\s+failed")
        errors = count(r"(\d+)\s+errors?")

        # Tuple ordering prefers successful compile/tests, then more passing
        # tests, then fewer failures/errors. It is deliberately modest: a
        # repair is not considered successful until pytest itself passes.
        return (compile_ok, pytest_ok, passed, -(failed + errors))

    @staticmethod
    def failed_test_output(pytest_result: dict[str, Any], limit: int = 7000) -> str:
        """Extract the most useful pytest output for the next diagnosis/repair."""
        text = "\n".join(
            [
                str(pytest_result.get("stdout", "") or ""),
                str(pytest_result.get("stderr", "") or ""),
            ]
        ).strip()
        return text[-limit:]

    @staticmethod
    def likely_failure_files(pytest_result: dict[str, Any], candidates: list[str]) -> list[str]:
        """Return candidate files explicitly mentioned by pytest output."""
        output = WorkflowEngine.failed_test_output(pytest_result, limit=9000)
        explicit = []
        for path in candidates:
            if path and path in output and path not in explicit:
                explicit.append(path)
        return explicit[:8]

    @staticmethod
    def deterministic_contract_test_repair(
        workspace: Path,
        failure_text: str,
        requirement: str,
    ) -> dict[str, Any] | None:
        """Safely repair stale exact-dict test expectations caused by additive fields.

        This is intentionally narrow and deterministic. It only fires when pytest
        proves that the implementation returned a dictionary with one or more new
        keys while the test expected the same dictionary without those keys. The
        added keys (and string values when present) must also be explicitly supported
        by the user's requirement. Existing expected keys/values are never changed.
        """
        if not failure_text or not requirement:
            return None

        requirement_lower = requirement.lower()

        # Example pytest location: tests/test_tasks.py:7: AssertionError
        location_matches = re.findall(
            r"(?m)^\s*([^\s:]+\.py):(\d+):\s*AssertionError\s*$",
            failure_text,
        )
        if not location_matches:
            return None

        # Example: At index 0 diff: {'title': '...', 'priority': 'Medium'} != {...}
        diff_match = re.search(
            r"At index\s+(\d+)\s+diff:\s*(.+?)\s+!=\s*(.+?)\s*$",
            failure_text,
            flags=re.MULTILINE,
        )
        if not diff_match:
            return None

        try:
            item_index = int(diff_match.group(1))
            actual_item = ast.literal_eval(diff_match.group(2).strip())
            expected_item_from_diff = ast.literal_eval(diff_match.group(3).strip())
        except (ValueError, SyntaxError):
            return None

        if not isinstance(actual_item, dict) or not isinstance(expected_item_from_diff, dict):
            return None

        # Only additive dictionary changes are eligible. No existing value may differ.
        extra_keys = [key for key in actual_item if key not in expected_item_from_diff]
        if not extra_keys:
            return None
        if set(expected_item_from_diff) - set(actual_item):
            return None
        for key in expected_item_from_diff:
            if actual_item.get(key) != expected_item_from_diff.get(key):
                return None

        # Requirement-gated: do not rewrite an old test merely because the output
        # happens to contain an extra field. The requested feature must name the
        # new field, and string values such as "Medium" must also be supported.
        for key in extra_keys:
            if str(key).lower() not in requirement_lower:
                return None
            value = actual_item[key]
            if isinstance(value, str) and value.lower() not in requirement_lower:
                return None

        # Read the implicated test file and find the assertion at/near the pytest
        # line. Resolve it through the workspace-safe path helper semantics by
        # rejecting absolute and escaping paths.
        test_rel, line_text = location_matches[0]
        try:
            test_path = (workspace / Path(test_rel)).resolve()
            workspace_resolved = workspace.resolve()
            test_path.relative_to(workspace_resolved)
            source = test_path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            return None

        try:
            tree = ast.parse(source)
            target_line = int(line_text)
        except (SyntaxError, ValueError):
            return None

        target_assert = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            end_line = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= target_line <= end_line:
                target_assert = node
                break
        if target_assert is None or not isinstance(target_assert.test, ast.Compare):
            return None

        compare = target_assert.test
        if len(compare.ops) != 1 or not isinstance(compare.ops[0], ast.Eq) or len(compare.comparators) != 1:
            return None

        # Prefer the literal side of the equality as the expected value.
        literal_node = None
        for candidate in (compare.comparators[0], compare.left):
            if isinstance(candidate, (ast.List, ast.Tuple, ast.Dict)):
                literal_node = candidate
                break
        if literal_node is None:
            return None

        try:
            expected_literal = ast.literal_eval(literal_node)
        except (ValueError, SyntaxError):
            return None

        updated_literal = None
        if (
            isinstance(expected_literal, list)
            and 0 <= item_index < len(expected_literal)
            and isinstance(expected_literal[item_index], dict)
        ):
            # Ensure the source expectation exactly matches the diff's expected item
            # before adding only the proven missing keys.
            source_expected_item = expected_literal[item_index]
            if source_expected_item != expected_item_from_diff:
                return None
            updated = list(expected_literal)
            updated_item = dict(source_expected_item)
            for key in extra_keys:
                updated_item[key] = actual_item[key]
            updated[item_index] = updated_item
            updated_literal = updated
        elif isinstance(expected_literal, dict) and item_index == 0:
            if expected_literal != expected_item_from_diff:
                return None
            updated_literal = dict(expected_literal)
            for key in extra_keys:
                updated_literal[key] = actual_item[key]

        if updated_literal is None:
            return None

        search = ast.get_source_segment(source, literal_node)
        if not search:
            return None
        replacement = repr(updated_literal)
        if replacement == search or any(str(key) not in replacement for key in extra_keys):
            return None

        relative = test_path.relative_to(workspace_resolved).as_posix()
        return {
            "edits": [{
                "path": relative,
                "search": search,
                "replace": replacement,
            }],
            "notes": (
                "Deterministically updated a stale exact dictionary test expectation "
                "to include the requirement-backed additive return field(s): "
                + ", ".join(map(str, extra_keys))
            ),
        }

    @staticmethod
    def _pytest_status(pytest_result: dict[str, Any]) -> str:
        """Create a compact factual pytest status for the engineering report."""
        if not isinstance(pytest_result, dict):
            return "Not run"

        message = str(pytest_result.get("message", "") or "").strip()
        if pytest_result.get("ok"):
            output = "\n".join(
                [
                    str(pytest_result.get("stdout", "") or ""),
                    str(pytest_result.get("stderr", "") or ""),
                ]
            ).strip()
            match = re.findall(r"([^\n]*\bpassed\b[^\n]*)", output, flags=re.I)
            return (match[-1].strip() if match else message) or "Passed"

        error = str(pytest_result.get("error", "") or "").strip()
        if pytest_result.get("error_type") == "infrastructure":
            return f"Infrastructure failure: {error or message or 'pytest unavailable'}"

        output = WorkflowEngine.failed_test_output(pytest_result, limit=700)
        summary_lines = [line.strip() for line in output.splitlines() if line.strip()]
        if summary_lines:
            return summary_lines[-1]
        return error or message or "Failed"

    @staticmethod
    def _verification_label(entry: dict[str, Any]) -> str:
        return str(entry.get("label", "verification"))

    def record_verification(
        self,
        state: WorkflowState,
        label: str,
        verification: dict[str, Any],
    ) -> None:
        """Persist a compact verification history for before/after reporting."""
        state.verification_history.append({
            "label": label,
            "ok": bool(verification.get("ok")),
            "quality": list(verification.get("quality", ())),
            "compile": verification.get("compile", {}),
            "pytest": verification.get("pytest", {}),
        })

    def build_engineering_report(self, state: WorkflowState, requirement: str) -> dict[str, Any]:
        """Build a deterministic engineering report from observed workflow state.

        This report is deliberately derived from execution records rather than
        model-generated claims, so files/tests/repair/sandbox facts remain truthful.
        """
        before = None
        after = None
        for item in state.verification_history:
            label = self._verification_label(item)
            if label == "before_repair":
                before = item
            elif label.startswith("after_repair_") and item.get("ok"):
                after = item

        if after is None:
            for item in reversed(state.verification_history):
                if item.get("ok") and self._verification_label(item).startswith("after_"):
                    after = item
                    break

        final_tests = state.verification.get("final_tests")
        if after is None and isinstance(final_tests, dict):
            after = {
                "label": "final_tests",
                "ok": bool(final_tests.get("ok")),
                "quality": list(final_tests.get("quality", ())),
                "compile": final_tests.get("compile", {}),
                "pytest": final_tests.get("pytest", {}),
            }

        repair_rows = []
        for item in state.repair_history:
            repair_rows.append({
                "attempt": item.get("attempt"),
                "changed_files": item.get("changed_files", []),
                "status": item.get("status", "unknown"),
                "before_quality": item.get("before_quality"),
                "after_quality": item.get("after_quality"),
                "rollback_verified": item.get("rollback_verified"),
            })

        sandbox = state.verification.get("sandbox", {})

        report = {
            "requirement": requirement[:4000],
            "changed_files": sorted(set(state.changed_files)),
            "tests_before_repair": before,
            "tests_after_repair": after,
            "verification_history": list(state.verification_history),
            "repair_attempts": state.repair_attempts,
            "repair_history": repair_rows,
            "sandbox_verification": sandbox,
            "overall_verified": bool(
                not state.aborted
                and isinstance(sandbox, dict)
                and sandbox.get("ok") is True
                and isinstance(final_tests, dict)
                and final_tests.get("ok") is True
            ),
        }

        state.engineering_report = report
        state.verification["engineering_report"] = report
        return report

    @staticmethod
    def render_engineering_report(report: dict[str, Any]) -> str:
        """Render the deterministic engineering report as compact Markdown."""
        changed = report.get("changed_files") or []
        before = report.get("tests_before_repair") or {}
        after = report.get("tests_after_repair") or {}
        sandbox = report.get("sandbox_verification") or {}
        repairs = report.get("repair_history") or []

        lines = [
            "## Engineering Verification Report",
            "",
            "### Changed files",
        ]
        if changed:
            lines.extend(f"- `{path}`" for path in changed)
        else:
            lines.append("- None recorded")

        lines.extend([
            "",
            "### Verification before repair",
            f"- Result: **{'PASSED' if before.get('ok') else 'FAILED' if before else 'Not recorded'}**",
        ])
        if before:
            lines.append(f"- Pytest: {WorkflowEngine._pytest_status(before.get('pytest', {}))}")

        lines.extend([
            "",
            "### Verification after repair",
            f"- Result: **{'PASSED' if after.get('ok') else 'FAILED' if after else 'Not recorded'}**",
        ])
        if after:
            lines.append(f"- Pytest: {WorkflowEngine._pytest_status(after.get('pytest', {}))}")

        lines.extend([
            "",
            "### Repair attempts",
            f"- Total attempts: **{report.get('repair_attempts', 0)}**",
        ])
        if repairs:
            for item in repairs:
                status = str(item.get("status", "unknown")).replace("_", " ")
                files = ", ".join(f"`{x}`" for x in item.get("changed_files", [])) or "no files recorded"
                rollback = item.get("rollback_verified")
                suffix = " · rollback verified" if rollback is True else ""
                lines.append(f"- Attempt {item.get('attempt')}: **{status}** · {files}{suffix}")
        else:
            lines.append("- No repair was required.")

        lines.extend([
            "",
            "### Sandbox verification",
            f"- Result: **{'PASSED' if sandbox.get('ok') else 'FAILED' if sandbox else 'Not run'}**",
        ])
        if sandbox.get("mode"):
            lines.append(f"- Mode: `{sandbox.get('mode')}`")
        if sandbox.get("http_status") is not None:
            lines.append(f"- HTTP status: `{sandbox.get('http_status')}`")
        if sandbox.get("error"):
            lines.append(f"- Error: `{str(sandbox.get('error'))[:500]}`")

        lines.extend([
            "",
            f"### Overall result",
            f"**{'VERIFIED' if report.get('overall_verified') else 'NOT VERIFIED'}**",
        ])
        return "\n".join(lines)

    def verify_project(self, workspace: Path) -> dict[str, Any]:
        """Run the deterministic verification stack once and return both results."""
        compile_result = self.compile_project(workspace)
        pytest_result = self.run_pytest(workspace)
        return {
            "compile": compile_result,
            "pytest": pytest_result,
            "ok": compile_result["ok"] and pytest_result["ok"],
            "quality": self.verification_quality(compile_result, pytest_result),
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
            env = os.environ.copy()
            env.update({
                "PYTHONUNBUFFERED": "1",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            })

            # DevPilot may run in an environment where pytest is not installed
            # even though the uploaded project contains pytest tests. Check the
            # same Python interpreter that will execute the tests, and install
            # pytest only when it is missing. This keeps the project itself
            # unchanged and avoids the misleading "No module named pytest"
            # testing failure.
            python_executable = sys.executable or "python"
            pytest_check = subprocess.run(
                [python_executable, "-c", "import pytest"],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=15,
                env=env,
            )

            if pytest_check.returncode != 0:
                install = subprocess.run(
                    [
                        python_executable,
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--no-input",
                        "pytest",
                    ],
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env,
                )

                if install.returncode != 0:
                    return {
                        "ok": False,
                        "mode": "pytest",
                        "error_type": "infrastructure",
                        "error": "pytest is not installed and could not be installed automatically.",
                        "install_stdout": (install.stdout or "")[-3000:],
                        "install_stderr": (install.stderr or "")[-3000:],
                    }

            proc = subprocess.run(
                [python_executable, "-m", "pytest", "-q"],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=40,
                env=env,
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
            # Use the exact Python interpreter running DevPilot. Calling the
            # generic `python` command can resolve to a different environment
            # that does not contain Streamlit, causing the child process to
            # exit immediately with a misleading sandbox failure.
            python_executable = sys.executable or "python"

            streamlit_check = subprocess.run(
                [python_executable, "-c", "import streamlit"],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=15,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            )

            if streamlit_check.returncode != 0:
                return {
                    "ok": False,
                    "error_type": "infrastructure",
                    "error": "Streamlit is not available in the DevPilot Python environment.",
                    "startup_log": (streamlit_check.stderr or streamlit_check.stdout or "")[-6000:],
                }

            proc = subprocess.Popen(
                [
                    python_executable,
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
                env={**os.environ, "PYTHONUNBUFFERED": "1", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
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
        state = WorkflowState(requirement=requirement[:5000])

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

            for path in applied["changed_files"]:
                if path not in state.changed_files:
                    state.changed_files.append(path)

            self.event(state, "Implementation applied", True, applied)
            self.update(state, "implementation", "done", f"Changed: {', '.join(applied['changed_files'])}")

            # Refresh code context after implementation. The original implementation
            # generated tests from pre-edit snippets, which could create tests for code
            # that no longer existed.
            current_snippets = []
            for path in relevant[:8]:
                try:
                    current_snippets.append(
                        f"FILE: {path}\n{self.read_text(state.workspace, path)[:7000]}"
                    )
                except Exception:
                    continue

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
- Base tests ONLY on the current post-implementation code supplied below.
- Keep tests small and deterministic.
- Prefer testing pure functions/classes instead of importing a Streamlit UI.
- Do not add dependencies.
- If the requirement cannot be safely tested without launching the application, return an empty tests list.
- Never overwrite an existing test file.
""",
                f"Requirement:\n{requirement[:5000]}\n\nCurrent post-implementation code:\n{chr(10).join(current_snippets)[:MAX_CONTEXT_CHARS]}",
                state,
            )

            tests = test_plan.get("tests", [])
            generated_test_file = None
            if isinstance(tests, list) and tests:
                test_code = "import pytest\n\n"
                for item in tests[:4]:
                    if not isinstance(item, dict):
                        continue
                    code = item.get("code")
                    if isinstance(code, str) and code.strip():
                        code = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.I)
                        code = re.sub(r"\s*```$", "", code).strip()
                        test_code += code + "\n\n"

                if test_code.strip() != "import pytest":
                    requested_test_file = str(test_plan.get("test_file", "tests/test_requirement.py"))
                    if not requested_test_file.startswith("tests/"):
                        requested_test_file = "tests/test_requirement.py"

                    target = self.safe_path(state.workspace, requested_test_file)
                    if target.exists():
                        stem = target.stem
                        suffix = target.suffix or ".py"
                        index = 2
                        while target.exists():
                            target = target.with_name(f"{stem}_generated_{index}{suffix}")
                            index += 1
                        requested_test_file = target.relative_to(state.workspace).as_posix()

                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(test_code, encoding="utf-8")
                    generated_test_file = requested_test_file
                    if generated_test_file not in state.changed_files:
                        state.changed_files.append(generated_test_file)
                    state.verification["test_file"] = generated_test_file
                    self.event(state, "Requirement tests generated", True, {"test_file": generated_test_file})
            else:
                self.event(state, "Requirement tests generated", True, {"test_file": None, "message": "No safe deterministic requirement test was generated."})

            self.update(state, "test_generation", "done", "Focused tests prepared")

            # 6-8 Verification / failure analysis / autonomous repair loop
            # The loop is intentionally closed: test -> diagnose -> patch -> retest.
            # Each repair is reversible, and a repair that makes verification worse
            # is rolled back before another strategy is attempted.
            verification = self.verify_project(state.workspace)
            state.verification["initial_verification"] = verification
            self.record_verification(state, "before_repair", verification)

            for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
                compile_result = verification["compile"]
                pytest_result = verification["pytest"]
                testing_ok = verification["ok"]

                self.update(
                    state,
                    "testing",
                    "active",
                    f"Verification cycle {attempt + 1}"
                )
                self.event(
                    state,
                    f"Testing cycle {attempt + 1}",
                    testing_ok,
                    {
                        "compile": compile_result,
                        "pytest": pytest_result,
                        "quality": verification["quality"],
                    },
                )

                if testing_ok:
                    state.verification["final_tests"] = verification
                    self.update(state, "testing", "done", "Compile and tests passed")
                    self.update(state, "failure_analysis", "skipped", "No test failures")
                    if state.repair_attempts:
                        self.update(
                            state,
                            "repair",
                            "done",
                            f"Closed-loop repair succeeded after {state.repair_attempts} attempt(s)",
                        )
                    else:
                        self.update(state, "repair", "skipped", "No repair required")
                    break

                state.verification["latest_failure"] = {
                    "compile": compile_result,
                    "pytest": pytest_result,
                    "quality": verification["quality"],
                }

                self.update(state, "testing", "error", "Verification tests failed")

                # No third diagnosis is needed when the bounded repair budget is
                # exhausted. This keeps the loop deterministic and avoids spending
                # another model call when there is no repair slot left.
                if attempt >= MAX_REPAIR_ATTEMPTS:
                    self.fail(state, "Maximum repair attempts reached.", "repair")
                    return self._finish(state)

                self.update(state, "failure_analysis", "active", "Diagnosing the observed failure")

                failure_text = self.failed_test_output(pytest_result)
                failure_payload = {
                    "compile": compile_result,
                    "pytest": pytest_result,
                    "failure_output": failure_text,
                    "repair_history": state.repair_history[-3:],
                }
                analysis = self.llm_json(
                    """
You are the failure-analysis specialist inside a closed-loop software repair agent.
Classify the CURRENT verification failure and determine whether a focused repair is safe.
Return JSON only:
{
  "type": "project" or "infrastructure",
  "cause": "brief concrete cause based on the observed output",
  "repair_needed": true or false,
  "target_files": ["file.py"],
  "strategy": "brief repair strategy"
}
Rules:
- Read the exact pytest/compile output, including expected vs actual values.
- If an existing test asserts an exact old return shape and the requested requirement intentionally adds a field or changes that contract, classify it as a project/test-contract failure and set repair_needed=true.
- Do NOT call an intentional requirement change an infrastructure failure.
- Mark infrastructure only for environment/tooling failures (missing runtime, dependency installation failure, OS/process failure, etc.).
- target_files must contain only files that are clearly implicated by the failure output or provided project context.
- Never invent a failure cause that is not supported by the supplied evidence.
- Use repair_needed=false when the evidence does not support a safe focused code/test repair.
""",
                    json.dumps(failure_payload)[:MAX_CONTEXT_CHARS],
                    state,
                )

                self.event(state, "Failure analyzed", True, analysis)

                if analysis.get("type") == "infrastructure":
                    self.fail(
                        state,
                        str(analysis.get("cause", "Verification environment failure")),
                        "failure_analysis",
                    )
                    return self._finish(state)

                if analysis.get("repair_needed") is not True:
                    self.fail(
                        state,
                        str(analysis.get("cause", "Failure could not be safely repaired.")),
                        "failure_analysis",
                    )
                    return self._finish(state)

                self.update(
                    state,
                    "failure_analysis",
                    "done",
                    str(analysis.get("cause", "Project failure"))[:120],
                )

                # Narrow the next repair's context to implicated files when possible.
                target_files = analysis.get("target_files", [])
                if not isinstance(target_files, list):
                    target_files = []
                target_files = [str(x) for x in target_files if isinstance(x, str)]
                target_files = [
                    path for path in target_files
                    if path in relevant
                    or path.startswith("tests/")
                    or path.endswith((".py", ".js", ".ts", ".cs"))
                ]
                hinted_files = self.likely_failure_files(pytest_result, relevant)
                focused = []
                for path in target_files + hinted_files + relevant:
                    if path not in focused:
                        focused.append(path)
                focused = focused[:8]

                self.update(
                    state,
                    "repair",
                    "active",
                    f"Repair attempt {attempt + 1} of {MAX_REPAIR_ATTEMPTS}",
                )

                changed_files_before_attempt = set(state.changed_files)

                current_code = []
                for path in focused:
                    try:
                        current_code.append(
                            f"FILE: {path}\n{self.read_text(state.workspace, path)[:7000]}"
                        )
                    except Exception:
                        pass

                # A common safe failure mode is an old exact-dict test contract after
                # an intentionally additive return field. Handle that narrow pattern
                # deterministically before asking the LLM to patch tests. This keeps
                # the agent reliable while remaining strictly requirement-gated.
                deterministic_repair = self.deterministic_contract_test_repair(
                    state.workspace,
                    failure_text,
                    requirement,
                )

                if deterministic_repair:
                    repair = deterministic_repair
                    self.event(
                        state,
                        "Deterministic test-contract repair selected",
                        True,
                        {
                            "changed_area": "stale exact dictionary assertion",
                            "notes": repair.get("notes", ""),
                        },
                    )
                else:
                    repair = self.llm_json(
                        """
You are the repair specialist for a closed-loop software development agent.
Return focused edits only:
{
  "edits": [
    {"path":"file.py","search":"exact current text","replace":"replacement"}
  ],
  "notes":"brief"
}
Rules:
- Maximum 3 edits.
- Use exact current text from the supplied code.
- Work only on the implicated failure and requested requirement.
- Do not rewrite whole files.
- Do not modify secret files.
- Do not create dependency-shadowing files.
- Read the latest pytest/compile evidence before deciding what to change.
- Preserve behavior outside the requirement.
- If the implementation intentionally adds a field to a returned dictionary and an existing test uses exact dictionary equality for the old contract, update that test to assert the new required field and keep the existing assertions intact.
- Do NOT remove a newly required field merely to make an old test pass.
- Never weaken tests with `assert True`, unconditional skips, or by deleting the failing assertion.
- Keep completed/open behavior and unrelated tests unchanged.
- Prefer the smallest patch that addresses the diagnosed cause.
""",
                        f"Failure analysis:\n{json.dumps(analysis)[:7000]}\n\nLatest failure evidence:\n{failure_text[:7000]}\n\nCurrent focused code:\n{chr(10).join(current_code)[:MAX_CONTEXT_CHARS]}",
                        state,
                    )

                repair_edits = self.normalize_edits(repair)
                repair_result = self.apply_edits(state.workspace, repair_edits)
                state.repair_attempts += 1

                if not repair_result["changed_files"]:
                    self.fail(state, "Repair stage produced no valid matching edits.", "repair")
                    return self._finish(state)

                for path in repair_result["changed_files"]:
                    if path not in state.changed_files:
                        state.changed_files.append(path)

                state.verification["repair_candidate"] = {
                    "attempt": state.repair_attempts,
                    "changed_files": repair_result["changed_files"],
                    "notes": str(repair.get("notes", ""))[:1000],
                }
                self.event(
                    state,
                    f"Repair attempt {state.repair_attempts} applied",
                    True,
                    {
                        "changed_files": repair_result["changed_files"],
                        "count": repair_result["count"],
                    },
                )

                # Immediate retest closes the agentic loop and gives the next
                # diagnosis concrete evidence from the exact repair just made.
                post_repair = self.verify_project(state.workspace)
                self.record_verification(
                    state,
                    f"after_repair_{state.repair_attempts}",
                    post_repair,
                )
                self.event(
                    state,
                    f"Retest after repair {state.repair_attempts}",
                    post_repair["ok"],
                    {
                        "compile": post_repair["compile"],
                        "pytest": post_repair["pytest"],
                        "quality": post_repair["quality"],
                    },
                )

                before_quality = verification["quality"]
                after_quality = post_repair["quality"]

                if post_repair["ok"]:
                    state.repair_history.append({
                        "attempt": state.repair_attempts,
                        "changed_files": repair_result["changed_files"],
                        "status": "verified",
                        "before_quality": before_quality,
                        "after_quality": after_quality,
                    })
                    state.verification["final_tests"] = post_repair
                    self.update(
                        state,
                        "repair",
                        "done",
                        f"Repair {state.repair_attempts} passed immediate retest",
                    )
                    verification = post_repair
                    continue

                # Never silently keep a repair that made deterministic verification
                # worse. Restore exactly what that repair changed and confirm the
                # rollback itself is healthy before another attempt.
                if after_quality < before_quality:
                    rollback = self.rollback_changes(
                        state.workspace,
                        repair_result.get("rollback_state", []),
                    )
                    self.event(
                        state,
                        f"Repair {state.repair_attempts} rolled back",
                        rollback["ok"],
                        rollback,
                    )

                    restored_verification = self.verify_project(state.workspace)
                    self.record_verification(
                        state,
                        f"after_rollback_{state.repair_attempts}",
                        restored_verification,
                    )
                    rollback_ok = rollback["ok"] and (
                        restored_verification["quality"] == before_quality
                    )
                    self.event(
                        state,
                        "Rollback verification",
                        rollback_ok,
                        {
                            "quality": restored_verification["quality"],
                            "expected_quality": before_quality,
                            "compile": restored_verification["compile"],
                            "pytest": restored_verification["pytest"],
                        },
                    )

                    state.repair_history.append({
                        "attempt": state.repair_attempts,
                        "changed_files": repair_result["changed_files"],
                        "status": "rolled_back",
                        "before_quality": before_quality,
                        "after_quality": after_quality,
                        "rollback_verified": rollback_ok,
                    })

                    if not rollback_ok:
                        self.fail(
                            state,
                            "A repair worsened verification and safe rollback could not be confirmed.",
                            "repair",
                        )
                        return self._finish(state)

                    state.changed_files = sorted(changed_files_before_attempt)
                    verification = restored_verification
                else:
                    state.repair_history.append({
                        "attempt": state.repair_attempts,
                        "changed_files": repair_result["changed_files"],
                        "status": "failed_retest",
                        "before_quality": before_quality,
                        "after_quality": after_quality,
                    })
                    verification = post_repair

                self.update(
                    state,
                    "repair",
                    "done",
                    f"Repair {state.repair_attempts} did not verify; continuing diagnosis",
                )

                # Continue to the next loop iteration with the exact post-repair
                # (or post-rollback) verification evidence.
                state.verification["repair_history"] = state.repair_history[-5:]
            else:
                self.fail(state, "Verification loop ended without a terminal result.", "testing")
                return self._finish(state)

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
                self.build_engineering_report(state, requirement)
                self.fail(state, f"Sandbox verification failed: {smoke.get('error', 'unknown error')}", "sandbox")
                return self._finish(state)

            self.update(state, "sandbox", "done", "Application responded successfully")

            # 10. Final report
            engineering_report = self.build_engineering_report(state, requirement)
            engineering_markdown = self.render_engineering_report(engineering_report)

            self.update(state, "final_report", "active", "Preparing verified engineering report")
            final = self.llm_json(
                """
You are the final-report specialist.
Return JSON only:
{
  "report": "markdown report",
  "verified": true or false
}
Never claim verification that is not present in the supplied results.
Mention changed files, tests before/after repair, every repair attempt, rollback events if any, and sandbox status.
The deterministic engineering report supplied in the input is the source of truth.
""",
                json.dumps({
                    "requirement": requirement[:4000],
                    "engineering_report": engineering_report,
                    "verification": state.verification,
                    "events": state.events[-12:],
                })[:MAX_CONTEXT_CHARS],
                state,
            )

            if final.get("verified") is True:
                narrative = str(final.get("report", "Development run verified."))
                state.final_report = narrative.rstrip() + "\n\n" + engineering_markdown
                self.update(state, "final_report", "done", "Verified engineering report generated")
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
        # Feature 3: always preserve a factual engineering report, even when the
        # workflow stops early. The report is derived from observed state and never
        # asks the model to invent execution facts.
        report = state.engineering_report
        if not report:
            report = self.build_engineering_report(state, state.requirement)

        engineering_markdown = self.render_engineering_report(report)

        if state.aborted:
            if not state.final_report:
                state.final_report = (
                    "## Result\n\n"
                    "Development run stopped safely.\n\n"
                    f"**Reason:** {state.abort_reason}"
                )
        elif not state.final_report:
            state.final_report = "## Result\n\nDevelopment run completed."

        if "## Engineering Verification Report" not in state.final_report:
            state.final_report = state.final_report.rstrip() + "\n\n" + engineering_markdown

        if self.callback:
            self.callback(state)

        return state
