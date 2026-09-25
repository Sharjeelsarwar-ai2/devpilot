from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any


ALLOWED_ACTIONS = {
    "open",
    "click",
    "fill",
    "select",
    "press",
    "assert_text",
    "assert_visible",
    "assert_url",
}

MAX_UI_TESTS = 5
MAX_UI_STEPS = 20
ACTION_TIMEOUT_MS = 7000
NAVIGATION_TIMEOUT_MS = 10000
BROWSER_START_TIMEOUT_SECONDS = 15
BROWSER_INSTALL_TIMEOUT_SECONDS = 120



def validate_ui_test_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate an LLM-produced UI plan before any browser action is executed."""
    if not isinstance(plan, dict):
        return {"ok": False, "error": "UI test plan is not an object."}

    tests = plan.get("tests", [])
    if tests in (None, []):
        return {"ok": True, "tests": [], "message": str(plan.get("message", "No UI tests generated."))[:500]}

    if not isinstance(tests, list):
        return {"ok": False, "error": "UI test plan must contain a tests list."}

    if len(tests) > MAX_UI_TESTS:
        return {"ok": False, "error": f"UI test plan exceeds the {MAX_UI_TESTS}-test limit."}

    normalized: list[dict[str, Any]] = []

    for index, test in enumerate(tests, start=1):
        if not isinstance(test, dict):
            return {"ok": False, "error": f"UI test {index} is not an object."}

        name = str(test.get("name", "")).strip()
        steps = test.get("steps")
        if not name or not isinstance(steps, list) or not steps:
            return {"ok": False, "error": f"UI test {index} must have a name and non-empty steps."}

        if len(steps) > MAX_UI_STEPS:
            return {"ok": False, "error": f"UI test {name!r} exceeds the {MAX_UI_STEPS}-step limit."}

        normalized_steps: list[dict[str, Any]] = []
        for step_index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                return {"ok": False, "error": f"UI test {name!r}, step {step_index} is not an object."}

            action = str(step.get("action", "")).strip().lower()
            if action not in ALLOWED_ACTIONS:
                return {
                    "ok": False,
                    "error": f"UI test {name!r}, step {step_index} uses unsupported action {action!r}.",
                }

            item = {"action": action}

            if action == "open":
                path = str(step.get("path", "/")).strip()
                if not path.startswith("/") or path.startswith("//"):
                    return {"ok": False, "error": f"UI test {name!r} has an unsafe open path."}
                item["path"] = path

            elif action in {"click", "fill", "select", "assert_text", "assert_visible"}:
                target = str(step.get("target", "")).strip()
                if not target:
                    return {"ok": False, "error": f"UI test {name!r}, step {step_index} needs a target."}
                item["target"] = target

                if action == "fill":
                    value = step.get("value")
                    if not isinstance(value, str):
                        return {"ok": False, "error": f"UI test {name!r}, fill step needs a string value."}
                    item["value"] = value

                if action == "select":
                    value = step.get("value")
                    if not isinstance(value, str) or not value.strip():
                        return {"ok": False, "error": f"UI test {name!r}, select step needs a non-empty string value."}
                    item["value"] = value

            elif action == "press":
                key = str(step.get("key", step.get("value", ""))).strip()
                if not key:
                    return {"ok": False, "error": f"UI test {name!r}, press step needs a key."}
                item["key"] = key

            elif action == "assert_url":
                expected = str(step.get("value", "")).strip()
                if not expected:
                    return {"ok": False, "error": f"UI test {name!r}, assert_url needs a value."}
                item["value"] = expected

            normalized_steps.append(item)

        normalized.append({"name": name[:200], "steps": normalized_steps})

    return {"ok": True, "tests": normalized, "message": str(plan.get("message", ""))[:500]}



def _safe_base_url(base_url: str) -> tuple[bool, str]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        return False, "Browser testing requires an HTTP(S) local application URL."

    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return False, "Browser testing is restricted to localhost/127.0.0.1."

    return True, base_url.rstrip("/")



def _text_locator(page, text: str):
    """Prefer semantic text targeting, with exact text first and partial text second."""
    exact = page.get_by_text(text, exact=True)
    if exact.count() > 0:
        return exact.first
    partial = page.get_by_text(text, exact=False)
    if partial.count() > 0:
        return partial.first
    return None



def _click(page, target: str) -> None:
    # Prefer accessible buttons/links before falling back to visible text.
    for role in ("button", "link"):
        locator = page.get_by_role(role, name=target, exact=True)
        if locator.count() > 0:
            locator.first.click(timeout=ACTION_TIMEOUT_MS)
            return

    locator = _text_locator(page, target)
    if locator is not None:
        locator.click(timeout=ACTION_TIMEOUT_MS)
        return

    raise RuntimeError(f"Could not find clickable UI target: {target}")



def _fill(page, target: str, value: str) -> None:
    label_locator = page.get_by_label(target, exact=True)
    if label_locator.count() > 0:
        label_locator.first.fill(value, timeout=ACTION_TIMEOUT_MS)
        return

    placeholder_locator = page.get_by_placeholder(target, exact=True)
    if placeholder_locator.count() > 0:
        placeholder_locator.first.fill(value, timeout=ACTION_TIMEOUT_MS)
        return

    raise RuntimeError(f"Could not find fillable UI field: {target}")



def _select(page, target: str, value: str) -> None:
    label_locator = page.get_by_label(target, exact=True)
    if label_locator.count() > 0:
        # Native <select> controls can use select_option directly.
        try:
            label_locator.first.select_option(label=value, timeout=ACTION_TIMEOUT_MS)
            return
        except Exception:
            label_locator.first.click(timeout=ACTION_TIMEOUT_MS)

    # Streamlit's selectbox is rendered as a combobox/listbox. After opening it,
    # target the visible option by its accessible name/text.
    option = page.get_by_role("option", name=value, exact=True)
    if option.count() > 0:
        option.first.click(timeout=ACTION_TIMEOUT_MS)
        return

    visible_value = _text_locator(page, value)
    if visible_value is not None:
        visible_value.click(timeout=ACTION_TIMEOUT_MS)
        return

    raise RuntimeError(f"Could not select {value!r} from UI field: {target}")



def _assert_text(page, value: str) -> None:
    locator = _text_locator(page, value)
    if locator is None:
        raise AssertionError(f"Expected visible text not found: {value}")
    locator.wait_for(state="visible", timeout=ACTION_TIMEOUT_MS)



def _assert_visible(page, target: str) -> None:
    locator = _text_locator(page, target)
    if locator is None:
        try:
            label = page.get_by_label(target, exact=True)
            if label.count() > 0:
                label.first.wait_for(state="visible", timeout=ACTION_TIMEOUT_MS)
                return
        except Exception:
            pass
        raise AssertionError(f"Expected visible UI target not found: {target}")
    locator.wait_for(state="visible", timeout=ACTION_TIMEOUT_MS)



def _execute_step(page, step: dict[str, Any], base_url: str) -> str:
    action = step["action"]

    if action == "open":
        path = step.get("path", "/")
        url = base_url + path
        page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
        page.wait_for_timeout(250)
        return f"Opened {path}"

    if action == "click":
        _click(page, step["target"])
        page.wait_for_timeout(250)
        return f"Clicked {step['target']}"

    if action == "fill":
        _fill(page, step["target"], step["value"])
        return f"Filled {step['target']}"

    if action == "select":
        _select(page, step["target"], step["value"])
        page.wait_for_timeout(250)
        return f"Selected {step['value']} in {step['target']}"

    if action == "press":
        page.keyboard.press(step["key"])
        page.wait_for_timeout(250)
        return f"Pressed {step['key']}"

    if action == "assert_text":
        _assert_text(page, step["target"])
        return f"Verified text: {step['target']}"

    if action == "assert_visible":
        _assert_visible(page, step["target"])
        return f"Verified visible: {step['target']}"

    if action == "assert_url":
        expected = step["value"]
        if expected not in page.url:
            raise AssertionError(f"Expected URL to contain {expected!r}, got {page.url!r}")
        return f"Verified URL contains {expected}"

    raise RuntimeError(f"Unsupported UI action: {action}")



def _browser_launch_mode() -> tuple[bool, str]:
    """Choose whether the Playwright browser should be visibly headed or headless.

    DEVPILOT_BROWSER_MODE can be:
      - headed: force a visible browser (local desktop only)
      - headless: force background browser testing
      - auto: headed on a local desktop, headless on CI/server environments
    """
    mode = os.getenv("DEVPILOT_BROWSER_MODE", "auto").strip().lower()
    if mode not in {"auto", "headed", "headless"}:
        mode = "auto"

    if mode == "headless":
        return True, "headless"

    if mode == "headed":
        # A headed browser needs a desktop/display. On Linux without DISPLAY,
        # safely fall back to headless rather than crashing the workflow.
        if os.name == "nt" or os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"):
            return False, "headed"
        return True, "headless_fallback_no_display"

    # Auto mode: visible on a local Windows/macOS/Linux desktop; background
    # in CI or server-only environments.
    if os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}:
        return True, "headless_ci"
    if os.name == "nt" or os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"):
        return False, "headed_auto"
    return True, "headless_server"


def _browser_hold_ms() -> int:
    """How long to keep a headed test page visible after each successful test."""
    raw = os.getenv("DEVPILOT_BROWSER_HOLD_MS", "1500").strip()
    try:
        return max(0, min(int(raw), 10000))
    except ValueError:
        return 1500


def _ensure_chromium(p) -> tuple[bool, str]:
    """Launch Chromium once; install its browser binary only if it is missing."""
    try:
        browser = p.chromium.launch(headless=True)
        browser.close()
        return True, "Chromium browser already available."
    except Exception as first_error:
        message = str(first_error)
        looks_like_missing_browser = "Executable doesn't exist" in message or "playwright install" in message
        if not looks_like_missing_browser:
            return False, message[:1200]

    auto_install = os.getenv("DEVPILOT_AUTO_INSTALL_PLAYWRIGHT", "1").strip().lower()
    if auto_install in {"0", "false", "no", "off"}:
        return False, "Chromium is not installed and automatic browser installation is disabled."

    try:
        install_env = {**os.environ}
        configured_browser_path = os.getenv("PLAYWRIGHT_BROWSERS_PATH")
        if configured_browser_path:
            install_env["PLAYWRIGHT_BROWSERS_PATH"] = configured_browser_path

        install = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            timeout=BROWSER_INSTALL_TIMEOUT_SECONDS,
            env=install_env,
        )
    except Exception as exc:
        return False, f"Could not install Chromium automatically: {exc}"

    if install.returncode != 0:
        output = (install.stderr or install.stdout or "").strip()
        return False, f"Chromium installation failed: {output[-1500:]}"

    try:
        browser = p.chromium.launch(headless=True)
        browser.close()
        return True, "Chromium installed and launched successfully."
    except Exception as exc:
        return False, f"Chromium remains unavailable after installation: {str(exc)[:1200]}"



def run_browser_ui_tests(
    base_url: str,
    plan: dict[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    """Execute a validated UI test plan against a localhost application."""
    validation = validate_ui_test_plan(plan)
    if not validation.get("ok"):
        return {
            "ok": False,
            "mode": "playwright",
            "error_type": "configuration",
            "error": validation.get("error", "Invalid UI test plan."),
        }

    tests = validation.get("tests", [])
    if not tests:
        return {
            "ok": True,
            "mode": "playwright_skipped",
            "message": validation.get("message") or "No safe UI tests were generated.",
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "results": [],
        }

    allowed, normalized_url = _safe_base_url(base_url)
    if not allowed:
        return {
            "ok": False,
            "mode": "playwright",
            "error_type": "security",
            "error": normalized_url,
        }

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        auto_install = os.getenv("DEVPILOT_AUTO_INSTALL_PLAYWRIGHT", "1").strip().lower()
        if auto_install not in {"0", "false", "no", "off"}:
            try:
                install = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "playwright"],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                if install.returncode == 0:
                    from playwright.sync_api import sync_playwright
                else:
                    output = (install.stderr or install.stdout or "").strip()
                    return {
                        "ok": False,
                        "mode": "playwright",
                        "error_type": "infrastructure",
                        "error": f"Playwright is not installed and automatic installation failed: {output[-1500:]}",
                    }
            except Exception as install_exc:
                return {
                    "ok": False,
                    "mode": "playwright",
                    "error_type": "infrastructure",
                    "error": f"Playwright is unavailable and could not be installed automatically: {install_exc}",
                }
        else:
            return {
                "ok": False,
                "mode": "playwright",
                "error_type": "infrastructure",
                "error": f"Playwright is not installed: {exc}",
            }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser_ok, browser_message = _ensure_chromium(p)
        if not browser_ok:
            return {
                "ok": False,
                "mode": "playwright",
                "error_type": "infrastructure",
                "error": browser_message,
            }

        headless, browser_mode = _browser_launch_mode()
        try:
            browser = p.chromium.launch(headless=headless)
        except Exception as exc:
            # If headed mode cannot start (for example, a restricted desktop),
            # fall back to headless so functional verification is still attempted.
            if not headless:
                try:
                    browser = p.chromium.launch(headless=True)
                    headless = True
                    browser_mode = browser_mode + ":headless_fallback"
                except Exception as fallback_exc:
                    return {
                        "ok": False,
                        "mode": "playwright",
                        "error_type": "infrastructure",
                        "error": f"Chromium could not be launched headed or headless: {str(fallback_exc)[:1200]}",
                    }
            else:
                return {
                    "ok": False,
                    "mode": "playwright",
                    "error_type": "infrastructure",
                    "error": f"Chromium could not be launched: {str(exc)[:1200]}",
                }

        try:
            context = browser.new_context(
                accept_downloads=False,
                ignore_https_errors=False,
                viewport={"width": 1440, "height": 1000},
            )

            for index, test in enumerate(tests, start=1):
                try:
                    page = context.new_page()
                except Exception as exc:
                    return {
                        "ok": False,
                        "mode": "playwright",
                        "error_type": "infrastructure",
                        "error": f"Could not create browser page: {str(exc)[:1200]}",
                        "tests_run": index - 1,
                        "tests_passed": sum(1 for item in results if item.get("ok")),
                        "tests_failed": 1,
                        "results": results,
                    }

                page.set_default_timeout(ACTION_TIMEOUT_MS)
                page.set_default_navigation_timeout(NAVIGATION_TIMEOUT_MS)

                console_errors: list[str] = []
                page_errors: list[str] = []
                page.on("console", lambda msg, bucket=console_errors: bucket.append(msg.text) if msg.type == "error" else None)
                page.on("pageerror", lambda exc, bucket=page_errors: bucket.append(str(exc)))

                test_result: dict[str, Any] = {
                    "name": test["name"],
                    "ok": False,
                    "steps_run": 0,
                    "steps_total": len(test["steps"]),
                    "error": None,
                    "failed_step": None,
                    "screenshot": None,
                    "console_errors": [],
                    "page_errors": [],
                }

                try:
                    # Every test gets an explicit local root visit first. A later
                    # open step can navigate within the same localhost origin only.
                    page.goto(normalized_url + "/", wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
                    page.wait_for_timeout(250)

                    for step_index, step in enumerate(test["steps"], start=1):
                        test_result["failed_step"] = step_index
                        _execute_step(page, step, normalized_url)
                        test_result["steps_run"] = step_index

                    test_result["failed_step"] = None
                    test_result["ok"] = True

                except Exception as exc:
                    screenshot = artifacts_dir / f"ui_failure_{index}.png"
                    try:
                        page.screenshot(path=str(screenshot), full_page=True)
                        test_result["screenshot"] = str(screenshot)
                    except Exception:
                        pass
                    test_result["error"] = str(exc)[:2000]
                    test_result["console_errors"] = console_errors[-10:]
                    test_result["page_errors"] = page_errors[-10:]

                finally:
                    if test_result["ok"]:
                        test_result["console_errors"] = console_errors[-10:]
                        test_result["page_errors"] = page_errors[-10:]
                        if not headless:
                            page.wait_for_timeout(_browser_hold_ms())
                    results.append(test_result)
                    page.close()

            passed = sum(1 for item in results if item.get("ok"))
            failed = len(results) - passed
            return {
                "ok": failed == 0,
                "mode": "playwright",
                "browser_mode": browser_mode,
                "message": browser_message,
                "tests_run": len(results),
                "tests_passed": passed,
                "tests_failed": failed,
                "results": results,
                "artifacts_dir": str(artifacts_dir),
            }

        finally:
            context.close()
            browser.close()
