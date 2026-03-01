from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import modal

from config import settings
from fix_generator import FixResult

logger = logging.getLogger("prism.sandbox")

# ---------------------------------------------------------------------------
# Modal image — built once and cached by Modal between runs
# ---------------------------------------------------------------------------

SANDBOX_IMAGE = (
    modal.Image.debian_slim()
    .run_commands(
        "apt-get update -qq && apt-get install -y git curl openjdk-17-jre-headless -qq",
        # ktlint standalone binary for Kotlin/Java linting (no Android SDK needed)
        "curl -sSL https://github.com/pinterest/ktlint/releases/download/1.3.1/ktlint "
        "-o /usr/local/bin/ktlint && chmod +x /usr/local/bin/ktlint",
    )
    .pip_install(["pytest", "pytest-cov", "ruff"])
)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    success: bool
    lint_output: str
    lint_passed: bool
    test_output: str
    test_passed: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_in_sandbox(repo_url: str, fix: FixResult) -> SandboxResult:
    """
    Spin up a Modal gVisor sandbox, apply the patch, run language-appropriate checks.
    Always terminates the sandbox, even on failure.
    """
    app = modal.App.lookup(settings.modal_app_name, create_if_missing=True)

    sandbox = modal.Sandbox.create(
        "sleep", "infinity",
        app=app,
        image=SANDBOX_IMAGE,
        timeout=300,
    )

    try:
        return _run_pipeline(sandbox, repo_url, fix)
    except Exception as exc:
        logger.exception("Sandbox pipeline raised an unexpected error")
        return SandboxResult(
            success=False,
            lint_output="",
            lint_passed=False,
            test_output="",
            test_passed=False,
            error=str(exc),
        )
    finally:
        try:
            sandbox.terminate()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_language(fix: FixResult) -> str:
    """Detect primary language from patched file extensions."""
    extensions = {Path(fp).suffix.lower() for fp in fix.patches}
    if extensions & {".kt", ".java"}:
        return "kotlin"
    if extensions & {".py"}:
        return "python"
    if extensions & {".js", ".ts", ".jsx", ".tsx"}:
        return "javascript"
    return "unknown"


def _exec(sandbox: modal.Sandbox, cmd: str) -> tuple[str, int]:
    """Run a bash command and return (combined stdout+stderr output, exit_code)."""
    proc = sandbox.exec("bash", "-c", cmd)
    output = proc.stdout.read()
    exit_code = proc.wait()
    return output, exit_code


def _run_pipeline(sandbox: modal.Sandbox, repo_url: str, fix: FixResult) -> SandboxResult:
    # Step 1: Clone the repository
    clone_out, clone_code = _exec(
        sandbox, f"git clone --depth 1 {repo_url} /workspace/repo 2>&1"
    )
    if clone_code != 0:
        return SandboxResult(
            success=False,
            lint_output="",
            lint_passed=False,
            test_output="",
            test_passed=False,
            error=f"git clone failed (exit {clone_code}):\n{clone_out}",
        )
    logger.debug("Clone succeeded")

    # Step 2: Write patched files into the sandbox
    for relative_path, new_content in fix.patches.items():
        container_path = f"/workspace/repo/{relative_path}"
        _exec(sandbox, f"mkdir -p $(dirname '{container_path}')")
        with sandbox.open(container_path, "w") as f:
            f.write(new_content)
    logger.debug("Wrote %d patched file(s)", len(fix.patches))

    # Step 3: Route to language-specific validation
    lang = _detect_language(fix)
    logger.info("Detected language: %s", lang)

    if lang == "kotlin":
        return _validate_kotlin(sandbox, fix)
    elif lang == "python":
        return _validate_python(sandbox)
    else:
        # JS/TS or unknown — no validator configured, treat as pass
        logger.info("No validator for language '%s' — skipping checks", lang)
        return SandboxResult(
            success=True,
            lint_output="(skipped — no validator for this language)",
            lint_passed=True,
            test_output="(skipped — no validator for this language)",
            test_passed=True,
        )


def _validate_python(sandbox: modal.Sandbox) -> SandboxResult:
    # install dependencies in the sandbox cause it'll need to test the proposed code on the clone of the repo in it
    _exec(
        sandbox,
        "cd /workspace/repo && "
        "(pip install -r requirements.txt -q 2>&1 || pip install -e . -q 2>&1 || true)",
    )

    # Lint with ruff
    lint_output, lint_code = _exec(sandbox, "cd /workspace/repo && ruff check . 2>&1")
    lint_passed = lint_code == 0
    logger.info("ruff: %s", "PASS" if lint_passed else "FAIL")

    # Run pytest (exit code 5 = no tests collected, still a pass)
    test_output, test_code = _exec(
        sandbox, "cd /workspace/repo && pytest --tb=short -q 2>&1"
    )
    test_passed = test_code in (0, 5)
    logger.info("pytest: %s (exit code %d)", "PASS" if test_passed else "FAIL", test_code)

    return SandboxResult(
        success=lint_passed and test_passed,
        lint_output=lint_output,
        lint_passed=lint_passed,
        test_output=test_output,
        test_passed=test_passed,
    )


def _validate_kotlin(sandbox: modal.Sandbox, fix: FixResult) -> SandboxResult:
    """
    Lint only the patched Kotlin/Java files with ktlint.
    Linting only patched files avoids pre-existing violations in the repo blocking the PR.
    Full Android compilation is skipped — it requires the Android SDK (several GB).
    """
    kt_files = [fp for fp in fix.patches if fp.endswith((".kt", ".kts"))]
    if not kt_files:
        return SandboxResult(
            success=True,
            lint_output="No .kt/.kts files to lint",
            lint_passed=True,
            test_output="No Kotlin files patched",
            test_passed=True,
        )

    file_args = " ".join(f'"/workspace/repo/{f}"' for f in kt_files)
    lint_output, lint_code = _exec(sandbox, f"ktlint {file_args} 2>&1")
    lint_passed = lint_code == 0
    logger.info("ktlint: %s\n%s", "PASS" if lint_passed else "FAIL", lint_output)

    test_output = "Android unit tests skipped — Android SDK not available in sandbox. ktlint verified patch syntax."
    test_passed = True

    return SandboxResult(
        success=lint_passed,
        lint_output=lint_output,
        lint_passed=lint_passed,
        test_output=test_output,
        test_passed=test_passed,
    )
