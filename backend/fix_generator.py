from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
import anthropic
from .classifier import ClassificationResult
from .config import settings

logger = logging.getLogger("prism.fix_generator")

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

@dataclass
class CodeChunk:
    file_path: str       
    function_name: str
    source_text: str
    score: float         

@dataclass
class FixResult:
    patches: dict[str, str]      
    explanation: str             
    files_changed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.files_changed:
            self.files_changed = list(self.patches.keys())

PROPOSE_FIX_TOOL = {
    "name": "propose_fix",
    "description": (
        "Propose targeted code patches to address a bug or UX issue reported by a user. "
        "Return complete file content (not diffs) for each modified file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": (
                    "Clear explanation of the root cause and exactly what the fix does. "
                    "This will appear verbatim in the GitHub PR description."
                ),
            },
            "patches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Relative file path from the repo root.",
                        },
                        "patched_source": {
                            "type": "string",
                            "description": (
                                "Complete new content for this file. "
                                "Must be the entire file, not just the changed lines."
                            ),
                        },
                    },
                    "required": ["file_path", "patched_source"],
                },
                "description": "One entry per modified file.",
            },
        },
        "required": ["explanation", "patches"],
    },
}

FIX_SYSTEM = """
You are a senior software engineer fixing a bug or UX issue reported by a mobile app user.

Rules:
- Make the minimal change necessary to address the reported issue.
- Return the COMPLETE file content for each modified file, not a diff.
- Only modify files that are directly responsible for the reported problem.
- Always make your best-effort fix using the provided code context. Only call propose_fix
  with an empty patches array if the context contains zero relevant code whatsoever.
- Do not introduce new dependencies.
- Preserve all existing functionality unrelated to the fix.
""".strip()

# key logic via claude to propose a code fix 

def generate_fix(
    classification: ClassificationResult,
    code_chunks: list[CodeChunk],
    repo_url: str,
) -> FixResult | None:
    user_message = _build_context_message(classification, code_chunks, repo_url)

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=8192,
        system=FIX_SYSTEM,
        tools=[PROPOSE_FIX_TOOL],
        tool_choice={"type": "tool", "name": "propose_fix"},
        messages=[{"role": "user", "content": user_message}],
    )

    try:
        tool_use = next(b for b in response.content if b.type == "tool_use")
        args = tool_use.input
    except StopIteration:
        logger.error("Claude failed to return a valid tool call for propose_fix")
        return None

    if not args.get("patches"):
        logger.warning("Claude declined to generate patches: %s", args.get("explanation", ""))
        return None

    patches = {p["file_path"]: p["patched_source"] for p in args["patches"]}
    return FixResult(patches=patches, explanation=args["explanation"])


def refine_fix(
    original_fix: FixResult,
    lint_errors: str,
    classification: ClassificationResult,
    code_chunks: list[CodeChunk],
    repo_url: str,
) -> FixResult | None:
    lines: list[str] = []

    lines.append("## Original Review")
    lines.append(f"> {classification.review_text}")

    lines.append("\n## Lint Errors in Your Previous Patch")
    lines.append("Your previous patch failed lint validation with these errors. Fix them:")
    lines.append("```")
    lines.append(lint_errors.strip())
    lines.append("```")

    # Show the exact lines around each error so Claude can pinpoint the issue
    error_context = _extract_error_context(lint_errors, original_fix.patches)
    if error_context:
        lines.append("\n## Exact Location of Each Error (>>> marks the error line)")
        lines.append(error_context)

    lines.append("\n## Your Previous Patches (contain the errors above)")
    for file_path, content in original_fix.patches.items():
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        lines.append(f"\n### `{file_path}`")
        lines.append(f"```{ext}")
        lines.append(content)
        lines.append("```")

    lines.append("\n## Original Code Context")
    for chunk in code_chunks:
        ext = chunk.file_path.rsplit(".", 1)[-1] if "." in chunk.file_path else ""
        lines.append(f"\n### `{chunk.file_path}` — `{chunk.function_name}()`")
        lines.append(f"```{ext}")
        lines.append(chunk.source_text)
        lines.append("```")

    lines.append(f"\n## Repository\n{repo_url}")
    lines.append("\nReturn corrected complete file content with all syntax errors fixed.")

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=8192,
        system=FIX_SYSTEM,
        tools=[PROPOSE_FIX_TOOL],
        tool_choice={"type": "tool", "name": "propose_fix"},
        messages=[{"role": "user", "content": "\n".join(lines)}],
    )

    try:
        tool_use = next(b for b in response.content if b.type == "tool_use")
        args = tool_use.input
    except StopIteration:
        logger.error("Claude failed to return a valid tool call for refine_fix")
        return None

    if not args.get("patches"):
        logger.warning("Claude declined to generate refined patches")
        return None

    patches = {p["file_path"]: p["patched_source"] for p in args["patches"]}
    return FixResult(patches=patches, explanation=args["explanation"])


_KTLINT_ERROR_RE = re.compile(r"/workspace/repo/(.+?):(\d+):\d+: (.+)")

def _extract_error_context(lint_errors: str, patches: dict[str, str]) -> str:
    """
    Parse ktlint error lines and extract the surrounding source lines from the
    patched file, marking the exact error line with '>>>'.
    """
    sections: list[str] = []
    for match in _KTLINT_ERROR_RE.finditer(lint_errors):
        file_path = match.group(1)
        line_num = int(match.group(2))
        message = match.group(3)
        if file_path not in patches:
            continue
        source_lines = patches[file_path].splitlines()
        start = max(0, line_num - 5)
        end = min(len(source_lines), line_num + 4)
        snippet_lines = []
        for i in range(start, end):
            prefix = ">>>" if i + 1 == line_num else "   "
            snippet_lines.append(f"{prefix} {i + 1}: {source_lines[i]}")
        sections.append(
            f"\n**`{file_path}` line {line_num}:** {message}\n```\n"
            + "\n".join(snippet_lines)
            + "\n```"
        )
    return "\n".join(sections)


def _build_context_message(
    classification: ClassificationResult,
    code_chunks: list[CodeChunk],
    repo_url: str,
) -> str:
    lines: list[str] = []

    lines.append("## User Review")
    lines.append(f"> {classification.review_text}")
    if classification.star_rating is not None:
        lines.append(f"\nStar rating: {classification.star_rating}/5")

    lines.append("\n## Classification")
    lines.append(f"- Intent: **{classification.intent}**")
    lines.append(f"- Is vague: {classification.is_vague}")
    lines.append(f"- Confidence: {classification.confidence:.0%}")
    lines.append(f"- Reasoning: {classification.reasoning}")

    if classification.follow_up_questions:
        lines.append("\n## Follow-up Questions (generated for context)")
        for i, q in enumerate(classification.follow_up_questions, 1):
            lines.append(f"{i}. {q}")

    lines.append(f"\n## Repository\n{repo_url}")

    lines.append("\n## Relevant Code (ranked by semantic similarity)")
    for chunk in code_chunks:
        lines.append(
            f"\n### `{chunk.file_path}` — `{chunk.function_name}()` (score: {chunk.score:.3f})"
        )
        ext = chunk.file_path.rsplit(".", 1)[-1] if "." in chunk.file_path else ""
        lines.append(f"```{ext}")
        lines.append(chunk.source_text)
        lines.append("```")

    lines.append(
        "\nPlease propose a targeted fix. Return complete file content for any modified files."
    )

    return "\n".join(lines)
