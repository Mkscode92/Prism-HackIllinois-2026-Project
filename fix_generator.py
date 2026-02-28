from __future__ import annotations
import logging
from dataclasses import dataclass, field
from google import genai
from google.genai import types
from classifier import ClassificationResult
from config import settings

logger = logging.getLogger("prism.fix_generator")

client = genai.Client(api_key=settings.gemini_api_key)

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    file_path: str        # Relative path inside the repo
    function_name: str
    source_text: str
    score: float          # Pinecone cosine similarity score


@dataclass
class FixResult:
    patches: dict[str, str]      # { relative_file_path: full_patched_source }
    explanation: str              # Summary of root cause and fix
    files_changed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.files_changed:
            self.files_changed = list(self.patches.keys())


# ---------------------------------------------------------------------------
# Tool definition (Using modern types.FunctionDeclaration)
# ---------------------------------------------------------------------------

PROPOSE_FIX_FUNC = types.FunctionDeclaration(
    name="propose_fix",
    description=(
        "Propose targeted code patches to address a bug or UX issue reported by a user. "
        "Return complete file content (not diffs) for each modified file."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "explanation": {
                "type": "STRING",
                "description": (
                    "Clear explanation of the root cause and exactly what the fix does. "
                    "This will appear verbatim in the GitHub PR description."
                ),
            },
            "patches": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "file_path": {
                            "type": "STRING",
                            "description": "Relative file path from the repo root.",
                        },
                        "patched_source": {
                            "type": "STRING",
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
)

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

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_fix(
    classification: ClassificationResult,
    code_chunks: list[CodeChunk],
    repo_url: str,
) -> FixResult | None:
    """
    Ask Gemini to propose a code fix based on the user review and relevant code context.
    Uses Thinking Mode (if Pro) to ensure high-quality architectural reasoning.
    """
    user_message = _build_context_message(classification, code_chunks, repo_url)

    # 2. Modern configuration pattern
    # We use ThinkingConfig to leverage Gemini 3.1's advanced reasoning
    config = types.GenerateContentConfig(
        system_instruction=FIX_SYSTEM,
        tools=[types.Tool(function_declarations=[PROPOSE_FIX_FUNC])],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=["propose_fix"]
            )
        ),
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user_message,
        config=config
    )

    # 3. Handle the structured output from the Tool call
    try:
        fc = response.candidates[0].content.parts[0].function_call
        args = fc.args
    except (AttributeError, IndexError):
        logger.error("Gemini failed to return a valid function call for propose_fix")
        return None

    if not args.get("patches"):
        logger.warning("Gemini declined to generate patches: %s", args.get("explanation", ""))
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
    """
    Ask Gemini to correct a patch that failed lint validation.
    Sends the original patches + lint error output so Gemini can self-correct.
    """
    lines: list[str] = []

    lines.append("## Original Review")
    lines.append(f"> {classification.review_text}")

    lines.append("\n## Lint Errors in Your Previous Patch")
    lines.append("Your previous patch failed lint validation with these errors. Fix them:")
    lines.append("```")
    lines.append(lint_errors.strip())
    lines.append("```")

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

    config = types.GenerateContentConfig(
        system_instruction=FIX_SYSTEM,
        tools=[types.Tool(function_declarations=[PROPOSE_FIX_FUNC])],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=["propose_fix"]
            )
        ),
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents="\n".join(lines),
        config=config,
    )

    try:
        fc = response.candidates[0].content.parts[0].function_call
        args = fc.args
    except (AttributeError, IndexError):
        logger.error("Gemini failed to return a valid function call for refine_fix")
        return None

    if not args.get("patches"):
        logger.warning("Gemini declined to generate refined patches")
        return None

    patches = {p["file_path"]: p["patched_source"] for p in args["patches"]}
    return FixResult(patches=patches, explanation=args["explanation"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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