from __future__ import annotations

import asyncio
import base64
import functools
import json
import logging
from datetime import datetime, timezone

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings

from classifier import classify_review
from fix_generator import generate_fix, refine_fix
from github_client.pr_creator import create_pr
from poller import fetch_reviews_debug, poll_app_once, poll_loop
from rag.indexer import ensure_repo_indexed
from rag.searcher import search_code
from sandbox.runner import run_in_sandbox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("prism.main")

@asynccontextmanager
async def lifespan(_: FastAPI):
    asyncio.create_task(
        poll_loop(_app_registry, _handle_scraped_review, interval=settings.poll_interval_seconds)
    )
    yield


app = FastAPI(title="PRism", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_review_history: list[dict] = []
_pipeline_queues: dict[str, asyncio.Queue] = {}

# Maps Play Store package name → GitHub repo URL, registered via the UI
_app_registry: dict[str, str] = {}

# Low-rated reviews scraped by the poller, waiting for the user to trigger the pipeline
_pending_reviews: list[dict] = []


# ---------------------------------------------------------------------------
# Scraped review handler (called by poller)
# ---------------------------------------------------------------------------

async def _handle_scraped_review(package_name: str, repo_url: str, r: dict) -> None:
    """Add a scraped low-rated review to the pending queue for the user to action."""
    if len(_pending_reviews) >= settings.max_pending_reviews:
        logger.debug("Pending queue full (%d) — skipping %s", settings.max_pending_reviews, r.get("reviewId"))
        return

    text = (r.get("content") or "").strip()
    if not text:
        return

    _pending_reviews.append({
        "review_id": r["reviewId"],
        "author": r.get("userName"),
        "text": text,
        "score": r.get("score", 0),
        "package_name": package_name,
        "repo_url": repo_url,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    })
    logger.info(
        "Pending: added review %s for %s (score %d)",
        r["reviewId"], package_name, r.get("score", 0),
    )


# ---------------------------------------------------------------------------
# Pub/Sub payload models
# ---------------------------------------------------------------------------

class PubSubMessage(BaseModel):
    data: str
    messageId: str
    publishTime: str


class PubSubPayload(BaseModel):
    message: PubSubMessage
    subscription: str


# ---------------------------------------------------------------------------
# Internal review model
# ---------------------------------------------------------------------------

class PlayStoreReview(BaseModel):
    reviewId: str
    authorName: str | None = None
    repo_url: str
    comments: list[dict] = []
    starRating: int | None = None

    @property
    def review_text(self) -> str:
        try:
            return self.comments[0]["userComment"]["text"]
        except (IndexError, KeyError):
            return ""

    @property
    def star_rating(self) -> int | None:
        try:
            return self.comments[0]["userComment"]["starRating"]
        except (IndexError, KeyError):
            return self.starRating


# ---------------------------------------------------------------------------
# UI endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def serve_landing():
    return FileResponse("static/landing.html")


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse("static/index.html")


@app.get("/api/reviews")
def get_reviews():
    return _review_history


# ---------------------------------------------------------------------------
# App registry endpoints
# ---------------------------------------------------------------------------

class AppRegistration(BaseModel):
    package_name: str  
    repo_url: str       


@app.post("/api/apps")
async def register_app(body: AppRegistration):
    _app_registry[body.package_name] = body.repo_url
    logger.info("Registered app: %s → %s", body.package_name, body.repo_url)
    # Poll immediately so the user doesn't wait up to 10 minutes for the first results
    asyncio.create_task(poll_app_once(body.package_name, body.repo_url, _handle_scraped_review))
    return {"status": "ok", "registered": dict(_app_registry)}


@app.delete("/api/apps/{package_name}")
def remove_app(package_name: str):
    _app_registry.pop(package_name, None)
    return {"status": "ok", "registered": dict(_app_registry)}


@app.get("/api/apps")
def list_apps():
    return _app_registry


@app.post("/api/poll-now")
async def poll_now():
    """Manually trigger an immediate poll for all registered apps."""
    if not _app_registry:
        return {"status": "no_apps"}
    for pkg, repo in list(_app_registry.items()):
        asyncio.create_task(poll_app_once(pkg, repo, _handle_scraped_review))
    return {"status": "polling", "apps": list(_app_registry.keys())}


@app.get("/api/poll-debug")
async def poll_debug():
    """
    Fetch the 30 newest reviews for all registered apps WITHOUT any filtering,
    and annotate each with why it was or wasn't selected. Use this to diagnose
    why reviews aren't appearing in the pending queue.
    """
    if not _app_registry:
        return {"status": "no_apps", "results": {}}
    results = {}
    for pkg in list(_app_registry.keys()):
        results[pkg] = await fetch_reviews_debug(pkg)
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# Pending reviews endpoints
# ---------------------------------------------------------------------------

@app.get("/api/pending-reviews")
def get_pending_reviews():
    return _pending_reviews


@app.post("/api/pending-reviews/{review_id}/run")
async def run_pending_review(review_id: str):
    entry = next((r for r in _pending_reviews if r["review_id"] == review_id), None)
    if not entry:
        return {"status": "not_found"}

    _pending_reviews[:] = [r for r in _pending_reviews if r["review_id"] != review_id]

    review = PlayStoreReview(
        reviewId=entry["review_id"],
        authorName=entry["author"],
        repo_url=entry["repo_url"],
        comments=[{"userComment": {"text": entry["text"], "starRating": entry["score"]}}],
    )
    _pipeline_queues[review.reviewId] = asyncio.Queue()
    _review_history.insert(0, {
        "review_id": review.reviewId,
        "text": review.review_text,
        "star_rating": review.star_rating,
        "repo_url": review.repo_url,
        "stages": {},
        "pr_url": None,
        "intent": None,
        "done": False,
    })
    asyncio.create_task(run_pipeline(review))
    return {"status": "accepted", "review_id": review.reviewId}


@app.delete("/api/pending-reviews/{review_id}")
def dismiss_pending_review(review_id: str):
    _pending_reviews[:] = [r for r in _pending_reviews if r["review_id"] != review_id]
    return {"status": "ok"}


@app.get("/api/pipeline/{review_id}/stream")
async def stream_pipeline(review_id: str):
    async def generate():
        # Already done — send snapshot immediately and close
        entry = next((r for r in _review_history if r["review_id"] == review_id), None)
        if entry and entry.get("done"):
            yield f"data: {json.dumps({'type': 'snapshot', 'data': entry})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        queue = _pipeline_queues.get(review_id)
        if not queue:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Review not found'})}\n\n"
            return

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=60)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") == "done":
                        break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Payload parser — handles real Play Store and manual test formats
# ---------------------------------------------------------------------------

def _parse_review(data: dict, message_id: str) -> PlayStoreReview | None:
    # Real Play Store notification
    if "reviewNotification" in data:
        package_name = data.get("packageName", "")
        repo_url = _app_registry.get(package_name)
        if not repo_url:
            logger.warning("No repo registered for package '%s' — skipping", package_name)
            return None
        review_data = data["reviewNotification"].get("review", {})
        return PlayStoreReview(
            reviewId=review_data.get("reviewId", message_id),
            authorName=review_data.get("authorName"),
            repo_url=repo_url,
            comments=review_data.get("comments", []),
        )

    # Manual test submission from the UI
    if "repo_url" in data:
        return PlayStoreReview(
            reviewId=data.get("reviewId", message_id),
            authorName=data.get("authorName"),
            repo_url=data["repo_url"],
            comments=data.get("comments", []),
            starRating=data.get("starRating"),
        )

    return None


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@app.post("/webhook")
async def webhook(payload: PubSubPayload, background_tasks: BackgroundTasks):
    """
    Receive a Pub/Sub push message.

    Handles two formats:
    - Real Play Store notification: contains 'reviewNotification' and 'packageName'.
      repo_url is resolved from the registered app registry.
    - Manual test submission from the UI: contains 'repo_url' and 'comments' directly.
    """
    try:
        raw_json = base64.b64decode(payload.message.data).decode("utf-8")
        data = json.loads(raw_json)
    except Exception as exc:
        logger.error("Failed to decode Pub/Sub message: %s", exc)
        return {"status": "decode_error", "acknowledged": True}

    review = _parse_review(data, payload.message.messageId)
    if review is None:
        logger.error("Unrecognised payload format — missing 'reviewNotification' or 'repo_url'")
        return {"status": "unrecognised_format", "acknowledged": True}

    if not review.review_text:
        logger.warning("Review %s has no text — skipping", review.reviewId)
        return {"status": "skipped", "reason": "no_review_text"}

    # Create queue and history entry BEFORE starting the background task
    _pipeline_queues[review.reviewId] = asyncio.Queue()
    _review_history.insert(0, {
        "review_id": review.reviewId,
        "text": review.review_text,
        "star_rating": review.star_rating,
        "repo_url": review.repo_url,
        "stages": {},
        "pr_url": None,
        "intent": None,
        "done": False,
    })

    logger.info("Accepted review %s for repo %s", review.reviewId, review.repo_url)
    background_tasks.add_task(run_pipeline, review)
    return {"status": "accepted", "review_id": review.reviewId}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Event emitter
# ---------------------------------------------------------------------------

async def _emit(review_id: str, event: dict) -> None:
    """Push an SSE event to the client and update the in-memory history entry."""
    queue = _pipeline_queues.get(review_id)
    if queue:
        await queue.put(event)

    entry = next((r for r in _review_history if r["review_id"] == review_id), None)
    if not entry:
        return

    if event.get("type") == "stage":
        entry["stages"][event["stage"]] = {
            "status": event["status"],
            "message": event.get("message", ""),
        }
        if event.get("intent"):
            entry["intent"] = event["intent"]

    if event.get("type") == "pr_created":
        entry["pr_url"] = event["pr_url"]

    if event.get("type") == "done":
        entry["done"] = True
        _pipeline_queues.pop(review_id, None)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

async def run_pipeline(review: PlayStoreReview) -> None:
    rid = review.reviewId
    repo = review.repo_url
    logger.info("[%s] === Pipeline start | repo=%s ===", rid, repo)

    # ------------------------------------------------------------------
    # Stage 1: Classify
    # ------------------------------------------------------------------
    await _emit(rid, {"type": "stage", "stage": "classify", "status": "running", "message": "Sending review to Claude..."})
    loop = asyncio.get_running_loop()
    try:
        classification = await loop.run_in_executor(
            None, functools.partial(classify_review, rid, review.review_text, review.star_rating)
        )
        msg = f"Intent: {classification.intent} · Confidence: {classification.confidence:.0%}"
        await _emit(rid, {"type": "stage", "stage": "classify", "status": "success", "message": msg, "intent": classification.intent})
        if classification.follow_up_questions:
            logger.info("[%s] Follow-up questions: %s", rid, classification.follow_up_questions)
    except Exception as exc:
        logger.exception("[%s] Classification failed", rid)
        await _emit(rid, {"type": "stage", "stage": "classify", "status": "error", "message": str(exc)})
        for s in ["rag", "fix", "sandbox", "pr"]:
            await _emit(rid, {"type": "stage", "stage": s, "status": "skip", "message": "Skipped"})
        await _emit(rid, {"type": "done"})
        return


    # ------------------------------------------------------------------
    # Stage 2: RAG
    # ------------------------------------------------------------------
    await _emit(rid, {"type": "stage", "stage": "rag", "status": "running", "message": "Indexing repo and searching codebase..."})
    try:
        await loop.run_in_executor(None, functools.partial(ensure_repo_indexed, repo))
        code_chunks = await loop.run_in_executor(
            None, functools.partial(search_code, review.review_text, repo)
        )
        await _emit(rid, {"type": "stage", "stage": "rag", "status": "success", "message": f"Found {len(code_chunks)} relevant code chunk(s)"})
    except Exception as exc:
        logger.exception("[%s] RAG failed", rid)
        await _emit(rid, {"type": "stage", "stage": "rag", "status": "error", "message": str(exc)})
        for s in ["fix", "sandbox", "pr"]:
            await _emit(rid, {"type": "stage", "stage": s, "status": "skip", "message": "Skipped"})
        await _emit(rid, {"type": "done"})
        return

    if not code_chunks:
        await _emit(rid, {"type": "stage", "stage": "rag", "status": "error", "message": "No relevant code found in repo"})
        for s in ["fix", "sandbox", "pr"]:
            await _emit(rid, {"type": "stage", "stage": s, "status": "skip", "message": "Skipped"})
        await _emit(rid, {"type": "done"})
        return

    # ------------------------------------------------------------------
    # Stage 3: Generate fix
    # ------------------------------------------------------------------
    await _emit(rid, {"type": "stage", "stage": "fix", "status": "running", "message": "Generating code fix with Claude..."})
    try:
        fix = await loop.run_in_executor(
            None, functools.partial(generate_fix, classification, code_chunks, repo)
        )
    except Exception as exc:
        logger.exception("[%s] Fix generation failed", rid)
        await _emit(rid, {"type": "stage", "stage": "fix", "status": "error", "message": str(exc)})
        for s in ["sandbox", "pr"]:
            await _emit(rid, {"type": "stage", "stage": s, "status": "skip", "message": "Skipped"})
        await _emit(rid, {"type": "done"})
        return

    if fix is None:
        await _emit(rid, {"type": "stage", "stage": "fix", "status": "error", "message": "Claude declined to generate a fix (insufficient context)"})
        for s in ["sandbox", "pr"]:
            await _emit(rid, {"type": "stage", "stage": s, "status": "skip", "message": "Skipped"})
        await _emit(rid, {"type": "done"})
        return

    await _emit(rid, {"type": "stage", "stage": "fix", "status": "success", "message": f"Patched {len(fix.patches)} file(s)"})

    # ------------------------------------------------------------------
    # Stage 4: Sandbox validation (with one lint-error retry)
    # ------------------------------------------------------------------
    MAX_LINT_RETRIES = 1
    current_fix = fix
    sandbox_result = None

    for attempt in range(MAX_LINT_RETRIES + 1):
        attempt_label = f"attempt {attempt + 1}/{MAX_LINT_RETRIES + 1}"
        await _emit(rid, {
            "type": "stage", "stage": "sandbox", "status": "running",
            "message": f"Validating patch in Modal sandbox... ({attempt_label})"
        })

        try:
            sandbox_result = await loop.run_in_executor(
                None, functools.partial(run_in_sandbox, repo, current_fix)
            )
        except Exception as exc:
            logger.exception("[%s] Sandbox executor failed", rid)
            await _emit(rid, {"type": "stage", "stage": "sandbox", "status": "error", "message": str(exc)})
            await _emit(rid, {"type": "stage", "stage": "pr", "status": "skip", "message": "Skipped"})
            await _emit(rid, {"type": "done"})
            return

        if sandbox_result.success:
            break

        # Lint failed — attempt self-correction before giving up
        if not sandbox_result.lint_passed and attempt < MAX_LINT_RETRIES:
            logger.info("[%s] Lint failed on attempt %d — requesting self-correction", rid, attempt + 1)
            await _emit(rid, {
                "type": "stage", "stage": "sandbox", "status": "running",
                "message": f"Lint errors found — asking Claude to self-correct... (attempt {attempt + 2}/{MAX_LINT_RETRIES + 1})"
            })
            try:
                refined = await loop.run_in_executor(
                    None, functools.partial(
                        refine_fix, current_fix, sandbox_result.lint_output,
                        classification, code_chunks, repo
                    )
                )
                if refined:
                    current_fix = refined
                    continue
                else:
                    logger.warning("[%s] refine_fix returned empty on attempt %d — skipping retry", rid, attempt + 1)
            except Exception:
                logger.exception("[%s] refine_fix failed on attempt %d", rid, attempt + 1)

        break  # No more retries, lint passed, or refinement unavailable

    fix = current_fix

    if not sandbox_result.success:
        lint_icon = "✓" if sandbox_result.lint_passed else "✕"
        test_icon = "✓" if sandbox_result.test_passed else "✕"

        # Distinguish failure modes for better observability
        if not sandbox_result.lint_passed:
            failure_reason = f"Lint {lint_icon} — could not auto-correct after {MAX_LINT_RETRIES} retry attempt(s)"
        elif not sandbox_result.test_passed:
            failure_reason = f"Lint {lint_icon}  Tests {test_icon} — tests failing (may be pre-existing, check repo health)"
        else:
            failure_reason = f"Lint {lint_icon}  Tests {test_icon} — unknown validation failure"

        logger.warning(
            "[%s] Sandbox validation failed — lint_passed=%s test_passed=%s",
            rid, sandbox_result.lint_passed, sandbox_result.test_passed
        )
        await _emit(rid, {"type": "stage", "stage": "sandbox", "status": "error", "message": failure_reason})
        await _emit(rid, {"type": "stage", "stage": "pr", "status": "skip", "message": "Skipped (validation failed)"})
        await _emit(rid, {"type": "done"})
        return

    await _emit(rid, {"type": "stage", "stage": "sandbox", "status": "success", "message": "Lint ✓  Tests ✓"})

    # ------------------------------------------------------------------
    # Stage 5: Open PR
    # ------------------------------------------------------------------
    await _emit(rid, {"type": "stage", "stage": "pr", "status": "running", "message": "Creating GitHub branch and pull request..."})
    try:
        pr_url = await loop.run_in_executor(
            None, functools.partial(create_pr, repo, rid, classification, fix, sandbox_result)
        )
        await _emit(rid, {"type": "stage", "stage": "pr", "status": "success", "message": pr_url})
        await _emit(rid, {"type": "pr_created", "pr_url": pr_url})
    except Exception as exc:
        logger.exception("[%s] PR creation failed", rid)
        await _emit(rid, {"type": "stage", "stage": "pr", "status": "error", "message": str(exc)})

    await _emit(rid, {"type": "done"})
    logger.info("[%s] === Pipeline complete ===", rid)
