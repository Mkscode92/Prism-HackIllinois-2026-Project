from __future__ import annotations

import asyncio
import functools
import logging
from datetime import datetime, timezone

from google_play_scraper import Sort, reviews

logger = logging.getLogger("prism.poller")
_seen_ids: set[str] = set()
_MAX_CANDIDATES = 10


def _fetch_reviews(package_name: str) -> list[dict]:
    """Synchronous fetch — run in executor so it doesn't block the event loop."""
    result, _ = reviews(
        package_name,
        lang="en",
        country="us",
        sort=Sort.NEWEST,
        count=50,
    )
    return result


async def poll_app_once(
    package_name: str,
    repo_url: str,
    on_new_review,
) -> int:
    """
    Immediately poll a single app and queue any new low-rated reviews.
    Takes the 10 most recent score < 3 reviews with no time-window restriction.
    Returns the number of reviews added. Safe to call concurrently with poll_loop.
    """
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, functools.partial(_fetch_reviews, package_name)
        )
        candidates = [r for r in result if r.get("score", 5) < 3]
        candidates = candidates[:_MAX_CANDIDATES]

        added = 0
        for r in candidates:
            rid = r["reviewId"]
            if rid in _seen_ids:
                continue
            _seen_ids.add(rid)
            added += 1
            await on_new_review(package_name, repo_url, r)

        if added:
            logger.info("poll_app_once: %d new review(s) for %s", added, package_name)
        else:
            logger.info("poll_app_once: no new low-rated reviews for %s", package_name)
        return added
    except Exception:
        logger.exception("poll_app_once: failed for %s", package_name)
        return 0


async def fetch_reviews_debug(package_name: str) -> list[dict]:
    """
    Fetch the 50 newest reviews and annotate each with why it was or wasn't selected.
    Used by the /api/poll-debug endpoint to diagnose filtering.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, functools.partial(_fetch_reviews, package_name)
    )
    now = datetime.now(timezone.utc)
    out = []
    for r in result:
        score = r.get("score", 5)
        at = r.get("at")
        if at is not None and isinstance(at, datetime):
            if at.tzinfo is None:
                at = at.replace(tzinfo=timezone.utc)
            age_minutes = (now - at).total_seconds() / 60
            posted_at_str = at.isoformat()
        else:
            age_minutes = None
            posted_at_str = None

        excluded_reasons = []
        if score >= 3:
            excluded_reasons.append(f"score={score} (need <3)")
        if r.get("reviewId") in _seen_ids:
            excluded_reasons.append("already seen this session")

        out.append({
            "reviewId": r.get("reviewId"),
            "author": r.get("userName"),
            "score": score,
            "posted_at": posted_at_str,
            "age_minutes": round(age_minutes, 1) if age_minutes is not None else None,
            "text": (r.get("content") or "")[:200],
            "would_be_selected": len(excluded_reasons) == 0,
            "excluded_reasons": excluded_reasons,
        })
    return out


async def poll_loop(
    app_registry: dict[str, str],
    on_new_review,     
    interval: int = 300, # seconds between polls (default 10 min)
) -> None:
    """
    Background task: every `interval` seconds, fetch the 5 worst recent reviews
    (score < 3, posted within the last 10 minutes) for every registered app and
    pass each new one to on_new_review. Uses _seen_ids to avoid duplicates.
    """
    logger.info("Poller started — checking every %ds for score < 3 reviews", interval)
    # Short delay so the server is fully ready before the first poll
    await asyncio.sleep(10)

    while True:
        loop = asyncio.get_running_loop()
        for package_name, repo_url in list(app_registry.items()):
            try:
                result = await loop.run_in_executor(
                    None, functools.partial(_fetch_reviews, package_name)
                )
                candidates = [r for r in result if r.get("score", 5) < 3]
                candidates = candidates[:_MAX_CANDIDATES]

                added = 0
                for r in candidates:
                    rid = r["reviewId"]
                    if rid in _seen_ids:
                        continue
                    _seen_ids.add(rid)
                    added += 1
                    await on_new_review(package_name, repo_url, r)

                if added:
                    logger.info(
                        "Poller: %d new low-rated review(s) queued for %s",
                        added, package_name,
                    )
            except Exception:
                logger.exception("Poller: failed to fetch reviews for %s", package_name)

        await asyncio.sleep(interval)
