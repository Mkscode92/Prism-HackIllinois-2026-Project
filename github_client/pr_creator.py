from __future__ import annotations

import base64
import logging

import httpx

from classifier import ClassificationResult
from config import settings
from fix_generator import FixResult
from sandbox.runner import SandboxResult

logger = logging.getLogger("prism.github")

GITHUB_API = "https://api.github.com"

def create_pr(
    repo_url: str,
    review_id: str,
    classification: ClassificationResult,
    fix: FixResult,
    sandbox_result: SandboxResult,
) -> str:
    """
    new branch -> commit changes -> open pull request -> return url of the pr 
    """
    owner, repo = _parse_repo(repo_url)
    branch_name = f"prism/fix-{review_id}"

    with httpx.Client(
        base_url=GITHUB_API,
        headers=_auth_headers(),
        timeout=30,
    ) as client:
        default_branch, head_sha = _get_default_branch_sha(client, owner, repo)
        logger.debug("Default branch: %s @ %s", default_branch, head_sha)

        _create_branch(client, owner, repo, branch_name, head_sha)
        logger.info("Created branch %s", branch_name)

        for file_path, new_content in fix.patches.items():
            _put_file(client, owner, repo, file_path, new_content, branch_name, review_id)
            logger.debug("Committed %s", file_path)

        pr_url = _open_pull_request(
            client,
            owner=owner,
            repo=repo,
            branch_name=branch_name,
            default_branch=default_branch,
            review_id=review_id,
            classification=classification,
            fix=fix,
            sandbox_result=sandbox_result,
        )

    return pr_url


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def _parse_repo(repo_url: str) -> tuple[str, str]:
    """'https://github.com/owner/repo[.git]' → ('owner', 'repo')"""
    clean = repo_url.rstrip("/").removesuffix(".git")
    parts = clean.split("/")
    return parts[-2], parts[-1]


def _get_default_branch_sha(
    client: httpx.Client, owner: str, repo: str
) -> tuple[str, str]:
    resp = client.get(f"/repos/{owner}/{repo}")
    resp.raise_for_status()
    default_branch: str = resp.json()["default_branch"]

    ref_resp = client.get(f"/repos/{owner}/{repo}/git/ref/heads/{default_branch}")
    ref_resp.raise_for_status()
    sha: str = ref_resp.json()["object"]["sha"]
    return default_branch, sha


def _create_branch(
    client: httpx.Client, owner: str, repo: str, branch_name: str, sha: str
) -> None:
    resp = client.post(
        f"/repos/{owner}/{repo}/git/refs",
        json={"ref": f"refs/heads/{branch_name}", "sha": sha},
    )
    resp.raise_for_status()


def _put_file(
    client: httpx.Client,
    owner: str,
    repo: str,
    file_path: str,
    new_content: str,
    branch_name: str,
    review_id: str,
) -> None:
    # GET the existing file to extract its blob SHA 
    existing_resp = client.get(
        f"/repos/{owner}/{repo}/contents/{file_path}",
        params={"ref": branch_name},
    )
    existing_sha: str | None = None
    if existing_resp.status_code == 200:
        existing_sha = existing_resp.json().get("sha")

    body: dict = {
        "message": f"fix: patch {file_path} for review {review_id} [Prism]",
        "content": base64.b64encode(new_content.encode("utf-8")).decode("ascii"),
        "branch": branch_name,
    }
    if existing_sha:
        body["sha"] = existing_sha

    resp = client.put(f"/repos/{owner}/{repo}/contents/{file_path}", json=body)
    resp.raise_for_status()


def _open_pull_request(
    client: httpx.Client,
    owner: str,
    repo: str,
    branch_name: str,
    default_branch: str,
    review_id: str,
    classification: ClassificationResult,
    fix: FixResult,
    sandbox_result: SandboxResult,
) -> str:
    title = (
        f"[Prism] {classification.intent.upper()}: "
        f"{fix.explanation[:60]}{'...' if len(fix.explanation) > 60 else ''}"
    )
    body = _build_pr_body(review_id, classification, fix, sandbox_result)

    resp = client.post(
        f"/repos/{owner}/{repo}/pulls",
        json={
            "title": title,
            "body": body,
            "head": branch_name,
            "base": default_branch,
            "draft": False,
        },
    )
    resp.raise_for_status()
    return resp.json()["html_url"]


def _build_pr_body(
    review_id: str,
    classification: ClassificationResult,
    fix: FixResult,
    sandbox_result: SandboxResult,
) -> str:
    follow_up_section = ""
    if classification.follow_up_questions:
        qs = "\n".join(f"{i}. {q}" for i, q in enumerate(classification.follow_up_questions, 1))
        follow_up_section = f"\n### Follow-up Questions Generated\n{qs}\n"

    files_list = "\n".join(f"- `{f}`" for f in fix.files_changed)
    star = f"{classification.star_rating}/5" if classification.star_rating else "N/A"

    return f"""## Prism Auto-Fix

**Review ID**: `{review_id}`
**Original Review** (⭐ {star}):
> {classification.review_text}

---

## Classification

| Field | Value |
|---|---|
| Intent | `{classification.intent}` |
| Is Vague | `{classification.is_vague}` |
| Confidence | `{classification.confidence:.0%}` |

**Reasoning**: {classification.reasoning}
{follow_up_section}
---

## Fix Summary

{fix.explanation}

**Files Changed**:
{files_list}

---
*Generated by **Prism** using `{settings.claude_model}`*
"""
