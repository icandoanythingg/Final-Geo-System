"""Normalized evidence row helpers for the global shelf."""

from __future__ import annotations

import hashlib
import json
from typing import Any


CLEAN_EVIDENCE_FIELDS = [
    "evidence_id",
    "source_type",
    "source_name",
    "text",
    "topic",
    "persona",
    "stage",
    "area",
    "project",
    "seed",
]

EVIDENCE_AUDIT_FIELDS = [
    "evidence_id",
    "source_type",
    "url",
    "date",
    "title",
    "question_text",
    "topic_evidence",
    "topic_reason",
    "persona_evidence",
    "persona_basis",
    "persona_reason",
    "stage_evidence",
    "stage_reason",
    "area_evidence",
    "area_reason",
    "promo",
    "promo_reason",
    "check",
    "meta",
]

QUESTION_SOURCES = frozenset(
    {"reddit", "portal_faq", "paa", "pasf", "autocomplete"}
)


def evidence_id(source_type: str, stable_key: str) -> str:
    return f"{source_type}:{stable_key}"


def stable_hash(*parts: str) -> str:
    blob = "|".join(p or "" for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def empty_norm(
    *,
    source_type: str,
    source_name: str,
    stable_key: str,
    text: str,
    title: str = "",
    url: str = "",
    seed: str = "",
    project: str = "",
    area_hint: str = "",
    date: str = "",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pre-tag normalized row (ingest output)."""
    return {
        "evidence_id": evidence_id(source_type, stable_key),
        "source_type": source_type,
        "source_name": source_name or source_type,
        "text": (text or "").strip(),
        "title": (title or text or "").strip()[:200],
        "url": url or "",
        "seed": seed or "",
        "project": project or "",
        "area_hint": area_hint or "",
        "date": date or "",
        "meta": meta or {},
        "thread_id": evidence_id(source_type, stable_key),  # cache key for tagger
    }


def meta_json(meta: Any) -> str:
    if not meta:
        return ""
    if isinstance(meta, str):
        return meta
    return json.dumps(meta, ensure_ascii=False, sort_keys=True)


def to_clean_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row.get(k, "") or "" for k in CLEAN_EVIDENCE_FIELDS}


def to_evidence_audit(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: row.get(k, "") or "" for k in EVIDENCE_AUDIT_FIELDS}
    out["meta"] = meta_json(row.get("meta"))
    return out


def as_thread(norm: dict[str, Any]) -> dict[str, Any]:
    """Shape a norm row like a Reddit thread for rules/LLM/merge."""
    text = norm.get("text") or ""
    title = norm.get("title") or text
    body = text if text != title else ""
    # Include area/project hints in body so rules can see them
    extras = []
    if norm.get("area_hint"):
        extras.append(str(norm["area_hint"]))
    if norm.get("project"):
        extras.append(str(norm["project"]))
    if extras:
        body = (body + "\n" + " ".join(extras)).strip()
    return {
        "thread_id": norm["thread_id"],
        "url": norm.get("url") or "",
        "title": title,
        "body": body,
        "created_utc": norm.get("date") or "",
        "comments": [],
        "subreddit": norm.get("source_name") or "",
    }
