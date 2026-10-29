"""Tag one normalized question-source item via rules + LLM + merge."""

from __future__ import annotations

from typing import Any, Callable

from lib.merge import merge_row
from lib.rules import build_hints
from lib.schema import as_thread
from lib.synth import synth_from_rules


def tag_question_item(
    norm: dict[str, Any],
    tags: dict[str, Any],
    prompt: str,
    cache: Any,
    *,
    skip_llm: bool = False,
    known_topics: set[str],
    known_personas: set[str],
    known_areas: set[str],
    tag_thread_fn: Callable[..., tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Returns (evidence_row, stats).
    evidence_row has clean_evidence + audit fields + new_* for report.
    """
    thread = norm.get("_thread") or as_thread(norm)
    # Ensure thread_id matches evidence cache key
    thread["thread_id"] = norm["thread_id"]

    hints = build_hints(thread, tags)

    def synth_fn(t: dict[str, Any], h: dict[str, Any]) -> dict[str, Any]:
        return synth_from_rules(t, h, tags)

    llm_v, st = tag_thread_fn(
        thread,
        hints,
        prompt,
        cache,
        skip_llm=skip_llm,
        known_topics=known_topics,
        known_personas=known_personas,
        known_areas=known_areas,
        synth_fn=synth_fn,
    )
    merged = merge_row(
        thread,
        hints,
        llm_v,
        known_personas=known_personas,
        known_topics=known_topics,
        is_sample=False,
    )
    for a in llm_v.get("new_areas") or []:
        if a and a not in known_areas and a not in (merged.get("new_areas") or []):
            merged.setdefault("new_areas", []).append(a)

    # Prefer portal area_hint when merge fell back to city-general and hint is specific
    area = merged.get("area") or ""
    hint = (norm.get("area_hint") or "").strip()
    if hint and area in ("", "Hyderabad (general)"):
        # map hint through known areas if possible
        area = _resolve_area_hint(hint, tags) or hint

    row = {
        "evidence_id": norm["evidence_id"],
        "source_type": norm["source_type"],
        "source_name": norm["source_name"],
        "text": (norm.get("title") or norm.get("text") or "")[:500],
        "topic": merged.get("topic") or "Other",
        "persona": merged.get("persona") or "",
        "stage": merged.get("stage") or "",
        "area": area,
        "project": norm.get("project") or "",
        "seed": norm.get("seed") or "",
        "url": merged.get("url") or norm.get("url") or "",
        "date": merged.get("date") or norm.get("date") or "",
        "title": merged.get("title") or norm.get("title") or "",
        "question_text": merged.get("question_text") or norm.get("text") or "",
        "topic_evidence": merged.get("topic_evidence") or "",
        "topic_reason": merged.get("topic_reason") or "",
        "persona_evidence": merged.get("persona_evidence") or "",
        "persona_basis": merged.get("persona_basis") or "",
        "persona_reason": merged.get("persona_reason") or "",
        "stage_evidence": merged.get("stage_evidence") or "",
        "stage_reason": merged.get("stage_reason") or "",
        "area_evidence": merged.get("area_evidence") or hint,
        "area_reason": merged.get("area_reason") or "",
        "promo": merged.get("promo") or "No",
        "promo_reason": merged.get("promo_reason") or "",
        "check": merged.get("check") or "",
        "meta": norm.get("meta") or {},
        "new_persona": merged.get("new_persona") or "",
        "new_topic": merged.get("new_topic") or "",
        "new_areas": merged.get("new_areas") or [],
        "thread_id": norm["evidence_id"],  # for new_* aggregation reuse
    }
    return row, st


def tag_trends_item(norm: dict[str, Any]) -> dict[str, Any]:
    """Signal-only row — no LLM."""
    return {
        "evidence_id": norm["evidence_id"],
        "source_type": "trends",
        "source_name": norm.get("source_name") or "google_trends",
        "text": norm.get("text") or norm.get("seed") or "",
        "topic": "",
        "persona": "",
        "stage": "",
        "area": norm.get("area") or "Hyderabad (general)",
        "project": "",
        "seed": norm.get("seed") or "",
        "url": "",
        "date": "",
        "title": norm.get("title") or norm.get("seed") or "",
        "question_text": "",
        "topic_evidence": "",
        "topic_reason": "",
        "persona_evidence": "",
        "persona_basis": "",
        "persona_reason": "",
        "stage_evidence": "",
        "stage_reason": "",
        "area_evidence": norm.get("seed") or "",
        "area_reason": "Trends seed mapped to locality alias.",
        "promo": "",
        "promo_reason": "",
        "check": "",
        "meta": norm.get("meta") or {},
        "new_persona": "",
        "new_topic": "",
        "new_areas": [],
        "thread_id": norm["evidence_id"],
    }


def reddit_merged_to_evidence(
    merged: dict[str, Any],
    thread: dict[str, Any],
) -> dict[str, Any]:
    """Convert a Reddit merge_row result into a clean_evidence row."""
    tid = merged["thread_id"]
    title = merged.get("title") or thread.get("title") or ""
    return {
        "evidence_id": f"reddit:{tid}",
        "source_type": "reddit",
        "source_name": thread.get("subreddit") or "reddit",
        "text": title,
        "topic": merged.get("topic") or "Other",
        "persona": merged.get("persona") or "",
        "stage": merged.get("stage") or "",
        "area": merged.get("area") or "Hyderabad (general)",
        "project": "",
        "seed": "",
        "url": merged.get("url") or "",
        "date": merged.get("date") or "",
        "title": title,
        "question_text": merged.get("question_text") or "",
        "topic_evidence": merged.get("topic_evidence") or "",
        "topic_reason": merged.get("topic_reason") or "",
        "persona_evidence": merged.get("persona_evidence") or "",
        "persona_basis": merged.get("persona_basis") or "",
        "persona_reason": merged.get("persona_reason") or "",
        "stage_evidence": merged.get("stage_evidence") or "",
        "stage_reason": merged.get("stage_reason") or "",
        "area_evidence": merged.get("area_evidence") or "",
        "area_reason": merged.get("area_reason") or "",
        "promo": merged.get("promo") or "No",
        "promo_reason": merged.get("promo_reason") or "",
        "check": merged.get("check") or "",
        "meta": {"thread_id": tid},
        "new_persona": merged.get("new_persona") or "",
        "new_topic": merged.get("new_topic") or "",
        "new_areas": merged.get("new_areas") or [],
        "thread_id": tid,
    }


def _resolve_area_hint(hint: str, tags: dict[str, Any]) -> str:
    h = hint.strip().lower()
    for name, aliases in (tags.get("areas") or {}).items():
        cands = [name.lower()] + [a.lower() for a in (aliases or [])]
        if h in cands or any(h == a or a in h for a in cands if a):
            if name == "Hyderabad":
                return "Hyderabad (general)"
            return name
    return hint.strip()
