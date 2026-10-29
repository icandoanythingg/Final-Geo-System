"""Step 2: rule hints for the LLM (not final answers)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _word_re(phrase: str) -> re.Pattern[str]:
    esc = re.escape(phrase.strip()).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{esc}\b", re.I)


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text)) if text else 0


def _question_text(title: str, body: str) -> str:
    # sentences ending in ? from the post body; else title
    parts = re.findall(r"[^.!?\n]*\?", body or "")
    qs = [p.strip() for p in parts if p.strip()]
    if qs:
        return " ".join(qs)
    return (title or "").strip()


def _promo(title: str, body: str) -> str:
    """Promo = author pushing their own tool/brand/listing/Flat/Project/Society — not a buyer Q&A poll."""
    blob = f"{title}\n{body}".lower()
    # Buyer polls asking peers are NOT promo (no bare \bpoll\b).
    if re.search(
        r"(https?://|www\.|"
        r"i built|my (tool|app|site|dashboard|product)|"
        r"check (out )?my |"
        r"we(?:'| a)re (a |an )?(brand|broker|agency)|"
        r"i cleaned|here'?s what moves|"
        r"app\.[a-z0-9.-]+\.[a-z]{2,}|"
        r"exclusive (deals|inventory)|our (tool|platform|product))",
        blob,
        re.I,
    ):
        return "Yes"
    # Soft brand CTA without a buyer question mark in title
    if re.search(r"\b(dm for price|book now|limited units|inventory available|book a visit|Urgent sale)\b", blob):
        return "Yes"
    return "No"


def build_hints(thread: dict[str, Any], tags: dict[str, Any]) -> dict[str, Any]:
    title = thread.get("title") or ""
    body = thread.get("body") or ""
    comments = "\n".join(c.get("text") or "" for c in thread.get("comments") or [])
    op = f"{title}\n{body}"
    everywhere = f"{op}\n{comments}"

    area_counts: Counter[str] = Counter()
    areas: dict[str, list[str]] = tags.get("areas") or {}
    for name, aliases in areas.items():
        n = 0
        for a in aliases:
            n += _count(_word_re(a), everywhere)
        if n:
            area_counts[name] = n
    area_hints = [
        {"name": n, "count": c}
        for n, c in sorted(area_counts.items(), key=lambda x: (-x[1], x[0]))
    ]

    topic_scores: dict[str, float] = {}
    for name, kws in (tags.get("topics") or {}).items():
        if name == "Other" or not kws:
            continue
        s = 0.0
        for kw in kws:
            pat = _word_re(kw)
            s += 3.0 * _count(pat, title)
            s += 1.0 * _count(pat, body)
            s += 0.5 * _count(pat, comments)
        if s:
            topic_scores[name] = s
    top2 = sorted(topic_scores.items(), key=lambda x: (-x[1], x[0]))[:2]
    topic_hint = [{"name": n, "score": s} for n, s in top2]

    persona_hint = None
    for label, phrases in (tags.get("personas") or {}).items():
        for ph in phrases:
            if _word_re(ph).search(op):
                persona_hint = {"label": label, "phrase": ph}
                break
        if persona_hint:
            break

    stage_hint = None
    for label, phrases in (tags.get("stages") or {}).items():
        for ph in phrases:
            if _word_re(ph).search(op):
                stage_hint = {"label": label, "phrase": ph}
                break
        if stage_hint:
            break

    return {
        "area_hints": area_hints,
        "topic_hint": topic_hint,
        "persona_hint": persona_hint,
        "stage_hint": stage_hint,
        "promo": _promo(title, body),
        "question_text": _question_text(title, body),
    }
