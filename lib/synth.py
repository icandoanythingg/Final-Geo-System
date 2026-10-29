"""Build LLM-shaped JSON from rules with verbatim quotes (cache fill / offline)."""

from __future__ import annotations

import re
from typing import Any


def _find_quote(phrase: str, text: str) -> str | None:
    if not phrase or not text:
        return None
    m = re.search(re.escape(phrase), text, re.I)
    if not m:
        return None
    return text[m.start() : m.end()]


def synth_from_rules(
    thread: dict[str, Any], hints: dict[str, Any], tags: dict[str, Any]
) -> dict[str, Any]:
    title = thread.get("title") or ""
    body = thread.get("body") or ""
    comments = "\n".join(c.get("text") or "" for c in thread.get("comments") or [])
    everywhere = f"{title}\n{body}\n{comments}"
    op = f"{title}\n{body}"

    areas = []
    for a in hints.get("area_hints") or []:
        name = a["name"]
        aliases = (tags.get("areas") or {}).get(name) or [name]
        quote = None
        for al in aliases:
            quote = _find_quote(al, everywhere)
            if quote:
                break
        if quote:
            areas.append(
                {
                    "name": name,
                    "quote": quote,
                    "reason": f"Locality '{name}' appears in the thread text.",
                }
            )

    topic = {
        "value": "Other",
        "quote": title[:80] or "Other",
        "suggested_label": None,
        "reason": "No strong topic keyword match; defaulted to Other.",
    }
    th = hints.get("topic_hint") or []
    if th:
        name = th[0]["name"]
        kws = (tags.get("topics") or {}).get(name) or []
        quote = None
        for kw in kws:
            quote = _find_quote(kw, everywhere)
            if quote:
                break
        topic = {
            "value": name,
            "quote": quote or title[:80],
            "suggested_label": None,
            "reason": f"Keyword match scored highest for topic '{name}'.",
        }

    persona = {"value": "none", "quote": None, "basis": None, "reason": None}
    ph = hints.get("persona_hint")
    if ph:
        q = _find_quote(ph.get("phrase") or "", op)
        if q:
            persona = {
                "value": ph["label"],
                "quote": q,
                "basis": "stated",
                "reason": f"OP uses the phrase that maps to '{ph['label']}'.",
            }

    stage = {"value": "none", "quote": None, "reason": None}
    sh = hints.get("stage_hint")
    if sh:
        q = _find_quote(sh.get("phrase") or "", op)
        if q:
            stage = {
                "value": sh["label"],
                "quote": q,
                "reason": f"OP wording matches stage '{sh['label']}'.",
            }

    promo_v = hints.get("promo") or "No"
    promo_q = None
    promo_reason = "No author tool/brand CTA found."
    if promo_v == "Yes":
        promo_reason = "Author appears to promote their own tool/brand/site."
        for pat in (
            r"https?://\S+",
            r"i cleaned",
            r"my (tool|app|site|dashboard)",
            r"exclusive (deals|inventory)",
        ):
            m = re.search(pat, f"{title}\n{body}", re.I)
            if m:
                promo_q = m.group(0)[:120]
                break

    return {
        "area": areas,
        "topic": topic,
        "persona": persona,
        "stage": stage,
        "promo": {"value": promo_v, "quote": promo_q, "reason": promo_reason},
    }
