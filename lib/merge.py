"""Step 4: merge rules + validated LLM into final row."""

from __future__ import annotations

from typing import Any


def _rule_topic(hints: dict[str, Any]) -> str | None:
    th = hints.get("topic_hint") or []
    if th:
        return th[0].get("name")
    return None


def _rule_persona(hints: dict[str, Any]) -> str | None:
    ph = hints.get("persona_hint")
    return ph.get("label") if ph else None


def _rule_stage(hints: dict[str, Any]) -> str | None:
    sh = hints.get("stage_hint")
    return sh.get("label") if sh else None


def merge_row(
    thread: dict[str, Any],
    hints: dict[str, Any],
    llm: dict[str, Any],
    *,
    known_personas: set[str],
    known_topics: set[str],
    is_sample: bool = False,
) -> dict[str, Any]:
    checks: list[str] = list(llm.get("_checks") or [])

    # --- area: union rules+LLM, ordered by first mention in thread text ---
    candidates: list[str] = []
    for a in hints.get("area_hints") or []:
        n = a.get("name")
        if n:
            candidates.append(n)
    for n in llm.get("area") or []:
        if n:
            candidates.append(n)
    # de-dupe preserving discovery order, then sort by first mention index
    blob = f"{thread.get('title') or ''}\n{thread.get('body') or ''}\n" + "\n".join(
        c.get("text") or "" for c in thread.get("comments") or []
    )
    blob_l = blob.lower()

    def first_idx(name: str) -> int:
        i = blob_l.find(name.lower())
        return i if i >= 0 else 10**9

    seen: set[str] = set()
    ordered: list[str] = []
    for n in sorted(dict.fromkeys(candidates), key=lambda x: (first_idx(x), x)):
        if n not in seen:
            ordered.append(n)
            seen.add(n)

    # city-only → Hyderabad (general)
    if not ordered or ordered == ["Hyderabad"]:
        area = "Hyderabad (general)"
    else:
        locs = [x for x in ordered if x != "Hyderabad"]
        area = " | ".join(locs if locs else ordered)

    area_evidence = " | ".join(llm.get("area_evidence") or [])
    if not area_evidence and hints.get("area_hints"):
        area_evidence = ", ".join(
            f"{a['name']}×{a['count']}" for a in hints["area_hints"]
        )

    # --- topic ---
    topic = llm.get("topic")
    topic_evidence = llm.get("topic_evidence") or ""
    new_topic = llm.get("new_topic") or llm.get("topic_suggested")
    if topic and topic not in known_topics and topic != "Other":
        new_topic = topic
        topic = "Other"
    if not topic:
        topic = _rule_topic(hints) or "Other"
        if not topic_evidence and hints.get("topic_hint"):
            topic_evidence = hints["topic_hint"][0].get("name") or ""
    if topic in (None, "", "Untagged"):
        topic = "Other"
        checks.append("no topic")

    # --- persona ---
    persona = llm.get("persona")
    persona_basis = llm.get("persona_basis") or ""
    persona_evidence = llm.get("persona_evidence") or ""
    new_persona = None
    if persona and persona not in known_personas:
        new_persona = persona
        persona = ""
        persona_basis = ""
        checks.append("new persona")
    elif not persona:
        # do not fill from rules into clean persona when LLM said none;
        # rules only if LLM invalid (no usable answer) — still leave empty per spec
        # "persona is empty when the LLM returns none or a new label"
        # rules fallback if LLM answer is invalid:
        if llm.get("_raw") is None or "llm quote failed" in (llm.get("_checks") or []):
            # invalid LLM — rules fallback for audit evidence only; clean stays empty
            # Spec: "topic, persona, stage: the validated LLM answer wins. The rules are the fallback if the LLM answer is invalid."
            rp = _rule_persona(hints)
            if rp and rp in known_personas:
                persona = rp
                persona_basis = "stated"
                ph = hints.get("persona_hint") or {}
                persona_evidence = ph.get("phrase") or ""
        else:
            persona = ""
            persona_basis = ""
    # never default General buyer

    # --- stage ---
    stage = llm.get("stage") or ""
    stage_evidence = llm.get("stage_evidence") or ""
    if not stage:
        if llm.get("_raw") is None or "llm quote failed" in (llm.get("_checks") or []):
            stage = _rule_stage(hints) or ""
            sh = hints.get("stage_hint") or {}
            if stage and not stage_evidence:
                stage_evidence = sh.get("phrase") or ""
        else:
            stage = ""

    # --- promo ---
    promo = llm.get("promo") or hints.get("promo") or "No"
    if promo == "Yes":
        checks.append("promo")

    # --- rules vs llm ---
    rt, rp, rs = _rule_topic(hints), _rule_persona(hints), _rule_stage(hints)
    final_topic = topic
    final_persona = persona
    final_stage = stage
    if rt and final_topic and rt != final_topic:
        checks.append("rules vs llm")
    elif rp and final_persona and rp != final_persona:
        checks.append("rules vs llm")
    elif rs and final_stage and rs != final_stage:
        checks.append("rules vs llm")

    if is_sample:
        checks.append("sample")

    # dedupe check reasons, preserve order
    seen_c: set[str] = set()
    check_list = []
    for c in checks:
        if c and c not in seen_c:
            check_list.append(c)
            seen_c.add(c)
    check = "; ".join(check_list)

    clean_persona = persona or ""
    if clean_persona and persona_basis:
        # keep label only in clean_threads (basis in audit)
        pass

    return {
        "thread_id": thread["thread_id"],
        "url": thread.get("url") or "",
        "date": (thread.get("created_utc") or "")[:10],
        "title": thread.get("title") or "",
        "question_text": hints.get("question_text") or "",
        "topic": topic,
        "persona": clean_persona,
        "persona_basis": persona_basis or "",
        "stage": stage or "",
        "area": area,
        "topic_evidence": topic_evidence,
        "persona_evidence": persona_evidence,
        "stage_evidence": stage_evidence,
        "area_evidence": area_evidence,
        "topic_reason": llm.get("topic_reason") or "",
        "persona_reason": llm.get("persona_reason") or "",
        "stage_reason": llm.get("stage_reason") or "",
        "area_reason": llm.get("area_reason") or "",
        "promo_reason": llm.get("promo_reason") or "",
        "promo": promo,
        "check": check,
        "new_persona": new_persona or "",
        "new_topic": new_topic or "",
        "new_areas": llm.get("new_areas") or [],
        "_rule_topic": rt or "",
        "_rule_persona": rp or "",
        "_rule_stage": rs or "",
    }
