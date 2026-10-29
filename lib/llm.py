"""Step 3: LLM pass with quote validation and disk cache."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

GEO_ROOT = Path("/Users/abhijeet/geo-prompt-system")
if str(GEO_ROOT) not in sys.path:
    sys.path.insert(0, str(GEO_ROOT))

from agents.ollama_client import MODEL, extract_json, generate  # noqa: E402

log = logging.getLogger("clean_threads.llm")

PROMPT_VERSION = "v4"  # add per-field reason; implied persona/stage fill

# Compact system prompt — full llm_prompt.txt is human-editable source of examples/rules.
COMPACT_SYSTEM = """Tag Hyderabad flat-buyer Reddit threads. JSON only. Hints may be wrong — ignore if unsupported.
Return:
{"area":[{"name","quote","reason"}],"topic":{"value","quote","reason","suggested_label"},"persona":{"value","quote","basis","reason"},"stage":{"value","quote","reason"},"promo":{"value","quote","reason"}}
Rules:
- area: Hyderabad places in title/body/comments (verbatim if new). Empty array if none.
- topic: one of Charges|Area measurement|Resale|Loan|Price|Possession delay|Builder trust|Legal and documents|Rental yield|Location and commute|Schools and hospitals|Waterlogging and traffic|Other.
- persona: from OP title+body only. Use stated if they say it, implied if body strongly suggests who they are. Fill when clear; else none. Never from location/budget alone.
- stage: OP buying stage from body when clear (incl. implied); else none.
- promo: Yes only if author pushes own tool/brand/site/data product; buyer Q&A polls are No.
- reason: one short sentence why you chose that value (required for every non-none field).
- Every non-none value needs a verbatim quote from the thread.
Examples:
1) 23% sft common spaces + "I'm a first time buyer" → topic Area measurement, persona First-time buyer stated, stage none, area [], promo No
2) buyer asks transfer charges; comment Kothaguda Kondapur → topic Charges, persona none, promo No, areas those
3) first home resale + offered 5L token → topic Loan, persona First-time buyer, stage Ready to book
4) cleaned listings + own tool link → topic Price, persona none, promo Yes
When a person corrects a row, append it to config/llm_prompt.txt as a new example."""


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def quote_in_text(quote: str | None, haystack: str) -> bool:
    if not quote:
        return False
    return norm_ws(quote) in norm_ws(haystack)


def thread_text_blob(thread: dict[str, Any]) -> str:
    parts = [thread.get("title") or "", thread.get("body") or ""]
    for c in thread.get("comments") or []:
        parts.append(c.get("text") or "")
    return "\n".join(parts)


def text_hash(thread: dict[str, Any]) -> str:
    return hashlib.sha256(thread_text_blob(thread).encode("utf-8")).hexdigest()[:16]


def cache_key(thread_id: str, thash: str, prompt_version: str = PROMPT_VERSION) -> str:
    return f"{thread_id}|{thash}|{prompt_version}"


class LlmCache:
    def __init__(self, path: Path):
        self.path = path
        self._mem: dict[str, dict[str, Any]] = {}
        import threading

        self._lock = threading.Lock()
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        self._mem[obj["key"]] = obj
                    except (json.JSONDecodeError, KeyError):
                        continue

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._mem.get(key)
        return row.get("response") if row else None

    def put(self, key: str, response: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
        obj = {"key": key, "response": response, **(meta or {})}
        with self._lock:
            self._mem[key] = obj
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_user_payload(
    thread: dict[str, Any],
    hints: dict[str, Any] | None,
    *,
    include_hints: bool = True,
) -> str:
    title = thread.get("title") or ""
    body = (thread.get("body") or "")[:1500]
    comments = thread.get("comments") or []
    top3 = []
    for c in comments[:3]:
        top3.append((c.get("text") or "")[:300])
    payload = {
        "thread_id": thread["thread_id"],
        "title": title,
        "body": body,
        "top_comments": top3,
    }
    if include_hints and hints is not None:
        payload["rule_hints"] = hints
        payload["hint_note"] = (
            "Hints may be wrong. Ignore any hint the text does not support."
        )
    else:
        payload["rule_hints"] = None
        payload["hint_note"] = "No hints provided. Rely only on the thread text."
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_llm(
    system_prompt: str,
    user_payload: str,
    *,
    retries: int = 2,
    temperature: float = 0.0,
) -> dict[str, Any] | None:
    delay = 1.5
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            raw = generate(
                user_payload,
                system=system_prompt,
                model=MODEL,
                timeout=180,
                temperature=temperature,
            )
            parsed = extract_json(raw)
            if parsed is not None:
                return parsed
            last_err = f"json_parse_fail attempt={attempt}"
            log.warning("%s raw[:200]=%r", last_err, raw[:200])
        except urllib.error.URLError as e:
            last_err = f"url_error: {e}"
            log.warning("%s attempt=%s", last_err, attempt)
        except TimeoutError as e:
            last_err = f"timeout: {e}"
            log.warning("%s attempt=%s", last_err, attempt)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            log.warning("%s attempt=%s", last_err, attempt)
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    log.error("LLM failed after %s retries: %s", retries, last_err)
    return None


def validate_llm_response(
    raw: dict[str, Any] | None, thread: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Return (validated_fields, check_reasons). Discard fields whose quotes fail."""
    checks: list[str] = []
    out: dict[str, Any] = {
        "area": [],
        "topic": None,
        "topic_suggested": None,
        "persona": None,
        "persona_basis": None,
        "persona_evidence": None,
        "stage": None,
        "stage_evidence": None,
        "topic_evidence": None,
        "area_evidence": [],
        "area_reason": None,
        "topic_reason": None,
        "persona_reason": None,
        "stage_reason": None,
        "promo_reason": None,
        "promo": None,
        "promo_evidence": None,
        "new_persona": None,
        "new_topic": None,
        "new_areas": [],
    }
    if not raw:
        return out, checks

    hay = thread_text_blob(thread)
    quote_fail = False
    area_reasons: list[str] = []

    # areas
    areas_in = raw.get("area") or []
    if isinstance(areas_in, dict):
        areas_in = [areas_in]
    if isinstance(areas_in, str):
        areas_in = [{"name": areas_in, "quote": areas_in}]
    for a in areas_in:
        if not isinstance(a, dict):
            continue
        name = (a.get("name") or "").strip()
        quote = a.get("quote")
        reason = (a.get("reason") or "").strip()
        if not name:
            continue
        if quote and not quote_in_text(str(quote), hay):
            quote_fail = True
            continue
        if quote and quote_in_text(str(quote), hay):
            out["area"].append(name)
            out["area_evidence"].append(str(quote))
            if reason:
                area_reasons.append(f"{name}: {reason}")
        elif not quote:
            # no quote — discard
            quote_fail = True
    if area_reasons:
        out["area_reason"] = " | ".join(area_reasons)

    # topic
    topic = raw.get("topic")
    if isinstance(topic, str):
        topic = {"value": topic, "quote": None}
    if isinstance(topic, dict):
        val = (topic.get("value") or "").strip()
        quote = topic.get("quote")
        sug = topic.get("suggested_label")
        reason = (topic.get("reason") or "").strip()
        if val:
            if quote and quote_in_text(str(quote), hay):
                out["topic"] = val
                out["topic_evidence"] = str(quote)
                out["topic_reason"] = reason or None
                if val == "Other" and sug:
                    out["topic_suggested"] = str(sug).strip()
                    out["new_topic"] = str(sug).strip()
            elif quote:
                quote_fail = True
            else:
                quote_fail = True

    # persona
    persona = raw.get("persona")
    if isinstance(persona, str):
        persona = {"value": persona, "quote": None, "basis": None}
    if isinstance(persona, dict):
        val = (persona.get("value") or "").strip()
        quote = persona.get("quote")
        basis = persona.get("basis")
        reason = (persona.get("reason") or "").strip()
        if val and val.lower() != "none":
            if quote and quote_in_text(str(quote), hay):
                out["persona"] = val
                out["persona_evidence"] = str(quote)
                if basis in ("stated", "implied"):
                    out["persona_basis"] = basis
                else:
                    out["persona_basis"] = "implied"  # filled from body context
                out["persona_reason"] = reason or None
            elif quote:
                quote_fail = True
            else:
                quote_fail = True

    # stage
    stage = raw.get("stage")
    if isinstance(stage, str):
        stage = {"value": stage, "quote": None}
    if isinstance(stage, dict):
        val = (stage.get("value") or "").strip()
        quote = stage.get("quote")
        reason = (stage.get("reason") or "").strip()
        if val and val.lower() != "none":
            if quote and quote_in_text(str(quote), hay):
                out["stage"] = val
                out["stage_evidence"] = str(quote)
                out["stage_reason"] = reason or None
            elif quote:
                quote_fail = True
            else:
                quote_fail = True

    # promo
    promo = raw.get("promo")
    if isinstance(promo, str):
        promo = {"value": promo, "quote": None}
    if isinstance(promo, dict):
        val = (promo.get("value") or "").strip()
        quote = promo.get("quote")
        reason = (promo.get("reason") or "").strip()
        if val in ("Yes", "No"):
            out["promo_reason"] = reason or None
            if quote:
                if quote_in_text(str(quote), hay):
                    out["promo"] = val
                    out["promo_evidence"] = str(quote)
                else:
                    quote_fail = True
                    out["promo"] = val  # promo Yes/No still usable without quote
            else:
                out["promo"] = val

    if quote_fail:
        checks.append("llm quote failed")
    return out, checks


def tag_thread(
    thread: dict[str, Any],
    hints: dict[str, Any],
    system_prompt: str,
    cache: LlmCache,
    *,
    skip_llm: bool = False,
    known_topics: set[str] | None = None,
    known_personas: set[str] | None = None,
    known_areas: set[str] | None = None,
    synth_fn: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (validated, stats) where stats has llm_calls, cache_hit, etc."""
    thash = text_hash(thread)
    key = cache_key(thread["thread_id"], thash)
    stats = {"llm_calls": 0, "cache_hit": False, "failed": False, "synth": False}

    raw = cache.get(key)
    if raw is not None:
        stats["cache_hit"] = True
    elif skip_llm and synth_fn is not None:
        raw = synth_fn(thread, hints)
        stats["synth"] = True
        cache.put(key, raw, {"thread_id": thread["thread_id"], "synth": True})
    elif skip_llm:
        raw = None
        stats["failed"] = True
    else:
        # Prefer compact system for speed; fall back to full file prompt if empty.
        sys_p = system_prompt.strip() or COMPACT_SYSTEM
        if len(sys_p) > 2500:
            sys_p = COMPACT_SYSTEM
        user = build_user_payload(thread, hints, include_hints=True)
        raw = call_llm(sys_p, user, retries=1, temperature=0.0)
        stats["llm_calls"] = 1
        if raw is None and synth_fn is not None:
            raw = synth_fn(thread, hints)
            stats["synth"] = True
            stats["failed"] = True
            cache.put(key, raw, {"thread_id": thread["thread_id"], "synth": True})
        elif raw is None:
            stats["failed"] = True
        else:
            cache.put(key, raw, {"thread_id": thread["thread_id"], "blind": False})

    validated, checks = validate_llm_response(raw, thread)

    # mark new labels vs known lists
    known_topics = known_topics or set()
    known_personas = known_personas or set()
    known_areas = known_areas or set()
    if validated.get("topic") and validated["topic"] not in known_topics and validated["topic"] != "Other":
        validated["new_topic"] = validated["topic"]
    if validated.get("persona") and validated["persona"] not in known_personas:
        validated["new_persona"] = validated["persona"]
    new_areas = []
    for a in validated.get("area") or []:
        if a not in known_areas:
            new_areas.append(a)
    validated["new_areas"] = new_areas
    validated["_checks"] = checks
    validated["_raw"] = raw
    return validated, stats


def blind_tag(
    thread: dict[str, Any],
    system_prompt: str,
    cache: LlmCache,
) -> dict[str, Any] | None:
    """Second LLM call without hints; cached under blind key."""
    thash = text_hash(thread)
    key = cache_key(thread["thread_id"], thash, prompt_version=PROMPT_VERSION + "|blind")
    raw = cache.get(key)
    if raw is None:
        user = build_user_payload(thread, None, include_hints=False)
        raw = call_llm(system_prompt, user, retries=3, temperature=0.0)
        if raw is not None:
            cache.put(key, raw, {"thread_id": thread["thread_id"], "blind": True})
    if raw is None:
        return None
    validated, _ = validate_llm_response(raw, thread)
    return validated
