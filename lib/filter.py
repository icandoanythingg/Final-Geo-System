"""Step 1: rule-only filter. Log every drop."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

# Outside cities (whole word). Keep if Hyderabad is also mentioned (comparison).
_OUTSIDE = re.compile(
    r"\b(jaipur|bangalore|bengaluru|mumbai|pune|delhi|noida|gurgaon|gurugram|"
    r"chennai|kolkata|ahmedabad|chandigarh|kochi|coimbatore|indore|lucknow|"
    r"goa|mysore|mysuru)\b",
    re.I,
)
_HYD = re.compile(r"\b(hyderabad|hyd)\b", re.I)
_HOUSING_RE: re.Pattern[str] | None = None
_LOCALITY_RE: re.Pattern[str] | None = None
_PLOT_LAND_VILLA = re.compile(r"\b(plots?|lands?|villas?)\b", re.I)
# Keep if clearly about flats/projects — not pure plot/land sales
_PROJECTISH = re.compile(
    r"\b(flat|apartment|bhk|resale|gated community|society|carpet area|sft|"
    r"project|builder|tower|high-?rise|iconia|apartment)\b",
    re.I,
)
_BUY_INTENT = re.compile(
    r"\b(where to buy|trying to (buy|understand)|looking to buy|want to buy|"
    r"planning to buy|real estate|vs\b)",
    re.I,
)
_AD = re.compile(
    r"(\[for rent\]|\[for sale\]|\bfor rent\b|\bfor sale\b|"
    r"\bdm\b|\bdm me\b|\bwhatsapp\b|\bwa\.me\b|"
    r"\b\+?91[\s-]?\d{10}\b|\b\d{10}\b|"
    r"[\w.+-]+@[\w.-]+\.\w+)",
    re.I,
)
_OFFTOPIC = re.compile(
    r"\b(match tickets?|tickets? available|hiring|job opening|looking for a job|"
    r"internship|ipl tickets?)\b",
    re.I,
)


def _compile_housing(words: list[str]) -> re.Pattern[str]:
    parts = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        parts.append(re.escape(w).replace(r"\ ", r"\s+"))
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I)


def _compile_localities(area_aliases: list[str]) -> re.Pattern[str] | None:
    parts = []
    for a in area_aliases:
        a = (a or "").strip()
        if len(a) < 3:
            continue
        parts.append(re.escape(a).replace(r"\ ", r"\s+"))
    if not parts:
        return None
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I)


def drop_reason(
    thread: dict[str, Any],
    housing_words: list[str],
    *,
    area_aliases: list[str] | None = None,
) -> str | None:
    global _HOUSING_RE, _LOCALITY_RE
    if _HOUSING_RE is None:
        _HOUSING_RE = _compile_housing(housing_words)
    if area_aliases is not None:
        _LOCALITY_RE = _compile_localities(area_aliases)

    title = thread.get("title") or ""
    body = thread.get("body") or ""
    post = f"{title}\n{body}"

    if _AD.search(post):
        return "rental_or_sale_ad"
    if _OFFTOPIC.search(post):
        return "off_topic"
    # Pure plot/land/villa ads only — keep project / apartment enquiries
    if (
        _PLOT_LAND_VILLA.search(post)
        and not _PROJECTISH.search(post)
    ):
        return "plot_land_or_villa_only"
    if _OUTSIDE.search(post) and not _HYD.search(post):
        return "outside_hyderabad"
    has_housing = bool(_HOUSING_RE.search(post))
    has_locality = bool(_LOCALITY_RE.search(post)) if _LOCALITY_RE else False
    has_buy = bool(_BUY_INTENT.search(post))
    has_project = bool(_PROJECTISH.search(post))
    # Keep: housing keyword, OR known locality + (buy intent / project talk)
    if not (has_housing or (has_locality and (has_buy or has_project))):
        return "no_housing_word"
    return None


def filter_threads(
    threads: list[dict[str, Any]],
    housing_words: list[str],
    *,
    area_aliases: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return (kept, dropped_rows). Drops include dedupe."""
    dropped: list[dict[str, str]] = []
    kept: list[dict[str, Any]] = []

    for t in threads:
        reason = drop_reason(t, housing_words, area_aliases=area_aliases)
        if reason:
            dropped.append(
                {
                    "thread_id": t["thread_id"],
                    "url": t.get("url") or "",
                    "title": t.get("title") or "",
                    "drop_reason": reason,
                }
            )
        else:
            kept.append(t)

    # Dedupe by URL, then near-identical titles (keep earliest created_utc)
    kept.sort(key=lambda x: x.get("created_utc") or "")
    by_url: dict[str, dict[str, Any]] = {}
    url_dupes: list[dict[str, Any]] = []
    for t in kept:
        u = (t.get("url") or "").rstrip("/").lower()
        if u in by_url:
            url_dupes.append(t)
        else:
            by_url[u] = t
    for t in url_dupes:
        dropped.append(
            {
                "thread_id": t["thread_id"],
                "url": t.get("url") or "",
                "title": t.get("title") or "",
                "drop_reason": "duplicate_url",
            }
        )
    kept = list(by_url.values())
    kept.sort(key=lambda x: x.get("created_utc") or "")

    # ponytail: block by first 24 chars so we don't O(n²) SequenceMatcher all pairs
    final: list[dict[str, Any]] = []
    buckets: dict[str, list[str]] = {}
    for t in kept:
        title = (t.get("title") or "").lower().strip()
        is_dupe = False
        if title:
            key = title[:24]
            for prev in buckets.get(key, []):
                if SequenceMatcher(None, title, prev).ratio() > 0.9:
                    is_dupe = True
                    break
        if is_dupe:
            dropped.append(
                {
                    "thread_id": t["thread_id"],
                    "url": t.get("url") or "",
                    "title": t.get("title") or "",
                    "drop_reason": "duplicate_title",
                }
            )
        else:
            final.append(t)
            if title:
                buckets.setdefault(title[:24], []).append(title)

    return final, dropped
