"""Load Reddit / portal FAQ / PAA / PASF / autocomplete / Trends into normalized rows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from lib.load import load_threads
from lib.schema import empty_norm, stable_hash


def _s(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _pick(row: dict[str, Any], *names: str) -> str:
    lower = {str(k).lower().strip(): k for k in row}
    for n in names:
        k = lower.get(n.lower())
        if k is not None and _s(row.get(k)):
            return _s(row[k])
    return ""


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    """CSV or first sheet of xlsx → list of dicts."""
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        return df.to_dict(orient="records")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def match_area(text: str, tags: dict[str, Any]) -> str:
    s = (text or "").strip().lower()
    if not s:
        return "Hyderabad (general)"
    best = None
    best_len = 0
    for name, aliases in (tags.get("areas") or {}).items():
        cands = [name.lower()] + [a.lower() for a in (aliases or [])]
        for a in cands:
            if a and (a == s or a in s) and len(a) > best_len:
                best = name
                best_len = len(a)
    if best is None or best == "Hyderabad":
        return "Hyderabad (general)"
    return best


def ingest_reddit_threads(threads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in threads:
        tid = t["thread_id"]
        title = t.get("title") or ""
        body = t.get("body") or ""
        text = title if not body else f"{title}\n{body}".strip()
        out.append(
            empty_norm(
                source_type="reddit",
                source_name=t.get("subreddit") or "reddit",
                stable_key=tid,
                text=text,
                title=title,
                url=t.get("url") or "",
                date=(t.get("created_utc") or "")[:10],
                meta={"thread_id": tid},
            )
        )
        out[-1]["_thread"] = t
    return out


def ingest_reddit_xlsx(path: Path) -> list[dict[str, Any]]:
    return ingest_reddit_threads(load_threads(path))


def ingest_portal_faq(path: Path) -> list[dict[str, Any]]:
    rows = []
    for i, row in enumerate(_iter_rows(path)):
        question = _pick(row, "question", "faq", "text")
        if not question:
            continue
        portal = _pick(row, "source", "portal", "site") or "portal"
        project = _pick(row, "project")
        area = _pick(row, "area", "locality", "area_a")
        key = stable_hash(portal, project, area, question)
        rows.append(
            empty_norm(
                source_type="portal_faq",
                source_name=portal,
                stable_key=key,
                text=question,
                title=question[:200],
                project=project,
                area_hint=area,
                meta={"row": i, "portal": portal},
            )
        )
    return rows


def ingest_paa(path: Path, tags: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """CSV sample or google_people_also_ask.xlsx — text = question."""
    tags = tags or {}
    rows = []
    for i, row in enumerate(_iter_rows(path)):
        question = _pick(row, "question", "text")
        if not question:
            continue
        seed = _pick(row, "seed", "query")
        position = _pick(row, "position", "rank", "pos")
        area_hint = _pick(row, "area_a", "area", "locality")
        if not area_hint and seed:
            area_hint = match_area(seed, tags) if tags else ""
        key = stable_hash("paa", seed, position, question)
        rows.append(
            empty_norm(
                source_type="paa",
                source_name="google_paa",
                stable_key=key,
                text=question,
                title=question[:200],
                seed=seed,
                area_hint=area_hint,
                meta={"row": i, "position": position},
            )
        )
    return rows


def ingest_pasf(path: Path, tags: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """google_people_also_search.xlsx — text = related_query."""
    tags = tags or {}
    rows = []
    for i, row in enumerate(_iter_rows(path)):
        text = _pick(row, "related_query", "query", "suggestion", "text")
        if not text:
            continue
        seed = _pick(row, "seed")
        position = _pick(row, "rank", "position")
        area_hint = _pick(row, "area_a", "area", "locality")
        if not area_hint:
            area_hint = match_area(text, tags) if tags else ""
        key = stable_hash("pasf", seed, position, text)
        rows.append(
            empty_norm(
                source_type="pasf",
                source_name="google_pasf",
                stable_key=key,
                text=text,
                title=text[:200],
                seed=seed,
                area_hint=area_hint,
                meta={"row": i, "position": position},
            )
        )
    return rows


def ingest_autocomplete(path: Path, tags: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """google_autocomplete_suggestions.xlsx — text = suggestion only; seed blank."""
    tags = tags or {}
    rows = []
    for i, row in enumerate(_iter_rows(path)):
        suggestion = _pick(row, "suggestion", "text")
        if not suggestion:
            continue
        position = _pick(row, "rank", "position")
        area_hint = match_area(suggestion, tags) if tags else ""
        key = stable_hash("autocomplete", suggestion, position)
        rows.append(
            empty_norm(
                source_type="autocomplete",
                source_name="google_autocomplete",
                stable_key=key,
                text=suggestion,
                title=suggestion[:200],
                seed="",  # user: suggestion only
                area_hint=area_hint,
                meta={
                    "row": i,
                    "position": position,
                    "scraped_at": _pick(row, "scraped_at"),
                },
            )
        )
    return rows


def ingest_trends(path: Path, tags: dict[str, Any]) -> list[dict[str, Any]]:
    """Signal rows — no LLM. Map seed → area."""
    rows = []
    for i, row in enumerate(_iter_rows(path)):
        seed = _pick(row, "seed", "keyword", "query")
        if not seed:
            continue
        direction = _pick(row, "direction", "trend")
        related = _pick(row, "top_related", "related")
        category = _pick(row, "category")
        interest_vals = []
        for k, v in row.items():
            if k and "interest" in str(k).lower() and _s(v):
                interest_vals.append(_s(v))
        area = match_area(seed, tags)
        key = stable_hash(seed, category, direction)
        rows.append(
            empty_norm(
                source_type="trends",
                source_name="google_trends",
                stable_key=key,
                text=seed,
                title=seed,
                seed=seed,
                area_hint=area,
                meta={
                    "row": i,
                    "category": category,
                    "direction": direction,
                    "top_related": related,
                    "interest": interest_vals,
                },
            )
        )
        rows[-1]["area"] = area
        rows[-1]["topic"] = ""
        rows[-1]["persona"] = ""
        rows[-1]["stage"] = ""
        rows[-1]["project"] = ""
    return rows


def load_extra_sources(
    *,
    portal_faq: Path | None,
    paa: Path | None,
    pasf: Path | None,
    autocomplete: Path | None,
    trends: Path | None,
    tags: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {
        "portal_faq": [],
        "paa": [],
        "pasf": [],
        "autocomplete": [],
        "trends": [],
    }
    if portal_faq and portal_faq.exists():
        out["portal_faq"] = ingest_portal_faq(portal_faq)
    if paa and paa.exists():
        out["paa"] = ingest_paa(paa, tags)
    if pasf and pasf.exists():
        out["pasf"] = ingest_pasf(pasf, tags)
    if autocomplete and autocomplete.exists():
        out["autocomplete"] = ingest_autocomplete(autocomplete, tags)
    if trends and trends.exists():
        out["trends"] = ingest_trends(trends, tags)
    return out


# back-compat alias
def load_csv_sources(**kwargs: Any) -> dict[str, list[dict[str, Any]]]:
    kwargs.setdefault("pasf", None)
    kwargs.setdefault("autocomplete", None)
    return load_extra_sources(**kwargs)
