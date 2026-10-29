#!/usr/bin/env python3
"""Global evidence pipeline → clean_evidence + clusters.

Sources: reddit, portal_faq, paa, pasf, autocomplete, trends.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.cluster import build_clusters  # noqa: E402
from lib.filter import filter_threads  # noqa: E402
from lib.ingest import load_extra_sources  # noqa: E402
from lib.llm import COMPACT_SYSTEM, LlmCache, tag_thread  # noqa: E402
from lib.load import load_threads  # noqa: E402
from lib.merge import merge_row  # noqa: E402
from lib.report import write_evidence_outputs  # noqa: E402
from lib.rules import build_hints  # noqa: E402
from lib.synth import synth_from_rules  # noqa: E402
from lib.tag_item import (  # noqa: E402
    reddit_merged_to_evidence,
    tag_question_item,
    tag_trends_item,
)

log = logging.getLogger("run_evidence")

DEFAULT_XLSX = ROOT / "arctic_reddit_5d.xlsx"
CONFIG_TAGS = ROOT / "config" / "tags.yaml"
DATA = ROOT / "data"
CACHE_PATH = DATA / "llm_cache.jsonl"
FIXTURES = ROOT / "fixtures"
DEFAULT_PORTAL = ROOT / "portal_faq_sample.csv"
DEFAULT_PAA = ROOT / "google_people_also_ask.xlsx"
DEFAULT_PASF = ROOT / "google_people_also_search.xlsx"
DEFAULT_AUTOCOMPLETE = ROOT / "google_autocomplete_suggestions.xlsx"
DEFAULT_TRENDS = ROOT / "trends_sample.csv"
# fixtures used only if samples missing
if not DEFAULT_PORTAL.exists():
    DEFAULT_PORTAL = FIXTURES / "portal_faqs.csv"
if not DEFAULT_PAA.exists():
    DEFAULT_PAA = FIXTURES / "paa.csv"
if not DEFAULT_TRENDS.exists():
    DEFAULT_TRENDS = FIXTURES / "trends.csv"

from clean_threads import FIXTURE_LLM, seed_fixture_cache, _acceptance as acceptance_check  # noqa: E402


def load_tags(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _process_reddit(
    xlsx: Path,
    tags: dict[str, Any],
    prompt: str,
    cache: LlmCache,
    *,
    skip_llm: bool,
    limit: int | None,
    seed: int,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], dict[str, dict], dict[str, Any]]:
    known_areas = set(tags.get("areas") or {})
    known_topics = set(tags.get("topics") or {})
    known_personas = set(tags.get("personas") or {})

    def synth_fn(thread: dict[str, Any], hints: dict[str, Any]) -> dict[str, Any]:
        return synth_from_rules(thread, hints, tags)

    threads = load_threads(xlsx)
    total = len(threads)
    if limit is not None:
        want = set(FIXTURE_LLM) | {"1x5mcsn", "1x5k198", "1x5j4rw"}
        picked = [t for t in threads if t["thread_id"] in want]
        rest = [t for t in threads if t["thread_id"] not in want]
        threads = picked + rest[: max(0, limit - len(picked))]

    area_aliases: list[str] = []
    for aliases in (tags.get("areas") or {}).values():
        area_aliases.extend(aliases or [])
    kept, dropped = filter_threads(
        threads,
        tags.get("housing_words") or [],
        area_aliases=area_aliases,
    )
    threads_by_id = {t["thread_id"]: t for t in threads}
    if seed_fixture_cache:
        seed_fixture_cache(cache, {t["thread_id"]: t for t in kept})

    stats = {
        "total_threads": total if limit is None else len(threads),
        "llm_calls": 0,
        "cache_hits": 0,
        "llm_failures": 0,
        "quote_failures": 0,
        "synth_fills": 0,
    }

    def process_one(t: dict[str, Any]) -> dict[str, Any]:
        hints = build_hints(t, tags)
        llm_v, st = tag_thread(
            t,
            hints,
            prompt,
            cache,
            skip_llm=skip_llm,
            known_topics=known_topics,
            known_personas=known_personas,
            known_areas=known_areas,
            synth_fn=synth_fn,
        )
        row = merge_row(
            t,
            hints,
            llm_v,
            known_personas=known_personas,
            known_topics=known_topics,
            is_sample=False,
        )
        for a in llm_v.get("new_areas") or []:
            if a and a not in known_areas and a not in (row.get("new_areas") or []):
                row.setdefault("new_areas", []).append(a)
        row["_st"] = st
        return row

    clean_rows: list[dict[str, Any]] = []
    if skip_llm or workers <= 1:
        for t in kept:
            clean_rows.append(process_one(t))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_one, t): t["thread_id"] for t in kept}
            for fut in as_completed(futs):
                clean_rows.append(fut.result())

    for r in clean_rows:
        st = r.pop("_st", {})
        stats["llm_calls"] += st.get("llm_calls", 0)
        stats["cache_hits"] += 1 if st.get("cache_hit") else 0
        stats["synth_fills"] += 1 if st.get("synth") else 0
        stats["llm_failures"] += 1 if st.get("failed") else 0
        if "llm quote failed" in (r.get("check") or ""):
            stats["quote_failures"] += 1

    evidence = [
        reddit_merged_to_evidence(r, threads_by_id.get(r["thread_id"]) or {"thread_id": r["thread_id"]})
        for r in clean_rows
    ]
    # attach subreddit from kept threads
    kept_by_id = {t["thread_id"]: t for t in kept}
    for e in evidence:
        tid = (e.get("meta") or {}).get("thread_id")
        t = kept_by_id.get(tid) or threads_by_id.get(tid)
        if t:
            e["source_name"] = t.get("subreddit") or e["source_name"]
    return evidence, clean_rows, dropped, threads_by_id, stats


def run(
    *,
    sources: list[str],
    xlsx: Path,
    portal_faq: Path,
    paa: Path,
    pasf: Path,
    autocomplete: Path,
    trends: Path,
    skip_llm: bool = False,
    limit: int | None = None,
    seed: int = 42,
    workers: int = 4,
    self_check: bool = False,
    reddit_self_check: bool = False,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tags = load_tags(CONFIG_TAGS)
    prompt = COMPACT_SYSTEM
    known_areas = set(tags.get("areas") or {})
    known_topics = set(tags.get("topics") or {})
    known_personas = set(tags.get("personas") or {})
    cache = LlmCache(CACHE_PATH)

    evidence: list[dict[str, Any]] = []
    reddit_clean: list[dict[str, Any]] | None = None
    reddit_dropped: list[dict[str, str]] | None = None
    threads_by_id: dict[str, dict] = {}
    stats: dict[str, Any] = {
        "llm_calls": 0,
        "cache_hits": 0,
        "synth_fills": 0,
        "llm_failures": 0,
        "quote_failures": 0,
    }

    if "reddit" in sources:
        if not xlsx.exists():
            log.warning("reddit xlsx missing: %s", xlsx)
        else:
            ev, clean, dropped, threads_by_id, st = _process_reddit(
                xlsx,
                tags,
                prompt,
                cache,
                skip_llm=skip_llm,
                limit=limit,
                seed=seed,
                workers=workers,
            )
            evidence.extend(ev)
            reddit_clean = clean
            reddit_dropped = dropped
            for k in stats:
                stats[k] = stats.get(k, 0) + st.get(k, 0)

    loaded = load_extra_sources(
        portal_faq=portal_faq if "portal_faq" in sources else None,
        paa=paa if "paa" in sources else None,
        pasf=pasf if "pasf" in sources else None,
        autocomplete=autocomplete if "autocomplete" in sources else None,
        trends=trends if "trends" in sources else None,
        tags=tags,
    )

    # Trends: signal only
    for norm in loaded.get("trends") or []:
        evidence.append(tag_trends_item(norm))

    # Question-like: FAQ + PAA + PASF + autocomplete → shared tag path
    q_norms: list[dict[str, Any]] = []
    for key in ("portal_faq", "paa", "pasf", "autocomplete"):
        chunk = list(loaded.get(key) or [])
        if limit is not None:
            chunk = chunk[:limit]
        q_norms.extend(chunk)

    def tag_one(norm: dict[str, Any]) -> tuple[dict[str, Any], dict]:
        return tag_question_item(
            norm,
            tags,
            prompt,
            cache,
            skip_llm=skip_llm,
            known_topics=known_topics,
            known_personas=known_personas,
            known_areas=known_areas,
            tag_thread_fn=tag_thread,
        )

    if q_norms:
        if skip_llm or workers <= 1:
            results = [tag_one(n) for n in q_norms]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(tag_one, n) for n in q_norms]
                for fut in as_completed(futs):
                    results.append(fut.result())
        for row, st in results:
            evidence.append(row)
            stats["llm_calls"] += st.get("llm_calls", 0)
            stats["cache_hits"] += 1 if st.get("cache_hit") else 0
            stats["synth_fills"] += 1 if st.get("synth") else 0
            stats["llm_failures"] += 1 if st.get("failed") else 0

    write_evidence_outputs(
        DATA,
        evidence,
        reddit_clean=reddit_clean,
        reddit_dropped=reddit_dropped,
        threads_by_id=threads_by_id,
        stats=stats,
    )

    ok = True
    if self_check:
        ok = evidence_self_check(evidence, sources) and ok
    if reddit_self_check and reddit_clean is not None and acceptance_check:
        ok = acceptance_check(reddit_clean, reddit_dropped or []) and ok
    elif self_check and "reddit" not in sources:
        pass
    return 0 if ok else 1


def evidence_self_check(evidence: list[dict[str, Any]], sources: list[str]) -> bool:
    errors: list[str] = []
    by_src: dict[str, list[dict[str, Any]]] = {}
    for r in evidence:
        by_src.setdefault(r.get("source_type") or "", []).append(r)

    for need in sources:
        if need == "reddit":
            continue
        if need not in by_src:
            errors.append(f"missing source_type={need} in clean_evidence")

    if "paa" in by_src:
        paa = by_src["paa"]
        if not any((r.get("text") or "").strip() for r in paa):
            errors.append("paa rows missing text")
        if not any((r.get("topic") or "").strip() for r in paa):
            errors.append("paa rows missing topic")

    if "pasf" in by_src:
        if not any((r.get("text") or "").strip() for r in by_src["pasf"]):
            errors.append("pasf rows missing text")

    if "autocomplete" in by_src:
        ac = by_src["autocomplete"]
        if not any((r.get("text") or "").strip() for r in ac):
            errors.append("autocomplete rows missing text/suggestion")
        if any((r.get("seed") or "").strip() for r in ac):
            errors.append("autocomplete should leave seed blank")

    for r in by_src.get("trends") or []:
        if (r.get("topic") or "").strip():
            errors.append(f"trends row should have blank topic: {r.get('evidence_id')}")
        if not (r.get("area") or "").strip():
            errors.append(f"trends row missing area: {r.get('evidence_id')}")

    if errors:
        print("EVIDENCE SELF-CHECK FAILED:")
        for e in errors:
            print(" ", e)
        return False
    print("EVIDENCE SELF-CHECK OK:", ", ".join(sorted(by_src)))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Global evidence pipeline")
    ap.add_argument(
        "--sources",
        default="portal_faq,paa,pasf,autocomplete,trends",
        help="Comma list: reddit,portal_faq,paa,pasf,autocomplete,trends",
    )
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--portal-faq", type=Path, default=DEFAULT_PORTAL)
    ap.add_argument("--paa", type=Path, default=DEFAULT_PAA)
    ap.add_argument("--pasf", type=Path, default=DEFAULT_PASF)
    ap.add_argument("--autocomplete", type=Path, default=DEFAULT_AUTOCOMPLETE)
    ap.add_argument("--trends", type=Path, default=DEFAULT_TRENDS)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit reddit threads and each non-reddit question source",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--reddit-self-check", action="store_true")
    args = ap.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    return run(
        sources=sources,
        xlsx=args.xlsx,
        portal_faq=args.portal_faq,
        paa=args.paa,
        pasf=args.pasf,
        autocomplete=args.autocomplete,
        trends=args.trends,
        skip_llm=args.skip_llm,
        limit=args.limit,
        seed=args.seed,
        workers=args.workers,
        self_check=args.self_check,
        reddit_self_check=args.reddit_self_check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
