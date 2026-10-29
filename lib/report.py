"""Step 5: write CSVs and run_report.txt."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lib.cluster import CLUSTER_FIELDS, build_clusters
from lib.schema import (
    CLEAN_EVIDENCE_FIELDS,
    EVIDENCE_AUDIT_FIELDS,
    QUESTION_SOURCES,
    to_clean_evidence,
    to_evidence_audit,
)


CLEAN_FIELDS = ["title", "topic", "persona", "stage", "area"]
AUDIT_FIELDS = [
    "thread_id",
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
]
DROP_FIELDS = ["thread_id", "url", "title", "drop_reason"]
NEW_FIELDS = ["label", "count", "example_thread_ids"]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _row_id(r: dict[str, Any]) -> str:
    return r.get("evidence_id") or r.get("thread_id") or ""


def _agg_new(
    rows: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        val = r.get(key) or ""
        rid = _row_id(r)
        if key == "new_areas":
            for a in r.get("new_areas") or []:
                if a:
                    buckets[a].append(rid)
            continue
        if val:
            buckets[str(val)].append(rid)
    out = []
    for label, ids in sorted(buckets.items(), key=lambda x: (-len(x[1]), x[0])):
        uniq = list(dict.fromkeys(ids))
        out.append(
            {
                "label": label,
                "count": len(uniq),
                "example_thread_ids": " ".join(uniq[:10]),
            }
        )
    return out


_STOP = set(
    "a an the and or to of in on for is are was were be been being this that it "
    "with as at by from my we you i me our your they them their not no yes if "
    "but so just about into over out up down any all can would could should "
    "what when where which who how do does did have has had will am".split()
)


def untagged_words(rows: list[dict[str, Any]], threads_by_id: dict[str, dict]) -> list[dict[str, Any]]:
    """Top 100 words/bigrams in threads whose topic stayed Other (untagged)."""
    word_c: Counter[str] = Counter()
    bi_c: Counter[str] = Counter()
    for r in rows:
        if r.get("topic") not in ("Other", "Untagged", ""):
            continue
        t = threads_by_id.get(r["thread_id"]) or {}
        text = f"{t.get('title','')} {t.get('body','')}".lower()
        toks = re.findall(r"[a-z]{3,}", text)
        toks = [w for w in toks if w not in _STOP]
        word_c.update(toks)
        bi_c.update(f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1))
    ranked = word_c.most_common(70) + bi_c.most_common(30)
    ranked = sorted(ranked, key=lambda x: (-x[1], x[0]))[:100]
    return [{"label": w, "count": c, "example_thread_ids": ""} for w, c in ranked]


def write_outputs(
    out_dir: Path,
    clean_rows: list[dict[str, Any]],
    dropped: list[dict[str, str]],
    threads_by_id: dict[str, dict],
    stats: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # stable sort for reproducibility
    clean_rows = sorted(clean_rows, key=lambda r: r["thread_id"])
    dropped = sorted(dropped, key=lambda r: (r.get("drop_reason") or "", r["thread_id"]))

    _write_csv(
        out_dir / "clean_threads.csv",
        CLEAN_FIELDS,
        clean_rows,
    )
    _write_csv(out_dir / "audit.csv", AUDIT_FIELDS, clean_rows)
    _write_csv(out_dir / "dropped.csv", DROP_FIELDS, dropped)

    new_p = _agg_new(clean_rows, "new_persona")
    new_t = _agg_new(clean_rows, "new_topic")
    new_a = _agg_new(clean_rows, "new_areas")
    _write_csv(out_dir / "new_personas.csv", NEW_FIELDS, new_p)
    _write_csv(out_dir / "new_topics.csv", NEW_FIELDS, new_t)
    _write_csv(out_dir / "new_areas.csv", NEW_FIELDS, new_a)

    uw = untagged_words(clean_rows, threads_by_id)
    _write_csv(out_dir / "untagged_words.csv", NEW_FIELDS, uw)

    # report
    drop_reasons = Counter(d["drop_reason"] for d in dropped)
    topic_counts = Counter(r["topic"] for r in clean_rows)
    n = len(clean_rows) or 1
    empty_persona = sum(1 for r in clean_rows if not r.get("persona"))
    empty_stage = sum(1 for r in clean_rows if not r.get("stage"))
    other_topic = sum(1 for r in clean_rows if r.get("topic") in ("Other", "Untagged"))
    check_reasons: Counter[str] = Counter()
    for r in clean_rows:
        ch = r.get("check") or ""
        if not ch:
            continue
        for part in ch.split(";"):
            part = part.strip()
            if part:
                check_reasons[part] += 1

    rng = random.Random(42)
    sample_drops = [d["title"] for d in dropped if d.get("title")]
    rng.shuffle(sample_drops)
    sample_drops = sample_drops[:20]

    blind = stats.get("blind_agreement") or {}
    lines = [
        "RUN REPORT",
        f"total_threads: {stats.get('total_threads', 0)}",
        f"kept: {len(clean_rows)}",
        f"dropped: {len(dropped)}",
        "dropped_by_reason:",
    ]
    for k, v in sorted(drop_reasons.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {k}: {v}")
    lines.append("20_random_dropped_titles:")
    for t in sample_drops:
        lines.append(f"  - {t[:120]}")
    lines.append("rows_per_topic:")
    for k, v in sorted(topic_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {k}: {v}")
    lines.append(f"pct_empty_persona: {100*empty_persona/n:.1f}")
    lines.append(f"pct_empty_stage: {100*empty_stage/n:.1f}")
    lines.append(f"pct_Other_topic: {100*other_topic/n:.1f}")
    lines.append(f"llm_calls: {stats.get('llm_calls', 0)}")
    lines.append(f"cache_hits: {stats.get('cache_hits', 0)}")
    lines.append(f"synth_fills: {stats.get('synth_fills', 0)}")
    lines.append(f"llm_failures: {stats.get('llm_failures', 0)}")
    lines.append(f"quote_failures: {stats.get('quote_failures', 0)}")
    lines.append("check_by_reason:")
    for k, v in sorted(check_reasons.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {k}: {v}")
    lines.append("blind_sample_agreement_pct:")
    for field in ("topic", "persona", "stage", "area", "promo"):
        lines.append(f"  {field}: {blind.get(field, 'n/a')}")
    lines.append(f"blind_sample_n: {blind.get('n', 0)}")
    lines.append("top_10_new_personas:")
    for row in new_p[:10]:
        lines.append(f"  {row['label']}: {row['count']}")
    lines.append("top_10_new_topics:")
    for row in new_t[:10]:
        lines.append(f"  {row['label']}: {row['count']}")
    lines.append("top_10_new_areas:")
    for row in new_a[:10]:
        lines.append(f"  {row['label']}: {row['count']}")

    report = "\n".join(lines) + "\n"
    (out_dir / "run_report.txt").write_text(report, encoding="utf-8")
    print(report)


def write_evidence_outputs(
    out_dir: Path,
    evidence_rows: list[dict[str, Any]],
    *,
    reddit_clean: list[dict[str, Any]] | None = None,
    reddit_dropped: list[dict[str, str]] | None = None,
    threads_by_id: dict[str, dict] | None = None,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Write global shelf + clusters; optionally Reddit clean_threads view."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = stats or {}
    threads_by_id = threads_by_id or {}

    evidence_rows = sorted(
        evidence_rows, key=lambda r: (r.get("source_type") or "", r.get("evidence_id") or "")
    )
    clean_ev = [to_clean_evidence(r) for r in evidence_rows]
    audit_ev = [to_evidence_audit(r) for r in evidence_rows]
    _write_csv(out_dir / "clean_evidence.csv", CLEAN_EVIDENCE_FIELDS, clean_ev)
    _write_csv(out_dir / "evidence_audit.csv", EVIDENCE_AUDIT_FIELDS, audit_ev)

    clusters = build_clusters(evidence_rows)
    _write_csv(out_dir / "clusters.csv", CLUSTER_FIELDS, clusters)

    # new_* / untagged over question sources only
    q_rows = [r for r in evidence_rows if r.get("source_type") in QUESTION_SOURCES]
    new_p = _agg_new(q_rows, "new_persona")
    new_t = _agg_new(q_rows, "new_topic")
    new_a = _agg_new(q_rows, "new_areas")
    _write_csv(out_dir / "new_personas.csv", NEW_FIELDS, new_p)
    _write_csv(out_dir / "new_topics.csv", NEW_FIELDS, new_t)
    _write_csv(out_dir / "new_areas.csv", NEW_FIELDS, new_a)

    word_c: Counter[str] = Counter()
    bi_c: Counter[str] = Counter()
    for r in q_rows:
        if r.get("topic") not in ("Other", "Untagged", ""):
            continue
        text = (r.get("text") or "").lower()
        if r.get("source_type") == "reddit":
            meta = r.get("meta") if isinstance(r.get("meta"), dict) else {}
            tid = (meta or {}).get("thread_id") or str(r.get("evidence_id", "")).removeprefix(
                "reddit:"
            )
            t = threads_by_id.get(tid) or {}
            blob = f"{t.get('title', '')} {t.get('body', '')}".lower().strip()
            if blob:
                text = blob
        toks = [w for w in re.findall(r"[a-z]{3,}", text) if w not in _STOP]
        word_c.update(toks)
        bi_c.update(f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1))
    ranked = sorted(
        (word_c.most_common(70) + bi_c.most_common(30)), key=lambda x: (-x[1], x[0])
    )[:100]
    _write_csv(
        out_dir / "untagged_words.csv",
        NEW_FIELDS,
        [{"label": w, "count": c, "example_thread_ids": ""} for w, c in ranked],
    )

    # Reddit view
    if reddit_clean is not None:
        rc = sorted(reddit_clean, key=lambda r: r.get("thread_id") or "")
        _write_csv(out_dir / "clean_threads.csv", CLEAN_FIELDS, rc)
        _write_csv(out_dir / "audit.csv", AUDIT_FIELDS, rc)
    if reddit_dropped is not None:
        dropped = sorted(
            reddit_dropped, key=lambda r: (r.get("drop_reason") or "", r.get("thread_id") or "")
        )
        _write_csv(out_dir / "dropped.csv", DROP_FIELDS, dropped)

    by_src = Counter(r.get("source_type") for r in evidence_rows)
    lines = [
        "EVIDENCE RUN REPORT",
        f"total_evidence: {len(evidence_rows)}",
        "by_source:",
    ]
    for k, v in sorted(by_src.items(), key=lambda x: (-x[1], x[0] or "")):
        lines.append(f"  {k}: {v}")
    lines.append(f"clusters: {len(clusters)}")
    lines.append(f"llm_calls: {stats.get('llm_calls', 0)}")
    lines.append(f"cache_hits: {stats.get('cache_hits', 0)}")
    lines.append(f"synth_fills: {stats.get('synth_fills', 0)}")
    lines.append(f"llm_failures: {stats.get('llm_failures', 0)}")
    if reddit_clean is not None:
        lines.append(f"reddit_kept: {len(reddit_clean)}")
    if reddit_dropped is not None:
        lines.append(f"reddit_dropped: {len(reddit_dropped)}")
    # Trends area popularity (signal only)
    trend_areas = Counter(
        r.get("area")
        for r in evidence_rows
        if r.get("source_type") == "trends" and r.get("area")
    )
    if trend_areas:
        lines.append("trends_area_counts:")
        for k, v in trend_areas.most_common(15):
            lines.append(f"  {k}: {v}")
    report = "\n".join(lines) + "\n"
    (out_dir / "evidence_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return clusters
