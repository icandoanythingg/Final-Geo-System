"""Group tagged question evidence into piles for prompt writing."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from lib.schema import QUESTION_SOURCES

CLUSTER_FIELDS = [
    "cluster_id",
    "area",
    "topic",
    "stage",
    "count",
    "source_types",
    "example_texts",
    "persona_hints",
    "evidence_ids",
]


def cluster_key(row: dict[str, Any]) -> str:
    area = (row.get("area") or "Hyderabad (general)").strip()
    topic = (row.get("topic") or "Other").strip()
    stage = (row.get("stage") or "").strip()
    return f"{area}|{topic}|{stage}"


def build_clusters(
    evidence_rows: list[dict[str, Any]],
    *,
    min_count: int = 3,
) -> list[dict[str, Any]]:
    """
    Cluster question sources only (not Trends).
    Keep cluster if count >= min_count OR 2+ distinct source_types.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in evidence_rows:
        if r.get("source_type") not in QUESTION_SOURCES:
            continue
        if not (r.get("topic") or "").strip():
            continue
        buckets[cluster_key(r)].append(r)

    out: list[dict[str, Any]] = []
    for key, items in buckets.items():
        source_types = sorted({i.get("source_type") or "" for i in items if i.get("source_type")})
        count = len(items)
        if count < min_count and len(source_types) < 2:
            continue
        area, topic, stage = key.split("|", 2)
        personas = []
        for i in items:
            p = (i.get("persona") or "").strip()
            if p and p not in personas:
                personas.append(p)
        texts = []
        for i in items:
            t = (i.get("text") or i.get("title") or "").strip()
            if t and t not in texts:
                texts.append(t)
            if len(texts) >= 5:
                break
        ids = [i["evidence_id"] for i in items]
        # stable cluster id from key
        cid = "c:" + key.replace(" ", "_").replace("|", "__")[:120]
        out.append(
            {
                "cluster_id": cid,
                "area": area,
                "topic": topic,
                "stage": stage,
                "count": count,
                "source_types": ",".join(source_types),
                "example_texts": " || ".join(texts),
                "persona_hints": ", ".join(personas),
                "evidence_ids": " ".join(ids[:50]),
            }
        )
    out.sort(key=lambda r: (-int(r["count"]), r["area"], r["topic"], r["stage"]))
    return out
