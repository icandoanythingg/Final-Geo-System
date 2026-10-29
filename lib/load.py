"""Load arctic Reddit xlsx into thread dicts. Authors are discarded after grouping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


COLS = {
    "subreddit": "subreddit",
    "kind": "Type",
    "url": "Reddit Link",
    "thread_id": "thread_id",
    "parent_id": "Parent comment id",
    "comment_id": "Comment id",
    "author": "author",
    "title": "title",
    "text": "Comment Text",
    "score": "score",
    "num_comments": "num_comments",
    "created_utc": "created_utc",
}


def _s(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def load_threads(path: Path) -> list[dict[str, Any]]:
    df = pd.read_excel(path)
    for need in COLS.values():
        if need not in df.columns:
            raise ValueError(f"missing column {need!r} in {path}")

    groups: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        tid = _s(row[COLS["thread_id"]])
        if not tid:
            continue
        kind = _s(row[COLS["kind"]]).lower()
        is_post = kind in ("posts", "post", "thread")
        text = _s(row[COLS["text"]])
        title = _s(row[COLS["title"]])
        url = _s(row[COLS["url"]])
        created = _s(row[COLS["created_utc"]])
        score = row[COLS["score"]]
        try:
            score_i = int(float(score)) if pd.notna(score) else 0
        except (TypeError, ValueError):
            score_i = 0

        g = groups.get(tid)
        if g is None:
            g = {
                "thread_id": tid,
                "subreddit": _s(row[COLS["subreddit"]]),
                "url": "",
                "title": "",
                "body": "",
                "created_utc": created,
                "score": 0,
                "comments": [],  # list of {text, score, created_utc} — no author
                "op_author": "",  # kept only to detect OP replies; stripped before write
            }
            groups[tid] = g

        if is_post:
            g["title"] = title or g["title"]
            g["body"] = text
            g["url"] = url or g["url"]
            g["created_utc"] = created or g["created_utc"]
            g["score"] = score_i
            g["op_author"] = _s(row[COLS["author"]])
            g["subreddit"] = _s(row[COLS["subreddit"]]) or g["subreddit"]
        else:
            g["comments"].append(
                {"text": text, "score": score_i, "created_utc": created}
            )
            if not g["url"] and url:
                # comment permalink → thread url
                base = url.split("/comment/")[0].rstrip("/") + "/"
                g["url"] = base
            if not g["title"] and title:
                g["title"] = title

    threads = list(groups.values())
    for t in threads:
        t["comments"].sort(key=lambda c: (-c["score"], c["created_utc"]))
        if not t["url"]:
            t["url"] = f"https://www.reddit.com/comments/{t['thread_id']}/"
    return threads
