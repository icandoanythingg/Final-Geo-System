"""Map thread_id / Comment id for a fixed list of Reddit authors.

Standalone — reads arctic_reddit_5d.xlsx only; does not touch other pipeline files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
XLSX = ROOT / "arctic_reddit_5d.xlsx"
OUT = ROOT / "author_thread_comment_ids.csv"

AUTHORS = [
    "ObservrAnirudh",
    "readdituser321",
    "Independent-Sir-7867",
    "PrudentTeam5207",
    "ContemporarySufi",
    "Weekly_Salary4706",
    "Glittering-Exam-4023",
    "Humble_Ebb_3033",
    "wide-bonus-735",
    "Equal-Community-8862",
    "LegalDiscipline8179",
    "JollySundae695",
    "Repulsive-Cream2518",
    "Dramatic-Isopod3549",
    "wadiyanspy",
]


def main() -> None:
    want = {a.lower() for a in AUTHORS}
    df = pd.read_excel(XLSX)
    hit = df[df["author"].astype(str).str.lower().isin(want)].copy()

    out = hit[
        [
            "author",
            "Type",
            "thread_id",
            "Comment id",
            "Parent comment id",
            "Reddit Link",
            "subreddit",
            "title",
            "Comment Text",
            "created_utc",
            "score",
        ]
    ].rename(columns={"Type": "type", "Comment id": "comment_id", "Parent comment id": "parent_comment_id"})

    out = out.sort_values(["author", "created_utc"], kind="stable")
    out.to_csv(OUT, index=False)

    found = set(hit["author"].astype(str).str.lower().unique())
    missing = sorted(want - found)
    print(f"wrote {len(out)} rows -> {OUT.name}")
    print("per author:")
    print(hit["author"].value_counts().to_string())
    if missing:
        print("not found in xlsx:", ", ".join(missing))


if __name__ == "__main__":
    main()
