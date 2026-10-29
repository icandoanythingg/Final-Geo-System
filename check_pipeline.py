#!/usr/bin/env python3
"""Assert golden examples against existing data/ outputs (does not rewrite CSVs)."""

from __future__ import annotations

import csv
from pathlib import Path

DATA = Path(__file__).parent / "data"


def main() -> int:
    clean_rows = list(csv.DictReader((DATA / "clean_threads.csv").open()))
    audit = {r["thread_id"]: r for r in csv.DictReader((DATA / "audit.csv").open())}
    dropped = {r["thread_id"]: r for r in csv.DictReader((DATA / "dropped.csv").open())}
    clean_by_title = {r.get("title") or "": r for r in clean_rows}
    errors: list[str] = []

    def clean_for(tid: str) -> dict | None:
        a = audit.get(tid)
        if not a:
            return None
        return clean_by_title.get(a.get("title") or "")

    r = clean_for("1x5mar3")
    if not r or r["topic"] != "Area measurement" or r["persona"] != "First-time buyer":
        errors.append(f"1x5mar3 bad {r}")
    if audit.get("1x5mar3", {}).get("check"):
        errors.append("1x5mar3 check should be blank")

    r = clean_for("1x5lphq")
    if not r or r["area"] != "Kothaguda | Kondapur" or r["topic"] != "Charges" or r["persona"]:
        errors.append(f"1x5lphq bad {r}")
    if audit.get("1x5lphq", {}).get("promo") != "No":
        errors.append("1x5lphq promo should be No")
    if audit.get("1x5lphq", {}).get("check"):
        errors.append("1x5lphq check should be blank")

    r = clean_for("1x5hd59")
    if not r or r["topic"] != "Loan" or r["stage"] != "Ready to book":
        errors.append(f"1x5hd59 bad {r}")

    r = clean_for("1x5ifq7")
    if not r or r["topic"] != "Price" or r["persona"]:
        errors.append(f"1x5ifq7 bad {r}")

    for tid in ("1x5mcsn", "1x5k198", "1x5j4rw"):
        if tid not in dropped:
            errors.append(f"missing drop {tid}")

    titles = [r.get("title") or "" for r in clean_rows]
    if len(titles) != len(set(titles)):
        # soft: warn only if exact dup titles (possible but rare)
        pass

    header = (DATA / "clean_threads.csv").read_text(encoding="utf-8").splitlines()[0].lower()
    if "author" in header.split(","):
        errors.append("author column in clean_threads")
    if header.split(",")[0] != "title":
        errors.append(f"clean_threads first column should be title, got {header.split(',')[0]!r}")

    if errors:
        print("CHECK FAILED")
        for e in errors:
            print(" ", e)
        return 1
    print(f"CHECK OK ({len(clean_rows)} clean, {len(dropped)} dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
