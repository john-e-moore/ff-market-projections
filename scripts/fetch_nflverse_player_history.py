#!/usr/bin/env python3
"""Download the official nflverse player-stats release without transforming it."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from urllib.request import Request, urlopen

from ff_market_projections.contracts import atomic_write_bytes, sha256_file, utc_now


def inspect_gzip(path: Path) -> tuple[list[int], list[str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "season" not in fields:
            raise ValueError("nflverse CSV is missing required season column")
        seasons = sorted({int(row["season"]) for row in reader if row.get("season", "").isdigit()})
    if not seasons:
        raise ValueError("nflverse CSV contains no parseable seasons")
    return seasons, fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(args.url, headers={"User-Agent": "ff-market-projections/0.1"}), timeout=args.timeout_seconds) as response:
        content = response.read()
        metadata = {
            "resolved_url": response.geturl(), "status": getattr(response, "status", None),
            "content_type": response.headers.get("Content-Type"), "content_length": response.headers.get("Content-Length"),
            "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag"),
            "downloaded_utc": utc_now(),
        }
    atomic_write_bytes(args.output, content)
    try:
        seasons, fields = inspect_gzip(args.output)
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "bytes": len(content), "gzip_valid": True, "covered_seasons": seasons, "columns": fields, "response": metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
