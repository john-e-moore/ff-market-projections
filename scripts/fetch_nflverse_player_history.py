#!/usr/bin/env python3
"""Download the official nflverse player-stats release without transforming it."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

from ff_market_projections.contracts import atomic_write_bytes, atomic_write_json, sha256_file, utc_now


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


def _load_cache_entry(cache_dir: Path, url: str) -> dict | None:
    index_path = cache_dir / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entry = index.get("entries", {}).get(url)
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
        return None
    cache_file = cache_dir / f"{entry['sha256']}.csv.gz"
    if not cache_file.is_file() or cache_file.stat().st_size == 0:
        return None
    try:
        inspect_gzip(cache_file)
    except (OSError, EOFError, ValueError, gzip.BadGzipFile):
        return None
    return {**entry, "cache_file": cache_file}


def _store_cache(cache_dir: Path, url: str, output: Path, response: dict) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(output)
    cache_file = cache_dir / f"{digest}.csv.gz"
    if not cache_file.exists():
        shutil.copyfile(output, cache_file)
    index_path = cache_dir / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        index = {}
    entries = index.setdefault("entries", {})
    entries[url] = {"sha256": digest, "bytes": output.stat().st_size, "response": response, "cached_utc": utc_now()}
    atomic_write_json(index_path, index)
    return {"hit": False, "path": str(cache_file), "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_dir and not args.refresh_cache:
        entry = _load_cache_entry(args.cache_dir, args.url)
        if entry is not None:
            atomic_write_bytes(args.output, entry["cache_file"].read_bytes())
            seasons, fields = inspect_gzip(args.output)
            print(json.dumps({
                "output": str(args.output), "sha256": sha256_file(args.output), "bytes": args.output.stat().st_size,
                "gzip_valid": True, "covered_seasons": seasons, "columns": fields,
                "response": entry.get("response", {}),
                "cache": {"hit": True, "path": str(entry["cache_file"]), "sha256": entry["sha256"]},
            }, sort_keys=True))
            return
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
    cache = _store_cache(args.cache_dir, args.url, args.output, metadata) if args.cache_dir else None
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "bytes": len(content), "gzip_valid": True, "covered_seasons": seasons, "columns": fields, "response": metadata, "cache": cache}, sort_keys=True))


if __name__ == "__main__":
    main()
