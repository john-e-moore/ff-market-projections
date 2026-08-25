"""Shared artifact, integrity, and validation-result contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ARTIFACT_NAMES = frozenset({
    "collections_validation.json", "historical_player_seasons.csv", "historical_validation.json",
    "historical_backtest_predictions.csv", "historical_calibration.json", "normalized_markets.csv",
    "normalized_validation.json", "player_map.csv", "name_match_suggestions.csv",
    "identity_validation.json", "priced_markets.csv", "pricing_validation.json",
    "dispersion_calibration.csv", "source_projections.csv", "model_validation.json",
    "consensus_stats.csv", "aggregation_validation.json", "fantasy_projections.csv",
    "scoring_validation.json", "workbook_validation.json",
})


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    severity: str = "error"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CheckResult":
        return cls(**value)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, content: bytes) -> str:
    """Atomically replace a file on its destination filesystem and return its hash."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return sha256_bytes(content)


def atomic_write_json(path: str | Path, value: Any) -> str:
    return atomic_write_bytes(path, canonical_json(value))


def artifact_path(run_dir: str | Path, name: str) -> Path:
    if name not in ARTIFACT_NAMES:
        raise ValueError(f"Unknown artifact name: {name}")
    return Path(run_dir) / "artifacts" / name


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
