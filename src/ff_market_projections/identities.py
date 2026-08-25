"""Conservative, auditable player identity reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from rapidfuzz.fuzz import ratio

from .contracts import CheckResult, atomic_write_bytes


_SOURCE_NAMES = frozenset({"draftkings", "fanduel", "kalshi"})
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_REQUIRED_ALIAS_COLUMNS = (
    "source", "raw_player_name", "canonical_player_id", "canonical_player_name",
)
IDENTITY_COLUMNS = (
    "canonical_player_id", "canonical_player_name", "canonical_position",
    "canonical_gsis_player_id", "identity_match_method", "identity_match_score",
    "identity_review_status",
)


class IdentityError(ValueError):
    """An alias or reconciliation result violates an identity hard gate."""

    def __init__(self, message: str, validation: dict[str, Any]) -> None:
        super().__init__(message)
        self.validation = validation


@dataclass(frozen=True)
class Alias:
    source: str
    raw_player_name: str
    normalized_match_key: str
    canonical_player_id: str
    canonical_player_name: str
    alias_file_row: int


@dataclass(frozen=True)
class _Candidate:
    canonical_player_id: str
    canonical_player_name: str
    match_keys: frozenset[str]
    stats: frozenset[str]
    position: str | None
    gsis_player_id: str | None
    teams: frozenset[str]
    source: str | None = None


@dataclass(frozen=True)
class _Identity:
    source: str
    raw_player_name: str
    normalized_match_key: str
    stats: frozenset[str]
    teams: frozenset[str]


@dataclass(frozen=True)
class IdentityReconciliation:
    rows: list[dict[str, Any]]
    player_map: list[dict[str, Any]]
    suggestions: list[dict[str, Any]]
    validation: dict[str, Any]


def normalize_match_key(value: str | None) -> str:
    """Normalize a display name without changing the preserved source display name."""

    if value is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    folded = without_marks.casefold().replace("-", " ")
    words = re.sub(r"[^a-z0-9\s]", "", folded).split()
    while words and words[-1] in _SUFFIXES:
        words.pop()
    return " ".join(words)


def _slug_plus_hash(match_key: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", match_key).strip("-") or "unknown-player"
    digest = hashlib.sha256(match_key.encode("utf-8")).hexdigest()[:12]
    return f"player:{slug}-{digest}"


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _check(name: str, passed: bool, *, severity: str = "error", message: str = "", **details: Any) -> CheckResult:
    return CheckResult(name, passed, severity, message, details)


def load_aliases(path: str | Path) -> list[Alias]:
    """Read aliases, rejecting ambiguous duplicate source/name instructions."""

    alias_path = Path(path)
    try:
        with alias_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            missing = sorted(set(_REQUIRED_ALIAS_COLUMNS) - set(fields))
            if missing:
                raise IdentityError(
                    f"Alias file is missing required column(s): {', '.join(missing)}",
                    {"status": "failed", "checks": [_check("identity.alias_columns", False, message="alias file must expose required columns", missing=missing).to_dict()]},
                )
            aliases: list[Alias] = []
            for row_number, row in enumerate(reader, start=2):
                source = _text(row.get("source"))
                raw_name = _text(row.get("raw_player_name"))
                canonical_id = _text(row.get("canonical_player_id"))
                canonical_name = _text(row.get("canonical_player_name"))
                if source not in _SOURCE_NAMES | {"*"} or not raw_name or not canonical_id or not canonical_name:
                    raise IdentityError(
                        f"Invalid alias at row {row_number}",
                        {"status": "failed", "checks": [_check("identity.alias_values", False, message="aliases require a known source (or *), raw name, canonical ID, and canonical name", row=row_number).to_dict()]},
                    )
                key = normalize_match_key(raw_name)
                if not key:
                    raise IdentityError(
                        f"Alias row {row_number} has an empty normalized name",
                        {"status": "failed", "checks": [_check("identity.alias_keys", False, message="alias names must normalize to a nonempty key", row=row_number).to_dict()]},
                    )
                aliases.append(Alias(source, raw_name, key, canonical_id, canonical_name, row_number))
    except OSError as exc:
        raise IdentityError(
            f"Alias file is unreadable: {exc}",
            {"status": "failed", "checks": [_check("identity.alias_readable", False, message=str(exc)).to_dict()]},
        ) from exc

    by_key: dict[tuple[str, str], Alias] = {}
    names_by_id: dict[str, str] = {}
    for alias in aliases:
        key = (alias.source, alias.normalized_match_key)
        prior = by_key.get(key)
        if prior and (prior.canonical_player_id, prior.canonical_player_name) != (alias.canonical_player_id, alias.canonical_player_name):
            raise IdentityError(
                f"Conflicting aliases for {alias.source}/{alias.raw_player_name}",
                {"status": "failed", "checks": [_check("identity.alias_contradictions", False, message="a source/name alias may map to only one canonical identity", source=alias.source, normalized_match_key=alias.normalized_match_key).to_dict()]},
            )
        prior_name = names_by_id.get(alias.canonical_player_id)
        if prior_name is not None and prior_name != alias.canonical_player_name:
            raise IdentityError(
                f"Canonical ID {alias.canonical_player_id} has conflicting alias names",
                {"status": "failed", "checks": [_check("identity.alias_contradictions", False, message="one explicit canonical ID must have one canonical display name", canonical_player_id=alias.canonical_player_id).to_dict()]},
            )
        by_key[key] = alias
        names_by_id[alias.canonical_player_id] = alias.canonical_player_name
    return sorted(by_key.values(), key=lambda item: (item.source, item.normalized_match_key, item.alias_file_row))


def _historical_candidates(rows: Iterable[dict[str, Any]]) -> list[_Candidate]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        gsis_id = _text(row.get("gsis_player_id"))
        name = _text(row.get("player_name"))
        if gsis_id and name:
            grouped.setdefault(gsis_id, []).append(row)
    candidates: list[_Candidate] = []
    for gsis_id, group in sorted(grouped.items()):
        names = sorted({_text(row.get("player_name")) for row in group if _text(row.get("player_name"))})
        keys = frozenset(normalize_match_key(name) for name in names if normalize_match_key(name))
        stats = frozenset(_text(row.get("stat")) for row in group if _text(row.get("stat")))
        positions = sorted({_text(row.get("position")) for row in group if _text(row.get("position"))})
        teams = frozenset(_text(row.get("team")) for row in group if _text(row.get("team")))
        if keys and stats:
            candidates.append(_Candidate(
                canonical_player_id=f"gsis:{gsis_id}", canonical_player_name=names[-1], match_keys=keys,
                stats=stats, position=positions[-1] if positions else None, gsis_player_id=gsis_id,
                teams=teams,
            ))
    return candidates


def _market_identities(rows: Iterable[dict[str, Any]]) -> list[_Identity]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        source, raw_name = _text(row.get("source")), _text(row.get("raw_player_name"))
        if source not in _SOURCE_NAMES or not raw_name:
            continue
        grouped.setdefault((source, raw_name), []).append(row)
    identities: list[_Identity] = []
    for (source, raw_name), group in sorted(grouped.items(), key=lambda item: (item[0][0], normalize_match_key(item[0][1]), item[0][1])):
        key = normalize_match_key(raw_name)
        stats = frozenset(_text(row.get("stat")) for row in group if _text(row.get("stat")))
        teams = frozenset(_text(row.get("team")) or _text(row.get("team_abbreviation")) for row in group if _text(row.get("team")) or _text(row.get("team_abbreviation")))
        if key and stats:
            identities.append(_Identity(source, raw_name, key, stats, teams))
    return identities


def _initials_match(query: str, candidate: _Candidate) -> bool:
    query_parts = query.split()
    if len(query_parts) < 2 or not all(len(part) == 1 for part in query_parts[:-1]):
        return False
    for candidate_key in candidate.match_keys:
        candidate_parts = candidate_key.split()
        if len(candidate_parts) >= 2 and candidate_parts[-1] == query_parts[-1] and all(candidate_parts[index].startswith(initial) for index, initial in enumerate(query_parts[:-1])):
            return True
    return False


def _compatible(identity: _Identity, candidate: _Candidate) -> bool:
    return bool(identity.stats & candidate.stats)


def _suggestion(identity: _Identity, candidate: _Candidate | None, score: float | None, runner_up_score: float | None, reason: str) -> dict[str, Any]:
    return {
        "source": identity.source, "raw_player_name": identity.raw_player_name,
        "normalized_match_key": identity.normalized_match_key,
        "observed_stats": "|".join(sorted(identity.stats)),
        "candidate_canonical_player_id": candidate.canonical_player_id if candidate else "",
        "candidate_canonical_player_name": candidate.canonical_player_name if candidate else "",
        "candidate_gsis_player_id": (candidate.gsis_player_id or "") if candidate else "",
        "candidate_position": (candidate.position or "") if candidate else "",
        "candidate_source": (candidate.source or "historical") if candidate else "",
        "match_score": score if score is not None else "",
        "runner_up_score": runner_up_score if runner_up_score is not None else "",
        "reason": reason, "review_status": "needs_review",
    }


def _alias_lookup(aliases: Iterable[Alias]) -> dict[tuple[str, str], Alias]:
    return {(alias.source, alias.normalized_match_key): alias for alias in aliases}


def reconcile_players(
    normalized_rows: Iterable[dict[str, Any]], historical_rows: Iterable[dict[str, Any]],
    names_config: dict[str, Any], aliases: Iterable[Alias],
) -> IdentityReconciliation:
    """Resolve market identities, keeping every unapproved ambiguity distinct."""

    source_rows = [dict(row) for row in normalized_rows]
    alias_by_key = _alias_lookup(aliases)
    candidates = _historical_candidates(historical_rows)
    identities = _market_identities(source_rows)
    assignments: dict[tuple[str, str], dict[str, Any]] = {}
    suggestions: list[dict[str, Any]] = []
    source_stat_assignments: dict[tuple[str, str], tuple[str, str]] = {}

    for identity in identities:
        alias = alias_by_key.get((identity.source, identity.normalized_match_key)) or alias_by_key.get(("*", identity.normalized_match_key))
        candidate_pool = [candidate for candidate in candidates if candidate.source != identity.source and _compatible(identity, candidate)]
        chosen: _Candidate | None = None
        method, score, alias_row = "unmatched", None, ""
        suggestion_written = False
        explicit_alias = alias is not None
        if alias:
            chosen = _Candidate(alias.canonical_player_id, alias.canonical_player_name, frozenset({identity.normalized_match_key}), identity.stats, None, None, frozenset(), None)
            method, score, alias_row = "explicit_alias", 100.0, str(alias.alias_file_row)
        else:
            exact_by_id: dict[str, _Candidate] = {}
            for candidate in candidate_pool:
                if identity.normalized_match_key in candidate.match_keys:
                    exact_by_id[candidate.canonical_player_id] = candidate
            if len(exact_by_id) == 1:
                chosen = next(iter(exact_by_id.values()))
                method, score = "exact_match_key", 100.0
            elif len(exact_by_id) > 1:
                suggestions.append(_suggestion(identity, None, 100.0, 100.0, "ambiguous_exact_match_key"))
                suggestion_written = True
            else:
                scored: dict[str, tuple[float, _Candidate]] = {}
                for candidate in candidate_pool:
                    candidate_score = 100.0 if _initials_match(identity.normalized_match_key, candidate) else max(ratio(identity.normalized_match_key, key) for key in candidate.match_keys)
                    existing = scored.get(candidate.canonical_player_id)
                    if existing is None or candidate_score > existing[0]:
                        scored[candidate.canonical_player_id] = (candidate_score, candidate)
                ranked = sorted(scored.values(), key=lambda item: (item[0], bool(identity.teams & item[1].teams), item[1].canonical_player_id), reverse=True)
                if ranked:
                    best_score, best = ranked[0]
                    runner_up_score = ranked[1][0] if len(ranked) > 1 else None
                    gap = best_score - runner_up_score if runner_up_score is not None else float("inf")
                    if bool(names_config["automatic_fuzzy_match"]) and best_score >= float(names_config["minimum_score"]) and gap >= float(names_config["minimum_runner_up_gap"]):
                        chosen, method, score = best, "fuzzy_auto", best_score
                    else:
                        reason = (
                            "automatic_fuzzy_disabled" if not bool(names_config["automatic_fuzzy_match"])
                            else "score_below_minimum" if best_score < float(names_config["minimum_score"])
                            else "ambiguous_fuzzy_match"
                        )
                        suggestions.append(_suggestion(identity, best, best_score, runner_up_score, reason))
                        suggestion_written = True

        collision = None
        if chosen and not explicit_alias:
            for stat in identity.stats:
                prior = source_stat_assignments.get((identity.source, stat))
                if prior and prior[0] == chosen.canonical_player_id and prior[1] != identity.raw_player_name and not explicit_alias:
                    collision = stat
                    break
        if collision:
            suggestions.append(_suggestion(identity, chosen, score, None, "source_level_collision"))
            chosen, method, score, alias_row = None, "unmatched", None, ""
            suggestion_written = True

        if chosen is None:
            if not suggestion_written:
                suggestions.append(_suggestion(identity, None, None, None, "no_candidate"))
            canonical_id = _slug_plus_hash(identity.normalized_match_key)
            canonical_name = identity.raw_player_name
            position = gsis_id = None
            review_status = "unmatched"
        else:
            canonical_id, canonical_name = chosen.canonical_player_id, chosen.canonical_player_name
            position, gsis_id = chosen.position, chosen.gsis_player_id
            review_status = "alias" if explicit_alias else "automatic"
        record = {
            "source": identity.source, "raw_player_name": identity.raw_player_name,
            "normalized_match_key": identity.normalized_match_key, "canonical_player_id": canonical_id,
            "canonical_player_name": canonical_name, "match_method": method,
            "match_score": "" if score is None else score, "alias_file_row": alias_row,
            "review_status": review_status, "canonical_position": position or "",
            "canonical_gsis_player_id": gsis_id or "",
        }
        assignments[(identity.source, identity.raw_player_name)] = record
        for stat in identity.stats:
            source_stat_assignments.setdefault((identity.source, stat), (canonical_id, identity.raw_player_name))
        if chosen and not chosen.gsis_player_id:
            candidates.append(_Candidate(canonical_id, canonical_name, frozenset({identity.normalized_match_key}), identity.stats, position, gsis_id, identity.teams, identity.source))

    decorated: list[dict[str, Any]] = []
    for row in source_rows:
        identity = assignments.get((_text(row.get("source")) or "", _text(row.get("raw_player_name")) or ""))
        if identity is None:
            continue
        output = dict(row)
        output.update({
            "canonical_player_id": identity["canonical_player_id"], "canonical_player_name": identity["canonical_player_name"],
            "canonical_position": identity["canonical_position"], "canonical_gsis_player_id": identity["canonical_gsis_player_id"],
            "identity_match_method": identity["match_method"], "identity_match_score": identity["match_score"],
            "identity_review_status": identity["review_status"],
        })
        decorated.append(output)

    player_map = [assignments[key] for key in sorted(assignments)]
    collision_checks = []
    for source in _SOURCE_NAMES:
        relevant = [record for record in player_map if record["source"] == source and record["review_status"] != "alias"]
        for stat in sorted({_text(row.get("stat")) for row in source_rows if _text(row.get("source")) == source and _text(row.get("stat"))}):
            names_by_id: dict[str, set[str]] = {}
            for record in relevant:
                identity = next(item for item in identities if item.source == source and item.raw_player_name == record["raw_player_name"])
                if stat in identity.stats:
                    names_by_id.setdefault(record["canonical_player_id"], set()).add(record["raw_player_name"])
            collision_checks.extend({"source": source, "stat": stat, "canonical_player_id": player_id, "raw_player_names": sorted(names)} for player_id, names in names_by_id.items() if len(names) > 1)
    checks = [
        _check("identity.input_rows_reconciled", len(decorated) == len(source_rows), message="every normalized market row must receive a canonical identity", rows=len(decorated), expected=len(source_rows)),
        _check("identity.player_map_keys_unique", len(player_map) == len({(row["source"], row["raw_player_name"]) for row in player_map}), message="player map source/raw-name keys must be unique"),
        _check("identity.canonical_ids_present", all(row["canonical_player_id"] and row["canonical_player_name"] for row in player_map), message="every map row requires canonical ID and name"),
        _check("identity.source_level_collisions", not collision_checks, message="unaliased source names may not collapse to one identity for a shared stat", collisions=collision_checks),
        _check("identity.fuzzy_suggestions", not suggestions, severity="warning", message="unmatched or ambiguous candidates require review", suggestions=len(suggestions)),
    ]
    errors = [check for check in checks if check.severity == "error" and not check.passed]
    validation = {"status": "failed" if errors else "passed", "checks": [check.to_dict() for check in checks], "summary": {"input_rows": len(source_rows), "player_map_rows": len(player_map), "suggestions": len(suggestions), "aliases_applied": sum(row["review_status"] == "alias" for row in player_map), "automatic_matches": sum(row["review_status"] == "automatic" for row in player_map), "unmatched": sum(row["review_status"] == "unmatched" for row in player_map)}}
    if errors:
        raise IdentityError("Identity validation failed", validation)
    return IdentityReconciliation(decorated, player_map, sorted(suggestions, key=lambda row: (row["source"], row["raw_player_name"], row["reason"])), validation)


def csv_bytes(rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def promote_reviewed_suggestions(suggestions_path: str | Path, aliases_path: str | Path) -> int:
    """Append explicitly approved suggestions to an alias file without overriding it."""

    with Path(suggestions_path).open(encoding="utf-8", newline="") as handle:
        suggestions = list(csv.DictReader(handle))
    existing = load_aliases(aliases_path)
    additions: list[dict[str, str]] = []
    for row_number, row in enumerate(suggestions, start=2):
        if (_text(row.get("review_status")) or "").casefold() not in {"approved", "approved_alias"}:
            continue
        source = _text(row.get("source")); raw_name = _text(row.get("raw_player_name"))
        canonical_id = _text(row.get("candidate_canonical_player_id")); canonical_name = _text(row.get("candidate_canonical_player_name"))
        if source not in _SOURCE_NAMES or not raw_name or not canonical_id or not canonical_name:
            raise IdentityError(
                f"Approved suggestion at row {row_number} is incomplete",
                {"status": "failed", "checks": [_check("identity.promote_reviewed_suggestions", False, message="approved rows require source, raw name, candidate canonical ID, and candidate canonical name", row=row_number).to_dict()]},
            )
        candidate = Alias(source, raw_name, normalize_match_key(raw_name), canonical_id, canonical_name, row_number)
        match = next((alias for alias in existing if (alias.source, alias.normalized_match_key) == (candidate.source, candidate.normalized_match_key)), None)
        if match and (match.canonical_player_id, match.canonical_player_name) != (candidate.canonical_player_id, candidate.canonical_player_name):
            raise IdentityError(
                f"Approved suggestion conflicts with existing alias for {source}/{raw_name}",
                {"status": "failed", "checks": [_check("identity.promote_alias_conflict", False, message="promotion never overwrites an existing alias", source=source, raw_player_name=raw_name).to_dict()]},
            )
        if match is None:
            additions.append({"source": source, "raw_player_name": raw_name, "canonical_player_id": canonical_id, "canonical_player_name": canonical_name, "notes": f"Promoted from {Path(suggestions_path).name}:{row_number}"})
            existing.append(candidate)
    if additions:
        aliases_path = Path(aliases_path)
        with aliases_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            current = list(reader)
            fields = list(reader.fieldnames or ())
        current.extend(additions)
        atomic_write_bytes(aliases_path, csv_bytes(current, fields))
    return len(additions)
