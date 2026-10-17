"""Bounded ordering of source observations, independent of extraction and models."""

import json
from dataclasses import dataclass

MAX_OBSERVATIONS = 48
MAX_PACKAGE_BYTES = 32 * 1024
COLD_PACKAGE_BYTES = 8 * 1024


@dataclass(frozen=True, slots=True)
class SourceObservation:
    text: str
    priority: int
    legacy_priority: int | None


def ranked_observations(groups):
    """Prioritize within each file, then alternate files without exhausting one."""
    queues = [(path, sorted(rows, key=lambda row: row.priority)) for path, rows in groups]
    for index in range(max((len(rows) for _, rows in queues), default=0)):
        for path, rows in queues:
            if index < len(rows):
                yield path, rows[index].text


def package_sizes(plan):
    """Include metadata and evidence, in both raw and escaped tool representations."""
    payload = json.dumps({item.relative_path: item.content for item in plan.files},
                         ensure_ascii=False)
    return (len(payload.encode("utf-8")),
            len(json.dumps(payload, ensure_ascii=False).encode("utf-8")))
