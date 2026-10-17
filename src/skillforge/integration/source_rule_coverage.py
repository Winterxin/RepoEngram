"""Pure entrypoint coverage accounting; structural presence is not semantic utility."""

from collections.abc import Sequence

from skillforge.integration.source_task_focus import explicit_source_bindings


def requested_bindings(sources: Sequence[dict], task: str) -> frozenset[tuple[str, str]]:
    bindings = tuple(dict.fromkeys(
        (source['path'], source['symbol']) for source in sources
        if source['symbol'] not in {'module', 'document'}
        and not source['symbol'].startswith('module.')
    ))
    return explicit_source_bindings(bindings, task)


def rule_bindings(rules: Sequence[dict], sources: Sequence[dict]) -> dict:
    paths = {source['source_id']: source['path'] for source in sources}
    return {index: frozenset((paths[ep['source_id']], ep['symbol'])
                             for ep in rule['entrypoints'])
            for index, rule in enumerate(rules)}


def coverage_report(expected: frozenset, bindings: dict, retained: Sequence[int]) -> dict:
    generated = frozenset().union(*bindings.values())
    delivered = frozenset().union(*(bindings[index] for index in retained))
    return {label: [list(key) for key in sorted(keys)] for label, keys in (
        ('requested', expected), ('missing_from_generation', expected - generated),
        ('dropped_by_budget', (expected & generated) - delivered),
        ('delivered', expected & delivered),
    )}
