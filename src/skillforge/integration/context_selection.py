"""Ordering policy for already-extracted, bounded current-source observations."""


def output_record_rank(entry):
    """Keep existing strong priorities; prefer anchors and conditional writes.

    This orders source evidence, not project rules or evaluated branch outcomes.
    It adds no records, scans, inferred conditions, or model calls.
    """
    priority, row = entry
    if priority != 1:
        return priority, 0, row["line"]
    if row.get("assign") and not row.get("write_fields"):
        relevance = 0  # Existing generation/final-transformation anchors.
    elif row.get("write_fields") and row.get("when"):
        relevance = 1  # Complete observed branch/exception context.
    else:
        relevance = 2
    return priority, relevance, row["line"]
