"""Task-to-source ranking uses bounded lexical evidence, never task execution."""

import ast

from skillforge.integration.source_task_relevance import source_task_relevance


def _function(path: str, source: str):
    node = ast.parse(source).body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return path, node.name, node, source


def test_booking_state_constraints_rank_above_unrelated_numeric_helper():
    functions = [
        _function(
            "src/reservations.py",
            "def apply_reservation_status(reservation_status):\n"
            "    if reservation_status == 'confirmed':\n"
            "        return reservation_status\n"
            "    if reservation_status == 'cancelled':\n"
            "        raise ValueError()\n"
            "    return 'pending'\n",
        ),
        _function(
            "src/math_helpers.py",
            "def sum_values(values):\n"
            "    return sum(values)\n",
        ),
    ]
    scores = source_task_relevance(
        functions,
        "Preserve booking status when a reservation is cancelled or confirmed",
    )
    assert scores[("src/reservations.py", "apply_reservation_status")] > 0
    assert scores[("src/reservations.py", "apply_reservation_status")] > scores[
        ("src/math_helpers.py", "sum_values")
    ]


def test_async_file_write_constraints_rank_above_unrelated_business_helper():
    functions = [
        _function(
            "src/storage.py",
            "async def write_atomically(path, content):\n"
            "    if not path:\n"
            "        raise ValueError()\n"
            "    await replace_file(path, content)\n",
        ),
        _function(
            "src/invoice.py",
            "def calculate_invoice_tax(amount, rate):\n"
            "    return amount * rate\n",
        ),
    ]
    scores = source_task_relevance(
        functions,
        "For async file write operations, preserve atomic replacement and avoid partial writes",
    )
    assert scores[("src/storage.py", "write_atomically")] > 0
    assert scores[("src/storage.py", "write_atomically")] > scores[
        ("src/invoice.py", "calculate_invoice_tax")
    ]


def test_repeated_templates_do_not_outweigh_distinct_short_condition_keys():
    functions = [
        _function(
            f"src/template_{index}.py",
            f"def render_{index}(state):\n"
            "    if state == 'ready':\n"
            "        return 'ready'\n"
            "    return '<main>" + ("repeated template text " * 20) + "</main>'\n",
        )
        for index in range(5)
    ]
    functions.append(
        _function(
            "src/reservation.py",
            "def update_reservation_state(reservation_status):\n"
            "    if reservation_status == 'confirmed':\n"
            "        return reservation_status\n"
            "    return reservation_status\n",
        )
    )
    scores = source_task_relevance(
        functions,
        "Update a confirmed reservation status; do not render the ready template",
    )
    target = scores[("src/reservation.py", "update_reservation_state")]
    assert target > max(
        scores[(f"src/template_{index}.py", f"render_{index}")]
        for index in range(5)
    )


def test_camel_case_and_renamed_symbols_keep_lexical_evidence():
    camel = _function(
        "src/booking.py",
        "def applyReservationStatus(reservation_status):\n"
        "    if reservation_status == 'pending':\n"
        "        return reservation_status\n"
        "    return reservation_status\n",
    )
    renamed = _function(
        "src/booking.py",
        "def storeBookingState(reservation_status):\n"
        "    if reservation_status == 'pending':\n"
        "        return reservation_status\n"
        "    return reservation_status\n",
    )
    query = "Preserve reservation status"
    assert source_task_relevance([camel], query)[("src/booking.py", "applyReservationStatus")] > 0
    assert source_task_relevance([renamed], query)[("src/booking.py", "storeBookingState")] > 0


def test_cross_language_query_has_zero_scores_without_forcing_a_rejection():
    function = _function(
        "src/booking.py",
        "def apply_reservation_status(value):\n    return value\n",
    )
    assert source_task_relevance([function], "保留预订状态") == {
        ("src/booking.py", "apply_reservation_status"): 0,
    }


def test_empty_query_and_empty_function_list_are_safe():
    function = _function("src/math.py", "def add_values(a, b):\n    return a + b\n")
    assert source_task_relevance([function], "") == {("src/math.py", "add_values"): 0}
    assert source_task_relevance([], "calculate a result") == {}
