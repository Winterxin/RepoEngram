"""Failed CLI turns must never authorize the next paid stage."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evals'))
from actor_outcome import InvalidActorRun, actor_outcome, require_valid_actor


def events(tmp_path, rows):
    path = tmp_path / 'events.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    return path


@pytest.mark.parametrize('exit_code,rows', [
    (0, [{'type': 'turn.started'}, {'type': 'turn.failed'}]),
    (0, [{'type': 'turn.started'}]),
    (1, [{'type': 'turn.started'}, {'type': 'turn.completed'}]),
    (0, [{'type': 'turn.completed'}]),
    (0, [{'type': 'turn.completed'}, {'type': 'turn.started'}]),
    (0, [{'type': 'turn.started'}, {'type': 'turn.completed'}, {'type': 'turn.started'}]),
])
def test_failed_or_incomplete_turn_rejected_even_with_zero_exit(tmp_path, exit_code, rows):
    with pytest.raises(InvalidActorRun, match='INVALID_INFRASTRUCTURE'):
        require_valid_actor({'exit_code': exit_code, 'timeout': False}, events(tmp_path, rows))


def test_quota_failure_has_explicit_infrastructure_reason(tmp_path):
    path = events(tmp_path, [{'type': 'turn.started'}, {'type': 'turn.failed',
                             'error': {'message': "You've hit your usage limit."}}])
    assert actor_outcome({'exit_code': 1, 'timeout': False}, path)['reason'] == 'ACCOUNT_USAGE_LIMIT'


def test_failed_check_can_be_fixed_inside_an_otherwise_valid_turn(tmp_path):
    path = events(tmp_path, [{'type': 'turn.started'}, {'type': 'item.completed',
        'item': {'type': 'command_execution', 'exit_code': 1}}, {'type': 'turn.completed'}])
    assert require_valid_actor({'exit_code': 0, 'timeout': False}, path)['valid']


def test_timeout_and_corrupt_log_fail_closed(tmp_path):
    path = events(tmp_path, [{'type': 'turn.started'}, {'type': 'turn.completed'}])
    assert actor_outcome({'exit_code': 0, 'timeout': True}, path)['status'] == 'INVALID_BUDGET'
    path.write_text('not JSON', encoding='utf-8')
    with pytest.raises(InvalidActorRun, match='MALFORMED_OR_MISSING_EVENTS'):
        require_valid_actor({'exit_code': 0, 'timeout': False}, path)
