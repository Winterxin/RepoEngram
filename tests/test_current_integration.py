"""Current caller facts stay bounded and cannot masquerade as execution proof."""

import ast
import hashlib
import json

from skillforge.domain.evidence import Evidence, EvidenceType, SourceLocation
from skillforge.integration.current_integration import (
    MAX_FLOW_BYTES,
    _mapping_flow,
    _output_flow,
    _parameter_mapping_reads,
    current_integration,
)
from skillforge.repository.workflow_code_evidence import (
    MAX_CODE_CONTEXT_BYTES,
    MAX_PATCH_IMPLEMENTATION_BYTES,
    _reference_size,
)

SOURCE = '''def _short(value):
    return value.strip()
def _format(value):
    return _short(value)
def _section():
    return external_factory(formatter=_format)
def _fallback():
    return _section()
def _apply(value):
    return value
def build(client=None):
    prior = _fallback()
    if client is None:
        return prior
    return _apply(client.generate())
'''


def refs():
    return (Evidence('body', 'repo', EvidenceType.SOURCE, SourceLocation('flow.py', line=1),
                     'Python implementation flow._short: def _short(value):\n'
                     '    return value.strip()', 'python_ast'),
            Evidence('patch', 'repo', EvidenceType.GIT, SourceLocation('flow.py', commit='old'),
                     '@@ -1,2 +1,2 @@ def _format(value):\n+    return _short(value)', 'git'))


def test_public_entry_and_callback_are_current_observations_not_execution(tmp_path):
    (tmp_path / 'flow.py').write_text(SOURCE, encoding='utf-8')
    context = current_integration(tmp_path, refs(), 'base')
    records = {row['name']: row for row in context['files'][0]['functions']}
    assert 'build' in records
    assert {'_fallback', '_apply'} == {name for name, _ in records['build']['local_calls']}
    assert '_format' not in {name for name, _ in records['build']['local_calls']}
    assert ('_format', [6]) in records['_section']['local_values']
    assert records['_section']['local_calls'] == []
    assert 'not execution' in context['limits']
    size = len(json.dumps(context, ensure_ascii=False, separators=(',', ':')).encode())
    assert size <= MAX_PATCH_IMPLEMENTATION_BYTES
    assert size + _reference_size(refs()[0]) <= MAX_CODE_CONTEXT_BYTES


def test_current_hash_changes_without_rewriting_published_rule(tmp_path):
    path = tmp_path / 'flow.py'
    path.write_text(SOURCE, encoding='utf-8')
    before = current_integration(tmp_path, refs(), 'base')
    path.write_text(SOURCE + '\nREVISION = 2\n', encoding='utf-8')
    after = current_integration(tmp_path, refs(), 'base')
    assert before['files'][0]['sha256'] != after['files'][0]['sha256']
    assert after['files'][0]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert after['base_commit'] == 'base'
    assert after['files'][0]['functions'] == before['files'][0]['functions']


def test_dynamic_attributes_nested_bodies_and_shadowed_names_are_not_local_calls(tmp_path):
    path = tmp_path / 'flow.py'
    path.write_text(SOURCE + '''
def shadow(_format):
    return _format('x')
def outer():
    def hidden():
        return _format('x')
    return object()._format('x')
''', encoding='utf-8')
    result = current_integration(tmp_path, refs(), 'base')
    names = {row['name'] for row in result['files'][0]['functions']}
    assert 'shadow' not in names and 'outer' not in names and 'hidden' not in names


def test_oversized_and_invalid_files_do_not_escape_read_budget(tmp_path):
    path = tmp_path / 'flow.py'
    path.write_bytes(b'x' * 524289)
    assert current_integration(tmp_path, refs(), 'base') is None
    path.write_text('def invalid(:', encoding='utf-8')
    assert current_integration(tmp_path, refs(), 'base') is None
    assert current_integration(tmp_path, (), 'base') is None


def test_large_public_fanout_preserves_entry_and_marks_omissions(tmp_path):
    helpers = '\n'.join(f'def unrelated_{i:03}():\n    return {i}' for i in range(100))
    calls = '\n'.join(f'    unrelated_{i:03}()' for i in range(100))
    source = SOURCE + helpers + '\n' + 'def public_report():\n    _format("x")\n' + calls + '\n'
    # Make this the first public caller so it is the independent entry anchor.
    source = source.replace('def build(client=None):', 'def _build(client=None):')
    (tmp_path / 'flow.py').write_text(source, encoding='utf-8')
    context = current_integration(tmp_path, refs(), 'base')
    rows = {row['name']: row for row in context['files'][0]['functions']}
    assert 'public_report' in rows
    entry = rows['public_report']
    assert entry['omitted_local_references'] > 0
    assert any(name == '_format' for name, _ in entry['local_calls'])
    assert len(entry['local_calls']) + entry['omitted_local_references'] == 101
    assert _size_for_test(context) <= MAX_PATCH_IMPLEMENTATION_BYTES
    assert _size_for_test(context) + _reference_size(refs()[0]) <= MAX_CODE_CONTEXT_BYTES


def _size_for_test(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode())


def test_public_output_flow_distinguishes_fresh_merge_and_fallback(tmp_path):
    source = '''def _clean(value):
    return value
def _fallback():
    return _clean("baseline")
def publish(client, english):
    base = _fallback()
    if client is None:
        return base
    try:
        fresh = external_generate(client, prompt="PRIVATE LONG PROMPT")
    except Exception:
        return base
    merged = external_merge(base, fresh)
    if english:
        merged = external_format(merged)
    else:
        merged = _clean(merged)
    return merged
'''
    (tmp_path / 'flow.py').write_text(source, encoding='utf-8')
    evidence = (Evidence('patch', 'repo', EvidenceType.GIT,
                        SourceLocation('flow.py', commit='old'),
                        '@@ -1,2 +1,2 @@ def _clean(value):\n+    return value', 'git'),)
    context = current_integration(tmp_path, evidence, 'base')
    entry = context['files'][0]['functions'][0]
    assert entry['name'] == 'publish'
    flow = entry['output_flow']
    fresh = next(row for row in flow if row.get('assign') == ['fresh'])
    assert 'external_generate' in fresh['calls']
    merge = next(row for row in flow if row.get('inputs') == ['base', 'fresh'])
    assert merge['assign'] == ['merged']
    assert flow[-1]['returns'] == ['merged']
    assert any(row.get('returns') == ['base'] and 'when' in row for row in flow)
    assert 'PRIVATE LONG PROMPT' not in json.dumps(context)
    assert _size_for_test(context) <= MAX_PATCH_IMPLEMENTATION_BYTES


def test_output_flow_does_not_infer_nested_dynamic_or_loop_execution(tmp_path):
    source = '''def _seed(value):
    return value
def publish(client):
    def ignored():
        return secret_function()
    value = _seed(client)
    value = client.generate(value)
    for item in client:
        value = secret_loop(value)
    callback = lambda: secret_lambda()
    return value
'''
    (tmp_path / 'flow.py').write_text(source, encoding='utf-8')
    evidence = (Evidence('patch', 'repo', EvidenceType.GIT,
                        SourceLocation('flow.py', commit='old'),
                        '@@ -1,2 +1,2 @@ def _seed(value):\n+    return value', 'git'),)
    context = current_integration(tmp_path, evidence, 'base')
    text = json.dumps(context)
    assert '<dynamic attribute>' in text
    assert 'secret_function' not in text
    assert 'secret_loop' not in text
    assert 'secret_lambda' not in text
    assert 'not execution or data-flow proof' in context['limits']


def test_long_output_flow_remains_bounded_with_explicit_omissions(tmp_path):
    source = 'def _seed(value):\n    return value\ndef publish(value):\n    result = _seed(value)\n'
    source += ''.join(f'    result = transform_{i:03}(result)\n' for i in range(100))
    source += '    return result\n'
    (tmp_path / 'flow.py').write_text(source, encoding='utf-8')
    evidence = (Evidence('patch', 'repo', EvidenceType.GIT,
                        SourceLocation('flow.py', commit='old'),
                        '@@ -1,2 +1,2 @@ def _seed(value):\n+    return value', 'git'),)
    context = current_integration(tmp_path, evidence, 'base')
    entry = context['files'][0]['functions'][0]
    assert entry['omitted_flow_records'] > 0
    assert entry['output_flow'][-1]['returns'] == ['result']
    assert len(entry['output_flow']) + entry['omitted_flow_records'] == 102
    assert _size_for_test(context) <= MAX_PATCH_IMPLEMENTATION_BYTES


def test_mapping_writes_distinguish_returned_output_from_unrelated_state():
    source = '''def publish(payload, field):
    output = dict(payload)
    cache = {}
    cache["unrelated"] = "PRIVATE VALUE"
    output["summary"] = normalize(payload)
    output[field] = normalize(payload)
    def nested():
        output["nested_secret"] = "PRIVATE VALUE"
    for item in payload:
        output["loop_secret"] = item
    return output
'''
    flow = _output_flow(ast.parse(source).body[0])
    writes = [tuple(field) for row in flow['output_flow']
              for field in row.get('write_fields', [])]
    assert writes == [('output', 'summary'), ('output', None)]
    assert flow['output_flow'][-1]['returns'] == ['output']
    assert 'PRIVATE VALUE' not in json.dumps(flow)
    assert 'unrelated' not in json.dumps(flow)
    assert 'nested_secret' not in json.dumps(flow)
    assert 'loop_secret' not in json.dumps(flow)
    assert _size_for_test(flow) <= MAX_FLOW_BYTES


def test_direct_output_parameter_and_oversized_mapping_key_stay_explicit():
    source = ('def publish(payload):\n'
              '    payload["summary"] = normalize(payload)\n'
              f'    payload[{"x" * 80!r}] = normalize(payload)\n'
              '    return payload\n')
    flow = _output_flow(ast.parse(source).body[0])
    writes = [tuple(field) for row in flow['output_flow']
              for field in row.get('write_fields', [])]
    assert writes == [('payload', 'summary'), ('payload', None)]
    assert flow['output_flow'][-1]['returns'] == ['payload']
    assert 'x' * 80 not in json.dumps(flow)


def test_source_field_mutator_and_public_path_share_existing_flow_budget(tmp_path):
    source = '''def _keep(value):
    return value
def _guard(payload):
    guarded = dict(payload)
    guarded["summary"] = _keep(payload.get("summary"))
    return guarded
def _fallback():
    return _guard({"summary": "baseline"})
def publish(client):
    base = _fallback()
    fresh = client.generate(prompt="PRIVATE PROMPT")
    merged = merge(base, fresh)
    merged = _guard(merged)
    return merged
'''
    (tmp_path / 'flow.py').write_text(source, encoding='utf-8')
    evidence = (Evidence('patch', 'repo', EvidenceType.GIT,
                        SourceLocation('flow.py', commit='old'),
                        '@@ -1,2 +1,2 @@ def _guard(payload):\n+    return payload', 'git'),)
    context = current_integration(tmp_path, evidence, 'base')
    rows = {row['name']: row for row in context['files'][0]['functions']}
    assert context['files'][0]['functions'][0]['name'] == 'publish'
    assert any(field == ('guarded', 'summary')
               for row in rows['_guard']['output_flow']
               for field in row.get('write_fields', []))
    assert rows['publish']['output_flow'][-1]['returns'] == ['merged']
    assert any(row.get('assign') == ['fresh'] for row in rows['publish']['output_flow'])
    flow_bytes = sum(_size_for_test({key: row[key] for key in
                                    ('output_flow', 'omitted_flow_records')})
                     for row in rows.values() if 'output_flow' in row)
    assert flow_bytes <= MAX_FLOW_BYTES
    assert _size_for_test(context) <= MAX_PATCH_IMPLEMENTATION_BYTES
    assert len(rows) <= 3
    assert 'PRIVATE PROMPT' not in json.dumps(context)


def test_mapping_write_fanout_keeps_return_and_reports_omissions():
    source = 'def publish(payload):\n    output = dict(payload)\n'
    source += ''.join(f'    output["field_{i:03}"] = convert(payload)\n' for i in range(100))
    source += '    return output\n'
    flow = _output_flow(ast.parse(source).body[0])
    assert flow['output_flow'][-1]['returns'] == ['output']
    assert flow['omitted_flow_records'] > 0
    assert len(flow['output_flow']) + flow['omitted_flow_records'] == 102
    assert _size_for_test(flow) <= MAX_FLOW_BYTES


def test_late_conditional_output_write_replaces_ordinary_fanout_under_same_cap():
    source = 'def publish(payload, permitted):\n    output = dict(payload)\n'
    source += ''.join(f'    output["field_{i:03}"] = convert(payload)\n' for i in range(100))
    source += ('    if permitted:\n'
               '        output["private_field"] = convert(payload)\n'
               '    return output\n')
    flow = _output_flow(ast.parse(source).body[0])
    rows = flow['output_flow']
    assert any(row.get('assign') == ['output'] for row in rows)
    assert rows[-1]['returns'] == ['output']
    boundary = next(row for row in rows if row.get('when') and row.get('write_fields'))
    assert boundary['when'] == [[103, 'if', 'permitted']]
    assert boundary['write_fields'] == [('output', 'private_field')]
    assert len(rows) + flow['omitted_flow_records'] == 103
    assert _size_for_test(flow) <= MAX_FLOW_BYTES

    helper = _mapping_flow(flow, 200)
    assert boundary['when'] in [row.get('when') for row in helper['output_flow']]
    assert _size_for_test(helper) <= 200
    assert (len(helper['output_flow']) + helper['omitted_flow_records']) == 103


def test_small_output_flow_keeps_all_records_and_source_order():
    source = '''def publish(payload, permitted):
    output = dict(payload)
    output["ordinary"] = convert(payload)
    if permitted:
        output["guarded"] = convert(payload)
    return output
'''
    flow = _output_flow(ast.parse(source).body[0])
    assert [row['line'] for row in flow['output_flow']] == [2, 3, 5, 6]
    assert flow['omitted_flow_records'] == 0


def test_input_reads_observe_literal_merge_keys_not_returned_or_prompt_values():
    node = ast.parse('''def merge(base, fresh):
    merged = dict(base)
    for key in ("theme", "summary", "findings"):
        value = fresh.get(key)
        merged[key] = value
    ignored = fresh.get(dynamic_key)
    def nested():
        return fresh.get("secret")
    return merged
''').body[0]
    assert _parameter_mapping_reads(node) == [
        ['fresh', ['theme', 'summary', 'findings'], 4], ['fresh', [None], 6]]


def test_input_reads_do_not_guess_rebound_parameters_or_loop_keys():
    source = '''def merge(fresh, other):
    fresh = other
    ignored = fresh.get("wrong_mapping")
    for key in ("one", "two"):
        key = other.name
        ignored = other.get(key)
    for key in dynamic_keys:
        ignored = other.get("unknown_loop")
    return other["summary"]
'''
    assert _parameter_mapping_reads(ast.parse(source).body[0]) == [['other', ['summary'], 9]]


def test_current_merge_field_reads_share_budget_and_keep_public_generation_anchor(tmp_path):
    source = '''def _guard(payload):
    result = dict(payload)
    result["summary"] = clean(payload)
    return result
def _merge(base, fresh):
    result = dict(base)
    for key in ("theme", "summary", "findings"):
        result[key] = fresh.get(key)
    return result
def publish(client):
    base = {}
    fresh = client.generate("PRIVATE PROMPT")
    merged = _merge(base, fresh)
    merged = _guard(merged)
    return merged
'''
    (tmp_path / 'flow.py').write_text(source, encoding='utf-8')
    evidence = (Evidence('patch', 'repo', EvidenceType.GIT,
                        SourceLocation('flow.py', commit='old'),
                        '@@ -1,2 +1,2 @@ def _guard(payload):\n+    return payload', 'git'),)
    context = current_integration(tmp_path, evidence, 'base')
    rows = context['files'][0]['functions']
    flow = rows[0]['output_flow']
    assert rows[0]['name'] == 'publish'
    assert any(row.get('assign') == ['fresh'] for row in flow)
    assert flow[-1]['returns'] == ['merged']
    assert any(record == ['_merge', 'fresh', ['theme', 'summary', 'findings'], 8]
               for row in flow for record in row.get('read_fields', []))
    assert sum(_size_for_test({key: row[key] for key in
                              ('output_flow', 'omitted_flow_records')})
               for row in rows if 'output_flow' in row) <= MAX_FLOW_BYTES
    assert _size_for_test(context) <= MAX_PATCH_IMPLEMENTATION_BYTES
    assert len(rows) <= 3
    assert 'PRIVATE PROMPT' not in json.dumps(context)


def test_oversized_read_group_keeps_call_locator_under_original_flow_cap():
    node = ast.parse('''def publish(client):
    fresh = client.generate()
    merged = merge(fresh)
    return merged
''').body[0]
    reads = {'merge': [['merge', 'fresh', ['key_' + str(i) for i in range(100)], 99]]}
    flow = _output_flow(node, field_reads=reads)
    assert any('merge' in row.get('calls', []) for row in flow['output_flow'])
    assert any(row.get('omitted_field_reads') == 1 for row in flow['output_flow'])
    assert flow['output_flow'][-1]['returns'] == ['merged']
    assert _size_for_test(flow) <= MAX_FLOW_BYTES
