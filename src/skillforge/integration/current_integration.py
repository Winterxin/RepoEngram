"""Bounded current local references, not additional mined rules or execution proof."""

import ast
import hashlib
import json
import re
from pathlib import Path

from skillforge.domain.evidence import EvidenceType
from skillforge.filesystem_safety import no_symlink
from skillforge.repository.workflow_code_evidence import (
    MAX_CODE_CONTEXT_BYTES,
    MAX_HELPER_DEPTH,
    MAX_IMPLEMENTATION_BYTES,
    MAX_PATCH_HELPERS,
    MAX_PATCH_IMPLEMENTATION_BYTES,
    _reference_size,
)

MAX_SOURCE_BYTES = 524288
MAX_FLOW_BYTES = 1024


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def _references(node, names):
    """Exclude nested function bodies; passing a function is not calling it."""
    calls, values = {}, {}
    bound = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args,
                                *node.args.kwonlyargs)}
    bound.update(arg.arg for arg in (node.args.vararg, node.args.kwarg) if arg)
    pending = list(node.body)
    while pending:
        item = pending.pop()
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            if hasattr(item, "name"):
                bound.add(item.name)
            continue
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
            bound.add(item.id)
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            bound.update(alias.asname or alias.name.split(".")[0] for alias in item.names)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
            name = item.func.id
            if name in names:
                calls.setdefault(name, set()).add(item.lineno)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id in names:
            values.setdefault(item.id, set()).add(item.lineno)
        pending.extend(ast.iter_child_nodes(item))
    for name, lines in calls.items():
        values[name] -= lines
    return ([(name, sorted(lines)) for name, lines in sorted(calls.items()) if name not in bound],
            [(name, sorted(lines)) for name, lines in sorted(values.items())
             if lines and name not in bound])


def _field_target(target):
    """Observe one direct mapping write; dynamic keys stay unknown, never guessed."""
    if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
        return None
    key = target.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        if not key.value or len(key.value.encode()) > 64 or any(
            char in key.value for char in "\r\n\x00"
        ):
            return (target.value.id, None)
        return (target.value.id, key.value)
    return (target.value.id, None)


def _parameter_mapping_reads(node):
    """Literal parameter-key reads, including a literal-key loop; no acceptance proof."""
    parameters = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args,
                                     *node.args.kwonlyargs)}
    # Rebound parameters cannot safely identify the incoming mapping.
    pending = list(node.body)
    while pending:
        item = pending.pop()
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
            parameters.discard(item.id)
        pending.extend(ast.iter_child_nodes(item))
    found = []

    def literal(key):
        if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value:
            if len(key.value.encode()) <= 64 and not any(c in key.value for c in '\r\n\x00'):
                return key.value
        return None

    def visit(item, loop_keys):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(item, (ast.For, ast.AsyncFor)):
            keys = None
            if isinstance(item, ast.For) and isinstance(item.target, ast.Name):
                if isinstance(item.iter, (ast.Tuple, ast.List)) and 0 < len(item.iter.elts) <= 16:
                    values = [literal(key) for key in item.iter.elts]
                    if all(values):
                        rebound = any(isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)
                                      and part.id == item.target.id
                                      for child in item.body for part in ast.walk(child))
                        if not rebound:
                            keys = values
            # Other loops are unknown, including their bodies.
            if keys is not None:
                for child in item.body:
                    visit(child, {**loop_keys, item.target.id: keys})
            return
        container, key = None, None
        if (isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                and item.func.attr == 'get' and isinstance(item.func.value, ast.Name)
                and item.args):
            container, key = item.func.value.id, item.args[0]
        elif (isinstance(item, ast.Subscript) and isinstance(item.ctx, ast.Load)
              and isinstance(item.value, ast.Name)):
            container, key = item.value.id, item.slice
        if container in parameters:
            keys = loop_keys.get(key.id) if isinstance(key, ast.Name) else None
            found.append([container, keys or [literal(key)], item.lineno])
        for child in ast.iter_child_nodes(item):
            visit(child, loop_keys)

    for item in node.body:
        visit(item, {})
    return sorted(found, key=lambda row: row[2])


def _output_flow(node, max_bytes=MAX_FLOW_BYTES, *, field_reads=None):
    """Static assignment/return slice; no claims about execution or correctness.

    Keep complete predicates, references and short mapping keys, never include
    literal values, prompt bodies, nested definitions or inferred call targets.
    """
    statements = []

    def visit(body, conditions):
        for item in body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(item, ast.If):
                test = ast.unparse(item.test)
                # An oversized predicate is explicitly unknown, not cut text.
                test = test if len(test.encode()) <= 160 else None
                visit(item.body, [*conditions, [item.lineno, 'if', test]])
                visit(item.orelse, [*conditions, [item.lineno, 'else', test]])
            elif isinstance(item, ast.Try):
                visit(item.body, [*conditions, [item.lineno, 'try']])
                for handler in item.handlers:
                    visit(handler.body, [*conditions, [handler.lineno, 'except']])
                visit(item.orelse, [*conditions, [item.lineno, 'try_else']])
                visit(item.finalbody, [*conditions, [item.lineno, 'finally']])
            elif isinstance(item, (ast.Assign, ast.AnnAssign, ast.Return)):
                statements.append((item, conditions))
            # Loop/with bodies remain unknown rather than flattening their paths.

    visit(node.body, [])
    assignments = [(item, conditions) for item, conditions in statements
                   if isinstance(item, (ast.Assign, ast.AnnAssign)) and item.value is not None]
    targets, writes = {}, {}
    for item, _ in assignments:
        lhs = item.targets if isinstance(item, ast.Assign) else [item.target]
        fields = [field for target in lhs if (field := _field_target(target)) is not None]
        writes[id(item)] = fields
        targets[id(item)] = [target.id for target in lhs if isinstance(target, ast.Name)]
        targets[id(item)].extend(container for container, _ in fields)

    def reads(value):
        # Do not traverse nested bodies or count a bare callee as a data input.
        found, pending = set(), [value]
        while pending:
            part = pending.pop()
            if isinstance(part, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load):
                found.add(part.id)
            if isinstance(part, ast.Call):
                pending.extend([*part.args, *(k.value for k in part.keywords)])
                if isinstance(part.func, ast.Attribute):
                    pending.append(part.func.value)
            else:
                pending.extend(ast.iter_child_nodes(part))
        return found

    known = {name for names in targets.values() for name in names}
    parameters = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args,
                                     *node.args.kwonlyargs)}
    direct_returns = set()
    for item, _ in statements:
        if isinstance(item, ast.Return):
            values = (item.value.elts if isinstance(item.value, (ast.Tuple, ast.List))
                      else [item.value])
            direct_returns.update(value.id for value in values if isinstance(value, ast.Name))
    known.update(parameters & direct_returns)
    returned = set().union(*(reads(item.value) & known for item, _ in statements
                            if isinstance(item, ast.Return) and item.value is not None))
    relevant = returned.copy()
    dependencies = {}
    for item, _ in assignments:
        inputs = reads(item.value) & known
        for name in targets[id(item)]:
            dependencies.setdefault(name, set()).update(inputs)
    pending = list(relevant)
    while pending:
        additions = dependencies.get(pending.pop(), set()) - relevant
        relevant.update(additions)
        pending.extend(additions)
    entries = []
    mapping_observations = {}
    first_assignment = set()
    last_assignment = {name: id(item) for item, _ in assignments if not writes[id(item)]
                       for name in targets[id(item)] if name in returned}
    terminal_returned = (reads(statements[-1][0].value) & known
                         if statements and isinstance(statements[-1][0], ast.Return)
                         and statements[-1][0].value is not None else set())
    for item, conditions in statements:
        if item.value is None:
            continue
        inputs = sorted(reads(item.value) & relevant)
        row = {'line': item.lineno}
        if conditions:
            row['when'] = conditions
        if isinstance(item, ast.Return):
            if not inputs:
                continue
            row['returns'] = inputs
            priority = 0 if item is statements[-1][0] else 2
        else:
            names = sorted(set(targets[id(item)]) & relevant)
            if not names:
                continue
            fields = [field for field in writes[id(item)] if field[0] in relevant]
            if fields:
                row.update(write_fields=fields, inputs=inputs)
                rebound = sorted(set(names) - {container for container, _ in fields})
                if rebound:
                    row['assign'] = rebound
            else:
                row.update(assign=names, inputs=inputs)
            calls, pending = set(), [item.value]
            while pending:
                part = pending.pop()
                if isinstance(part, ast.Lambda):
                    continue
                if isinstance(part, ast.Call):
                    if isinstance(part.func, ast.Name):
                        calls.add(part.func.id)
                    elif isinstance(part.func, ast.Attribute):
                        calls.add('<dynamic attribute>')
                pending.extend(ast.iter_child_nodes(part))
            row['calls'] = sorted(calls)
            # Same-file callees already parsed for this reference. Keep whole
            # key groups beside the actual call, never prompt or example values.
            consumed = [record for call in sorted(calls)
                        for record in (field_reads or {}).get(call, ())]
            if consumed:
                mapping_observations[item.lineno] = consumed
            if field_reads and isinstance(item.value, ast.Call):
                outer = (item.value.func.id if isinstance(item.value.func, ast.Name)
                         else '<dynamic attribute>')
                # Nested prompt helpers remain in local_calls; do not duplicate
                # their fanout inside the limited output-path slice.
                row['calls'] = sorted({outer, *(record[0] for record in consumed)})
            # Keep the last observed output transformation as well as its
            # origin, before prompt/config literals or repeated early returns.
            priority = (1 if fields
                        or any(name in terminal_returned
                               and last_assignment.get(name) == id(item) for name in names)
                        or (calls and set(names) - first_assignment)
                        else 2 if set(names) & terminal_returned else 3)
            if field_reads and (any(len(record[2]) > 1 for record in consumed)
                                or any(name in terminal_returned
                                       and last_assignment.get(name) == id(item)
                                       for name in names)):
                priority = (-1 if any(name in terminal_returned
                                      and last_assignment.get(name) == id(item)
                                      for name in names) else 0)
            first_assignment.update(names)
        entries.append((priority, row))
    merge_inputs = {name for _, row in entries
                    if any(len(record[2]) > 1
                           for record in mapping_observations.get(row['line'], ()))
                    for name in row.get('inputs', ())}
    origins = set()
    prioritized = []
    for priority, row in entries:
        assigned = set(row.get('assign', ()))
        if assigned & merge_inputs - origins:
            priority = 0
        origins.update(assigned)
        prioritized.append((priority, row))
    entries = prioritized
    selected = []
    # Reallocate the same flow slice, retaining generation/merge/return anchors
    # ahead of intermediate paths. Whole literal-key groups take precedence.
    reserve = min(192, max_bytes // 4) if mapping_observations else 0
    for _, row in sorted(entries, key=lambda pair: (pair[0], pair[1]['line'])):
        candidate = sorted([*selected, row], key=lambda value: value['line'])
        candidate_size = _size({'output_flow': candidate, 'omitted_flow_records': len(entries)})
        if candidate_size <= max_bytes - reserve:
            selected = candidate
    observed = [(row, record) for row in selected
                for record in mapping_observations.get(row['line'], ())]
    for row in selected:
        if row['line'] in mapping_observations:
            row['omitted_field_reads'] = len(mapping_observations[row['line']])
            if _size({'output_flow': selected, 'omitted_flow_records': len(entries)}) > max_bytes:
                row.pop('omitted_field_reads')
    for row, record in sorted(observed, key=lambda pair: (-len(pair[1][2]), pair[0]['line'],
                                                        pair[1][3])):
        prior = row.get('read_fields', [])
        row['read_fields'] = [*prior, record]
        row['omitted_field_reads'] = (len(mapping_observations[row['line']])
                                      - len(row['read_fields']))
        if _size({'output_flow': selected, 'omitted_flow_records': len(entries)}) > max_bytes:
            if prior:
                row['read_fields'] = prior
                row['omitted_field_reads'] += 1
            else:
                row.pop('read_fields')
                row['omitted_field_reads'] += 1
                current_size = _size({'output_flow': selected,
                                      'omitted_flow_records': len(entries)})
                if current_size > max_bytes:
                    row.pop('omitted_field_reads')
    # A missing read_fields record grants no accepted-field/path coverage.
    return {'output_flow': selected, 'omitted_flow_records': len(entries) - len(selected)}


def _mapping_flow(flow, max_bytes):
    """Keep complete field locators without duplicating the helper's call list."""
    selected = []
    total = len(flow['output_flow']) + flow['omitted_flow_records']
    for record in flow['output_flow']:
        if not record.get('write_fields'):
            continue
        row = {key: record[key] for key in ('line', 'when', 'write_fields') if key in record}
        candidate = [*selected, row]
        if _size({'output_flow': candidate, 'omitted_flow_records': total}) <= max_bytes:
            selected = candidate
    return {'output_flow': selected, 'omitted_flow_records': total - len(selected)}


def current_integration(root: Path, evidence, base_commit: str):
    """Use residual code-context budget and source-bound files only, once per read.

    Historical hunk function names are search seeds, never current-policy claims.
    Whole references are selected; omitted references/dynamic calls remain unknown.
    """
    implementations = [item for item in evidence if item.kind == EvidenceType.SOURCE
                       and item.description.startswith("Python implementation ")]
    used = sum(_reference_size(item) for item in implementations)
    slots = MAX_PATCH_HELPERS - len(implementations)
    has_patch = any(item.kind == EvidenceType.GIT and "@@" in item.description
                    for item in evidence)
    cap = MAX_PATCH_IMPLEMENTATION_BYTES if has_patch else MAX_IMPLEMENTATION_BYTES
    budget = min(cap, MAX_CODE_CONTEXT_BYTES - used)
    if slots <= 0 or budget <= 0:
        return None
    seeds = {}
    for item in evidence:
        path = item.location.path
        if not path.endswith(".py"):
            continue
        names = set()
        if item in implementations:
            label = item.description.partition(": ")[0]
            names.add(label.rsplit(".", 1)[-1])
        elif item.kind == EvidenceType.GIT:
            names.update(re.findall(r"@@[^\n]*\b(?:async\s+)?def\s+(\w+)\s*\(",
                                    item.description))
        if names:
            seeds.setdefault(path, set()).update(names)
    result = {"schema": "skillforge-current-local-references-v1",
              "base_commit": base_commit,
              "limits": "Static source observations, not execution or data-flow proof. "
                        "read_fields=[callee,parameter,keys,line]; null=unknown. "
                        "No accepted-field/path guarantees. "
                        "Value references are not execution. Dynamic targets, omitted records, "
                        "loops and cross-module behavior are unknown. "
                        "Current code may be defective; not a requirement.",
              "files": []}
    for path, focus in sorted(seeds.items()):
        if len(result["files"]) >= slots:
            break
        source = root / path
        if not source.absolute().is_relative_to(root.absolute()):
            continue
        try:
            no_symlink(source, root)
            if not source.is_file():
                continue
            with source.open("rb") as stream:
                raw = stream.read(MAX_SOURCE_BYTES + 1)
            if len(raw) > MAX_SOURCE_BYTES:
                continue
            tree = ast.parse(raw.decode("utf-8-sig"))
        except (OSError, ValueError, SyntaxError, UnicodeError, RecursionError):
            continue
        functions = {node.name: node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        focus &= functions.keys()
        if not focus:
            continue
        focus_flows = {name: _output_flow(functions[name]) for name in focus}
        field_focus = {name for name, flow in focus_flows.items()
                       if any(row.get('write_fields') for row in flow['output_flow'])}
        refs = {name: _references(node, functions.keys())
                for name, node in functions.items()}
        selected = {name: 0 for name in focus}
        for depth in range(1, MAX_HELPER_DEPTH + 1):
            previous = set(selected)
            for name, (calls, values) in refs.items():
                if name not in selected and {key for key, _ in calls + values} & previous:
                    selected[name] = depth
        # A same-file public entry is a separate source anchor, not a claim that
        # callbacks execute or an extension of the helper-closure depth limit.
        public = [name for name, (calls, values) in refs.items()
                  if not name.startswith("_")
                  and {key for key, _ in calls + values} & selected.keys()]
        public_anchor = min(public, key=lambda name: functions[name].lineno) if public else None
        if public_anchor is not None:
            selected[public_anchor] = 0
        field_reads = {}
        if public_anchor is not None:
            # Limit downward observations to direct same-file public calls.
            # The caller's binding-aware references exclude shadowed callees.
            for callee, _ in refs[public_anchor][0]:
                observed = _parameter_mapping_reads(functions[callee])
                if observed:
                    field_reads[callee] = [[callee, *entry] for entry in observed]
            # Prefer complete input key groups over unrelated capability reads.
            grouped = {name: [entry for entry in observed if len(entry[2]) > 1]
                       for name, observed in field_reads.items()}
            if any(grouped.values()):
                field_reads = {name: observed for name, observed in grouped.items() if observed}
        rows = []
        record = {"path": path, "sha256": hashlib.sha256(raw).hexdigest(),
                  "functions": rows, "omitted_related_functions": len(selected)}
        flow_remaining = MAX_FLOW_BYTES
        # A public caller offers integration context before another private helper.
        for name in sorted(selected, key=lambda key: (key != public_anchor,
                                                     key.startswith("_"),
                                                     key not in field_focus,
                                                     -selected[key], functions[key].lineno)):
            if len(rows) >= MAX_PATCH_HELPERS:
                break
            node = functions[name]
            calls, values = refs[name]
            row = {"name": name, "lines": [node.lineno, node.end_lineno],
                   "local_calls": calls, "local_values": values}
            if (not rows and not name.startswith('_')) or name in field_focus:
                # Reserve part of the existing slice for the source-bound field
                # mutator. Public generation/merge locators still come first.
                allowance = (flow_remaining - 192
                             if not rows and not name.startswith('_') and field_focus
                             else flow_remaining)
                flow = (_mapping_flow(focus_flows[name], max(0, allowance))
                        if rows and name in field_focus
                        else _output_flow(node, max(0, allowance),
                                          field_reads=(field_reads
                                                       if name == public_anchor else None)))
                if flow['output_flow']:
                    row.update(flow)
                elif flow['omitted_flow_records']:
                    row['omitted_flow_records'] = flow['omitted_flow_records']
            rows.append(row)
            record["omitted_related_functions"] -= 1
            if _size({**result, "files": [*result["files"], record]}) > budget:
                # Preserve the caller anchor when unrelated fanout would evict
                # it. Never truncate a name or its source-line tuple.
                row["omitted_local_references"] = 0
                while _size({**result, "files": [*result["files"], record]}) > budget:
                    candidates = [(key, index, ref) for key in ("local_calls", "local_values")
                                  for index, ref in enumerate(row[key])]
                    if not candidates:
                        # Keep the locator even if no complete flow fits.
                        if row.get('output_flow'):
                            row['omitted_flow_records'] += len(row.pop('output_flow'))
                        break
                    key, index, _ = max(candidates, key=lambda item: (
                        item[2][0] not in selected, _size(item[2]), item[2][0]))
                    row[key].pop(index)
                    row["omitted_local_references"] += 1
                if _size({**result, "files": [*result["files"], record]}) > budget:
                    rows.pop()
                    record["omitted_related_functions"] += 1
            if row in rows and 'output_flow' in row:
                flow_remaining -= _size({key: row[key] for key in
                                         ('output_flow', 'omitted_flow_records')})
        if rows:
            result["files"].append(record)
    return result if result["files"] else None
