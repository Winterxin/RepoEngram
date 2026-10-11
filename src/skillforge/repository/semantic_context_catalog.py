"""Exact shared input blocks; candidate scopes stay unchanged."""

from __future__ import annotations

import json
from collections import Counter

_REFERENCE = "$skillforge_context"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ExactContextCatalog:
    """Factor identical large values, never similar knowledge or candidate IDs.

    Blocks form an acyclic graph of exact values. Candidate contexts specify which
    blocks are in scope; no knowledge selection or Evidence-domain pooling occurs.
    """

    def __init__(self, contexts):
        counts = Counter()

        def collect(value):
            if isinstance(value, dict):
                if _REFERENCE in value:
                    raise ValueError("reserved shared-context reference in original input")
                counts[_canonical(value)] += 1
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                counts[_canonical(value)] += 1
                for child in value:
                    collect(child)
            elif isinstance(value, str):
                counts[_canonical(value)] += 1

        for context in contexts:
            collect(context)
        self.blocks = {}
        keys = {}

        def encode(value, defining=False):
            text = _canonical(value)
            size = len(text.encode("utf-8"))
            # Do not factor tiny IDs/labels. Whole-wire size comparison also pays
            # for catalog keys, references, bound schemas and instructions.
            if not defining and counts[text] > 1 and size >= 128:
                if text not in keys:
                    key = f"b{len(keys)}"
                    keys[text] = key
                    self.blocks[key] = encode(value, defining=True)
                return {_REFERENCE: keys[text]}
            if isinstance(value, dict):
                return {key: encode(child) for key, child in value.items()}
            if isinstance(value, list):
                return [encode(child) for child in value]
            return value

        self.contexts = [encode(context) for context in contexts]
        if [self.expand(context, self.blocks) for context in self.contexts] != contexts:
            raise ValueError("shared-context reconstruction differs from original input")
        original_size = len(_canonical(contexts).encode("utf-8"))
        shared_size = len(_canonical([self.contexts, self.blocks]).encode("utf-8"))
        # Unprofitable factoring is never sent to the model.
        if shared_size >= original_size:
            self.contexts, self.blocks = contexts, {}

    @staticmethod
    def expand(value, blocks):
        def resolve(item, active, depth):
            if depth > 128:
                raise ValueError("shared-context depth limit")
            if isinstance(item, dict):
                if _REFERENCE in item:
                    if set(item) != {_REFERENCE}:
                        raise ValueError("ambiguous shared-context reference")
                    key = item[_REFERENCE]
                    if not isinstance(key, str) or key not in blocks:
                        raise ValueError("unknown shared-context reference")
                    if key in active:
                        raise ValueError("cyclic shared-context reference")
                    return resolve(blocks[key], active | {key}, depth + 1)
                return {key: resolve(child, active, depth + 1) for key, child in item.items()}
            if isinstance(item, list):
                return [resolve(child, active, depth + 1) for child in item]
            return item

        return resolve(value, set(), 0)
