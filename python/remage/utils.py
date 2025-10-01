from __future__ import annotations

from collections import defaultdict


def _to_list(thing):
    if not isinstance(thing, tuple | list):
        return [thing]
    return thing


def _icp_to_dict(thing):
    d = defaultdict(list)
    for k, v in thing:
        d[k].append(v)

    return dict(d)
