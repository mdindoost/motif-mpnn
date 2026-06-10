# src/datasets/substructure_counts.py
"""Local exact substructure counters used by the expressivity demos and the motif
pipeline. Pure functions over networkx graphs; igraph used where it is faster.

NOTE: These are special-purpose features for Phase 0 (CSL needs long cycles, which
are OUTSIDE the size-<=5 orbit library; SRG needs 4-cliques, which ARE in it).
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict

import networkx as nx


def triangles_per_node(g: nx.Graph, num_nodes: int) -> Dict[int, int]:
    tri = nx.triangles(g)
    return {v: int(c) for v, c in tri.items() if c > 0}


def four_cliques_per_node(g: nx.Graph, num_nodes: int) -> Dict[int, int]:
    """Number of 4-cliques each vertex participates in (exact)."""
    counts: Counter = Counter()
    for clique in nx.enumerate_all_cliques(g):
        if len(clique) == 4:
            for v in clique:
                counts[v] += 1
        elif len(clique) > 4:
            break  # enumerate_all_cliques yields by increasing size
    return {v: int(c) for v, c in counts.items() if c > 0}


def simple_cycles_per_node(g: nx.Graph, num_nodes: int, l_max: int = 10) -> Dict[int, Dict[int, int]]:
    """Per-node counts of simple cycles by length L for 3 <= L <= l_max.

    Returns {node: {L: count}}. Uses nx.simple_cycles with a length bound
    (networkx >= 3.1). Each undirected simple cycle is counted once.
    """
    out: Dict[int, Dict[int, int]] = defaultdict(dict)
    for cycle in nx.simple_cycles(g, length_bound=l_max):
        L = len(cycle)
        if L < 3:
            continue
        for v in cycle:
            out[v][L] = out[v].get(L, 0) + 1
    return out
