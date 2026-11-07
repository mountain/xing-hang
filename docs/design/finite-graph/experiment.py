#!/usr/bin/env python3
"""Exact, standard-library-only finite research; no game rules are installed.

python3 docs/design/finite-graph/experiment.py --check
python3 docs/design/finite-graph/experiment.py --output /tmp/xing-hang-results.json
"""
import argparse
from collections import Counter, deque
from copy import deepcopy
import hashlib
from itertools import combinations, product
import json
from math import comb
from pathlib import Path

SOURCE_COMMIT = "b1df89bafd27fcc213a798915a2db22aeb3a3dab"
SOURCE_SHA256 = "cc243861afcb55365d249b71c297fe2bbbce4665a4bf5c79076db10f7d67e186"
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def edge(u, v):
    return (min(u, v), max(u, v))


def cycle_edges(vertices):
    return [edge(vertices[i], vertices[(i + 1) % len(vertices)])
            for i in range(len(vertices))]


def components(adj, omitted=frozenset()):
    unseen = set(adj)
    sizes = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        queue = deque([root])
        size = 0
        while queue:
            u = queue.popleft()
            size += 1
            for v in adj[u]:
                if v in unseen and edge(u, v) not in omitted:
                    unseen.remove(v)
                    queue.append(v)
        sizes.append(size)
    return sorted(sizes)


def validate_board(board):
    n = len(board["cities"])
    require(n == 302, "city count")
    raw = board["neighbors"]
    require(set(raw) == {str(i) for i in range(n)}, "vertex IDs")
    for key, ns in raw.items():
        u = int(key)
        require(all(type(v) is int and 0 <= v < n for v in ns), "neighbor IDs")
        require(len(ns) == len(set(ns)), "duplicate neighbor")
        require(u not in ns, "self-loop")
    adj = {int(u): frozenset(ns) for u, ns in raw.items()}
    require(all(u in adj[v] for u in adj for v in adj[u]), "reciprocity")
    edges = {edge(u, v) for u in adj for v in adj[u]}
    require(len(edges) == 600, "edge count")
    groups = [(0, 60, 5), (60, 210, 4), (210, 290, 3), (290, 302, 5)]
    for lo, hi, degree in groups:
        require(all(len(adj[v]) == degree for v in range(lo, hi)), "degree class")
    require(components(adj) == [302], "connectivity")

    # Independent reconstruction 1: all boundaries of the 300 quadrilaterals.
    quads = board["polysmalls"] + board["polylarges"]
    require(len(board["polysmalls"]) == 240 and len(board["polylarges"]) == 60,
            "quadrilateral counts")
    require(all(len(q) == len(set(q)) == 4 for q in quads), "quadrilateral vertices")
    incidence = Counter(e for q in quads for e in cycle_edges(q))
    require(set(incidence) == edges, "quadrilateral edge set")
    require(set(incidence.values()) == {2}, "quadrilateral incidence")

    # Independent reconstruction 2: original edge/face incidence via encoder.
    enc = [frozenset(row) for row in board["encoder"]]
    require(len(enc) == len(set(enc)) == 302, "encoder uniqueness")
    require(all(enc[v] == {v} for v in range(60)), "original vertex encoder")
    triangles, pentagons = board["triangles"], board["pentagons"]
    require(len(triangles) == 80 and len(pentagons) == 12, "original face counts")
    faces = triangles + pentagons
    require(all(len(f) == len(set(f)) and all(0 <= v < 60 for v in f)
                for f in faces), "original face vertices")
    original_incidence = Counter(e for f in faces for e in cycle_edges(f))
    require(len(original_incidence) == 150 and set(original_incidence.values()) == {2},
            "original edge incidence")
    midpoint = {enc[v]: v for v in range(60, 210)}
    require(set(midpoint) == {frozenset(e) for e in original_incidence}, "edge encoder")
    require(all(enc[210 + i] == set(f) for i, f in enumerate(faces)), "face encoder")
    reconstructed = set()
    for endpoints, mid in midpoint.items():
        reconstructed.update(edge(v, mid) for v in endpoints)
    for i, face in enumerate(faces):
        reconstructed.update(edge(210 + i, midpoint[frozenset(e)]) for e in cycle_edges(face))
    require(reconstructed == edges, "encoder reconstructed edges")

    left = set(range(60)) | set(range(210, 302))
    right = set(range(60, 210))
    require(all((u in left) != (v in left) for u, v in edges), "bipartition")
    return adj, sorted(edges), left, right


def cut(adj, occupied):
    return sum(len(adj[u] - occupied) for u in occupied)


def legal(adj, own, other):
    occupied = own | other
    return {(u, v) for u in own for v in adj[u] - occupied}


def relay(adj, own, other):
    """Candidate only: an occupied friendly middle vertex enables a length-2 move."""
    occupied = own | other
    return {(u, v) for u in own for w in adj[u] & own for v in adj[w] - occupied}


def bridges_without(adj, skipped=None):
    """DFS low-link bridge search in the graph with at most one edge removed."""
    discovery, low, bridges = {}, {}, []

    def visit(u, parent):
        discovery[u] = low[u] = len(discovery)
        for v in sorted(adj[u]):
            if edge(u, v) == skipped or v == parent:
                continue
            if v not in discovery:
                visit(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > discovery[u]:
                    bridges.append(edge(u, v))
            else:
                low[u] = min(low[u], discovery[v])

    visit(min(adj), None)
    require(len(discovery) == len(adj), "bridge search disconnected input")
    return bridges


def edge_connectivity_certificate(adj, edges):
    require(not bridges_without(adj), "one-edge cut found")
    for e in edges:
        require(not bridges_without(adj, e), "two-edge cut found")
    # Independent lower-bound check: three unit flows from 0 to each other
    # vertex. Each undirected edge is represented by opposite unit-capacity
    # arcs; the capacity of any cut is its undirected edge-boundary size.
    for sink in range(1, len(adj)):
        residual = {(u, v): 1 for u in adj for v in adj[u]}
        for _ in range(3):
            parent, queue = {0: None}, deque([0])
            while queue and sink not in parent:
                u = queue.popleft()
                for v in sorted(adj[u]):
                    if residual[u, v] > 0 and v not in parent:
                        parent[v] = u
                        queue.append(v)
            require(sink in parent, "three-flow lower-bound check")
            v = sink
            while v != 0:
                u = parent[v]
                residual[u, v] -= 1
                residual[v, u] += 1
                v = u
    witness = sorted(edge(210, v) for v in adj[210])
    sizes = components(adj, frozenset(witness))
    require(sizes == [1, 301], "three-edge cut witness")
    return {"value": 3, "one_edge_deletions_with_no_remaining_bridges": len(edges),
            "three_edge_cut_witness": witness, "witness_component_sizes": sizes,
            "independent_three_unit_flow_checks_from_vertex_0": len(adj) - 1,
            "algorithm": "DFS low-link after each of 600 single-edge deletions"}


def occupancy_enumeration(adj, max_k):
    result = []
    vertices = sorted(adj)
    for k in range(max_k + 1):
        hist, first = Counter(), {}
        for ids in combinations(vertices, k):
            occupied = frozenset(ids)
            # Compare degree/internal-edge identity with explicit moves.
            internal = sum(v in adj[u] for u, v in combinations(ids, 2))
            identity = sum(len(adj[u]) for u in ids) - 2 * internal
            direct = sum(len(adj[u] - occupied) for u in ids)
            require(identity == direct, "cut identity during enumeration")
            hist[direct] += 1
            first.setdefault(direct, ids)
        require(sum(hist.values()) == comb(len(adj), k), "enumeration coverage")
        result.append({"k": k, "states": sum(hist.values()), "min": min(hist), "max": max(hist),
                       "argmax_count": hist[max(hist)], "max_witness": first[max(hist)],
                       "min_witness": first[min(hist)], "histogram": dict(sorted(hist.items()))})
    return result


def small_graph_checks():
    graphs = {
        "C6": {v: frozenset({(v - 1) % 6, (v + 1) % 6}) for v in range(6)},
        "P4": {0: frozenset({1}), 1: frozenset({0, 2}),
               2: frozenset({1, 3}), 3: frozenset({2})},
        "K1_3": {0: frozenset({1, 2, 3}), **{v: frozenset({0}) for v in range(1, 4)}},
    }
    result = {}
    for name, adj in graphs.items():
        n, total = len(adj), 0
        for state in product(range(3), repeat=n):
            own = frozenset(i for i, c in enumerate(state) if c == 1)
            other = frozenset(i for i, c in enumerate(state) if c == 2)
            occupied = own | other
            ma, mb = legal(adj, own, other), legal(adj, other, own)
            degree_a = sum(len(adj[u]) for u in own)
            eaa = sum(v in adj[u] for u, v in combinations(own, 2))
            eab = sum(len(adj[u] & other) for u in own)
            require(len(ma) == degree_a - 2 * eaa - eab, "one-side formula")
            require(len(ma) + len(mb) == cut(adj, occupied), "sum formula")
            require(cut(adj, occupied) == cut(adj, set(adj) - occupied), "complement")
            successors = set()
            for u, v in ma:
                successors.add((frozenset((own - {u}) | {v}), other))
            for u, v in mb:
                successors.add((own, frozenset((other - {u}) | {v})))
            require(len(successors) == len(ma) + len(mb), "successor injectivity")
            for x in set(adj) - occupied:
                require(cut(adj, occupied | {x}) - cut(adj, occupied)
                        == len(adj[x]) - 2 * len(adj[x] & occupied), "placement marginal")
                # Adding a friendly token can only remove baseline actions of old tokens.
                new = legal(adj, own | {x}, other)
                require({m for m in new if m[0] in own} == {m for m in ma if m[1] != x},
                        "friendly enabling impossibility")
            total += 1
        result[name] = {"colored_states": total,
                        "occupancy_profile": occupancy_enumeration(adj, n)}
    c4 = {v: frozenset({(v - 1) % 4, (v + 1) % 4}) for v in range(4)}
    require(relay(c4, {0, 1, 3}, set()) == {(0, 2)}, "relay witness deduplication")
    c6 = graphs["C6"]
    result["C6_examples"] = [
        {"occupied": s, "baseline": cut(c6, set(s)), "relay_extra": len(relay(c6, set(s), set()))}
        for s in [[0, 1, 2], [0, 1, 3], [0, 2, 4]]]
    result["relay_duplicate_witness_check"] = {"graph": "C4", "own": [0, 1, 3],
                                                "move": [0, 2], "witnesses": [1, 3], "count": 1}
    return result


def negative_controls(board):
    mutations = []
    b = deepcopy(board); b["neighbors"]["0"].append(0)
    mutations.append(("self_loop", b))
    b = deepcopy(board); b["neighbors"]["0"].append(b["neighbors"]["0"][0])
    mutations.append(("duplicate_neighbor", b))
    b = deepcopy(board); b["neighbors"]["0"].remove(70)
    mutations.append(("asymmetric_edge", b))
    b = deepcopy(board); b["neighbors"]["0"].remove(70); b["neighbors"]["70"].remove(0)
    mutations.append(("deleted_edge", b))
    b = deepcopy(board); b["polysmalls"][0][0] = 211
    mutations.append(("wrong_quad", b))
    b = deepcopy(board); b["encoder"][60] = [0, 1]
    mutations.append(("wrong_encoder", b))
    result = []
    for label, bad in mutations:
        try:
            validate_board(bad)
        except ValueError as error:
            result.append({"mutation": label, "rejected_by": str(error)})
        else:
            raise ValueError("negative control accepted: " + label)
    return result


def board_examples(adj):
    rows = []
    for a, b in [([0, 1], []), ([0, 84], []), ([0, 1, 2], []),
                 ([0, 1, 60], []), ([0, 1, 70], []), ([0, 1, 84], []),
                 ([0, 84, 290], []), ([0, 1], [84]), ([84], [0, 1]), ([0, 1, 290], [])]:
        own, other = set(a), set(b)
        ma, mb = legal(adj, own, other), legal(adj, other, own)
        extra = relay(adj, own, other) - ma
        rows.append({"A": a, "B": b, "M_A": len(ma), "M_B": len(mb),
                     "total": len(ma) + len(mb), "A_relay_extra": len(extra),
                     "A_baseline_plus_relay": len(ma | relay(adj, own, other))})
    return rows


def run(source):
    raw = source.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA256, "source SHA-256")
    board = json.loads(raw)
    adj, edges, left, right = validate_board(board)
    canonical = (json.dumps({"vertices": sorted(adj), "edges": edges},
                            separators=(",", ":")) + "\n").encode()
    connectivity = edge_connectivity_certificate(adj, edges)
    high = {v for v in adj if len(adj[v]) == 5}
    require(len(high) == 72 and cut(adj, high) == 360, "high-degree independent set")
    require(cut(adj, left) == cut(adj, right) == 600, "maximum cut witnesses")
    half = left - {210}
    require(len(half) == 151 and cut(adj, half) == 597, "151 witness")
    return {
        "schema": "xing-hang-finite-graph-v1",
        "status": "conditional adjacent-move research; no adopted game rule",
        "source": {"commit": SOURCE_COMMIT, "path": "docs/design/pascal/source-board.json",
                   "bytes": len(raw), "sha256": SOURCE_SHA256,
                   "canonical_graph_sha256": hashlib.sha256(canonical).hexdigest()},
        "graph": {"vertices": 302, "edges": len(edges), "components": components(adj),
                  "degree_histogram": dict(sorted(Counter(map(len, adj.values())).items())),
                  "degree_classes": [
                      {"kind": "original_vertex", "id_first": 0, "id_last": 59, "count": 60, "degree": 5},
                      {"kind": "edge_midpoint", "id_first": 60, "id_last": 209, "count": 150, "degree": 4},
                      {"kind": "triangle_center", "id_first": 210, "id_last": 289, "count": 80, "degree": 3},
                      {"kind": "pentagon_center", "id_first": 290, "id_last": 301, "count": 12, "degree": 5}],
                  "quadrilaterals": 300, "quadrilateral_edge_incidence": 2,
                  "euler_characteristic": 302 - 600 + 300,
                  "neighbor_quad_encoder_edge_sets_equal": True,
                  "bipartition": {"left": sorted(left), "right": sorted(right)},
                  "edge_connectivity": connectivity},
        "board_enumeration": occupancy_enumeration(adj, 3),
        "analytic_certificates": {
            "max_profile_0_through_72": "F(k)=5k; all optimizers are k-subsets of the 72 degree-5 vertices",
            "degree_5_vertices": sorted(high),
            "complement_profile": "F(302-k)=F(k)",
            "global_maximum": 600, "global_maximizing_cardinalities": [150, 152],
            "global_maximizing_sets": [sorted(right), sorted(left)],
            "k151_value": 597, "k151_witness": sorted(half),
            "remaining_unsolved_k": [*range(73, 150), *range(153, 230)]},
        "board_examples": board_examples(adj),
        "small_graphs": small_graph_checks(),
        "negative_controls": negative_controls(board),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE.parent / "pascal" / "source-board.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="compare exact replay bytes with frozen results")
    group.add_argument("--output", type=Path, help="write new result file; refuse to overwrite existing paths")
    args = parser.parse_args()
    result = run(args.source)
    data = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if args.check:
        require(data == (HERE / "results.json").read_bytes(), "frozen result mismatch")
        print("PASS: exact replay matches results.json")
    else:
        with args.output.open("xb") as stream:
            stream.write(data)
        print("Wrote", args.output)
    print("Result SHA-256:", hashlib.sha256(data).hexdigest())
    print("Board states enumerated:", sum(r["states"] for r in result["board_enumeration"]))


if __name__ == "__main__":
    main()
