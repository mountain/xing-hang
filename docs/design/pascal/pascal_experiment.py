#!/usr/bin/env python3
"""Read-only board research; outputs go exclusively to --output."""

import argparse
import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path

import numpy as np


EPS_EQUATOR = 1e-8
EPS_LATITUDE = 1e-8
EPS_CONIC = 1e-8
EPS_NONDEGENERATE = 1e-6
ORDER = [62, 83, 106, 67, 109, 88]
COUNTEREXAMPLE = [0, 1, 2, 3, 4, 5]
Z_AXIS = np.ones(3) / np.sqrt(3)
X_AXIS = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
BASIS = np.array([X_AXIS, np.cross(Z_AXIS, X_AXIS), Z_AXIS])


def unit(a):
    a = np.asarray(a, dtype=float)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(norms < 1e-15):
        raise ValueError("Undefined homogeneous vector")
    return a / norms


def monomials(h):
    x, y, z = h.T
    return np.column_stack([x*x, x*y, y*y, x*z, y*z, z*z])


def matrix_q(c):
    a, b, c0, d, e, f = c
    return np.array([[a, b/2, d/2], [b/2, c0, e/2], [d/2, e/2, f]])


def conic_metrics(points, ids):
    h = unit(points[ids])
    m = monomials(h)
    scales = np.linalg.norm(m, axis=1)
    m = m / scales[:, None]
    _, s, vt = np.linalg.svd(m)
    coeff = vt[-1]
    q = matrix_q(coeff)
    qs = np.linalg.svd(q, compute_uv=False)
    exact_coeff = unit(np.array([1., 0., 1., 0., 0., -2.]))
    triple_dets = [abs(np.linalg.det(h[list(t)]))
                   for t in itertools.combinations(range(6), 3)]
    return {
        "ids": ids,
        "singular_values_row_normalized": s.tolist(),
        "smin_over_smax": float(s[-1]/s[0]),
        "snext_over_smax": float(s[-2]/s[0]),
        "coefficient_order": ["a", "b", "c", "d", "e", "f"],
        "fit_coefficients_unit_l2": coeff.tolist(),
        "fit_max_row_residual": float(np.max(abs(m @ coeff))),
        "Q_singular_values": qs.tolist(),
        "Q_smin_over_smax": float(qs[-1]/qs[0]),
        "Q_det_over_frobenius_cubed": float(np.linalg.det(q)/np.linalg.norm(q)**3),
        "candidate_circle_max_row_residual": float(np.max(abs(m @ exact_coeff))),
        "minimum_abs_triple_determinant": float(min(triple_dets)),
    }


def pascal(points, ids):
    h = unit(points[ids])
    # AB.DE, BC.EF, CD.FA, with all operations in projective coordinates.
    pairs = [((0, 1), (3, 4)), ((1, 2), (4, 5)), ((2, 3), (5, 0))]
    intersections, line_cross_norms, chord_norms, lines = [], [], [], []
    for pair in pairs:
        raw_lines = [np.cross(h[i], h[j]) for i, j in pair]
        chord_norms.extend(float(np.linalg.norm(line)) for line in raw_lines)
        line1, line2 = [unit(line) for line in raw_lines]
        cross = np.cross(line1, line2)
        line_cross_norms.append(float(np.linalg.norm(cross)))
        intersections.append(unit(cross))
        lines.extend([line1, line2])
    intersections = np.array(intersections)
    s = np.linalg.svd(intersections, compute_uv=False)
    pascal_line = unit(np.cross(intersections[0], intersections[1]))
    distinct = [float(np.linalg.norm(np.cross(intersections[i], intersections[j])))
                for i, j in [(0, 1), (1, 2), (0, 2)]]
    affine = [None if abs(p[2]) <= EPS_EQUATOR else (p[:2]/p[2]).tolist()
              for p in intersections]
    return {
        "order_ABCDEF": ids,
        "intersection_definitions": ["AB intersect DE", "BC intersect EF", "CD intersect FA"],
        "intersections_unit_homogeneous": intersections.tolist(),
        "intersections_affine": affine,
        "chord_cross_norms": chord_norms,
        "paired_line_cross_norms": line_cross_norms,
        "intersection_pair_cross_norms": distinct,
        "collinearity_singular_values": s.tolist(),
        "smin_over_smax": float(s[-1]/s[0]),
        "pascal_line_unit_l2": pascal_line.tolist(),
        "max_normalized_incidence_residual": float(np.max(abs(intersections @ pascal_line))),
    }


def search_latitudes(sphere):
    heights = sphere @ Z_AXIS
    retained = np.flatnonzero(heights > EPS_EQUATOR)
    groups = []
    for i in sorted(retained, key=lambda i: (heights[i], i)):
        if not groups or heights[i] - heights[groups[-1][0]] > EPS_LATITUDE:
            groups.append([int(i)])
        else:
            groups[-1].append(int(i))
    rings = [sorted(g) for g in groups if len(g) >= 6]
    return {
        "positive_stable_count": int(len(retained)),
        "negative_stable_count": int(np.sum(heights < -EPS_EQUATOR)),
        "excluded_equatorial_ids": np.flatnonzero(abs(heights) <= EPS_EQUATOR).tolist(),
        "smallest_retained_z": float(np.min(heights[retained])),
        "largest_affine_radius": float(np.max(
            np.linalg.norm(sphere[retained] @ BASIS[:2].T, axis=1)/heights[retained])),
        "latitude_cluster_count": len(groups),
        "rings_at_least_six": [
            {"ids": g, "mean_z": float(np.mean(heights[g])),
             "height_spread": float(np.ptp(heights[g]))} for g in rings
        ],
        "latitude_sextuples_tested": sum(math.comb(len(g), 6) for g in rings),
        "all_hemisphere_sextuples_not_enumerated": math.comb(len(retained), 6),
    }


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_evidence(board_path):
    import sympy as sp

    phi = (1+sp.sqrt(5))/2
    half = sp.Rational(1, 2)
    ideals = {62: [0, 0, 1], 83: [1, 0, 0], 106: [0, 1, 0],
              67: [half, (1-phi)/2, phi/2],
              88: [phi/2, half, (1-phi)/2],
              109: [(1-phi)/2, phi/2, half]}
    basis = sp.Matrix([[1/sp.sqrt(2), -1/sp.sqrt(2), 0],
                       [1/sp.sqrt(6), 1/sp.sqrt(6), -2/sp.sqrt(6)],
                       [1/sp.sqrt(3), 1/sp.sqrt(3), 1/sp.sqrt(3)]])
    h = [basis*sp.Matrix(ideals[i]) for i in ORDER]
    circle_residuals = [sp.simplify(p[0]**2+p[1]**2-2*p[2]**2) for p in h]
    m = sp.Matrix([[x*x, x*y, y*y, x*z, y*z, z*z] for x, y, z in h])
    rank = m.rank()
    pairs = [((0, 1), (3, 4)), ((1, 2), (4, 5)), ((2, 3), (5, 0))]
    intersections = [
        h[a].cross(h[b]).cross(h[c].cross(h[d])).applyfunc(sp.simplify)
        for (a, b), (c, d) in pairs]
    affine = [(p/p[2]).applyfunc(sp.simplify) for p in intersections]
    line = affine[0].cross(affine[1]).applyfunc(sp.simplify)
    line = (line/line[1]).applyfunc(sp.simplify)
    incidences = [sp.simplify(line.dot(p)) for p in affine]
    assert rank == 5 and all(r == 0 for r in circle_residuals+incidences)
    assert all(p != sp.zeros(3, 1) and p[2] != 0 for p in intersections)
    rational_board = json.loads(board_path.read_text(), parse_float=sp.Rational)
    rational_checks = {}
    for name, ids in [("selected_sextuple", ORDER), ("counterexample", COUNTEREXAMPLE)]:
        rows = [rational_board["cities"][i] for i in ids]
        rational_m = sp.Matrix([[x*x, x*y, y*y, x*z, y*z, z*z] for x, y, z in rows])
        determinant = rational_m.det(method="domain-ge")
        assert determinant != 0
        rational_checks[name] = {
            "rank": 6,
            "determinant_nonzero": True,
            "exact_determinant_source_frame": str(determinant),
        }
    return {
        "sympy_version": sp.__version__,
        "ideal_conic_matrix_rank": rank,
        "ideal_circle_residuals": [str(r) for r in circle_residuals],
        "ideal_Q": [[1, 0, 0], [0, 1, 0], [0, 0, -2]],
        "ideal_Q_determinant": -2,
        "ideal_pascal_intersections_affine": [[str(v) for v in p] for p in affine],
        "ideal_pascal_line_coefficients_Y_one": [str(v) for v in line],
        "ideal_pascal_incidence_residuals": [str(v) for v in incidences],
        "runtime_json_decimal_coordinates_as_exact_rationals": rational_checks,
        "warning": "Exact ideal identities do not make rounded runtime coordinates exactly conconic.",
    }


def plot(points, result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(Circle((0, 0), np.sqrt(2), fill=False, color="#64748b", lw=1.5))
    h = points[ORDER]
    xy = h[:, :2]/h[:, 2, None]
    pairs = [((0, 1), (3, 4)), ((1, 2), (4, 5)), ((2, 3), (5, 0))]
    colors = ["#2563eb", "#d97706", "#059669"]
    for pair, color in zip(pairs, colors):
        for i, j in pair:
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=color, alpha=.75, lw=1.6)
    ax.scatter(*xy.T, color="#172554", s=45, zorder=5)
    for letter, idx, p in zip("ABCDEF", ORDER, xy):
        offset = p/np.linalg.norm(p)*.15
        ax.text(*(p+offset), f"{letter}  #{idx}", ha="center", va="center", fontsize=11)
    ip = np.array(result["intersections_affine"])
    ax.scatter(*ip.T, c=colors, s=70, zorder=6, edgecolors="white")
    for letter, p, color in zip("PQR", ip, colors):
        ax.annotate(letter, p, xytext=(10, 4), textcoords="offset points",
                    color=color, fontsize=12, weight="bold")
    l = np.array(result["pascal_line_unit_l2"])
    yy = np.linspace(-1.6, 1.6, 2)
    if abs(l[0]) > 1e-8:
        ax.plot(-(l[1]*yy+l[2])/l[0], yy, "--", color="#be185d", lw=1.8)
    else:
        xx = np.linspace(-1.6, 1.6, 2)
        ax.plot(xx, -(l[0]*xx+l[2])/l[1], "--", color="#be185d", lw=1.8)
    ax.set(xlim=(-1.8, 1.8), ylim=(-1.8, 1.8), aspect="equal",
           xlabel="X / Z", ylabel="Y / Z",
           title="Board sextuple and Pascal line\n"
                 "Fixed north axis (1,1,1)/sqrt(3); circle X^2 + Y^2 = 2")
    ax.text(.02, .02, "P = AB intersect DE\nQ = BC intersect EF\nR = CD intersect FA",
            transform=ax.transAxes, fontsize=10,
            bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#cbd5e1"})
    ax.grid(alpha=.12)
    fig.tight_layout()
    fig.savefig(output/"pascal-six-points.png", dpi=180)
    fig.savefig(output/"pascal-six-points.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=Path)
    source.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.source_root:
        root = args.source_root.resolve()
        if out == root or root in out.parents:
            raise ValueError("Outputs must be outside the Games source tree")
        board_path = root/"astrial/src/board.json"
        other_path = root/"polyclash/polyclash/web/data/board.json"
        if board_path.read_bytes() != other_path.read_bytes():
            raise ValueError("Astrial and PolyClash static boards differ")
        geometry_path = root/"polyclash/model3d/board.npz"
        sources = ["astrial/src/board.json", "astrial/src/data.rs",
                   "polyclash/polyclash/web/data/board.json",
                   "polyclash/polyclash/data/data.py",
                   "polyclash/model3d/board.npz",
                   "polyclash/model3d/snub_dodecahedron_new.npz",
                   "polyclash/scripts/gen_mesh.py", "polyclash/scripts/reindex.py",
                   "polyclash/scripts/gen_cities.py", "polyclash/scripts/gen_board.py",
                   "polyclash/scripts/export_data.py"]
        provenance = {f: sha(root/f) for f in sources}
        shutil.copyfile(board_path, out/"source-board.json")
        shutil.copyfile(geometry_path, out/"source-geometry.npz")
        (out/"source-hashes.json").write_text(json.dumps(provenance, indent=2)+"\n")
    else:
        board_path = args.snapshot/"source-board.json"
        geometry_path = args.snapshot/"source-geometry.npz"
        provenance = json.loads((args.snapshot/"source-hashes.json").read_text())
        assert sha(board_path) == provenance["astrial/src/board.json"]
        assert sha(geometry_path) == provenance["polyclash/model3d/board.npz"]
    board = json.loads(board_path.read_text())
    with np.load(geometry_path, allow_pickle=False) as data:
        raw = data["cities"].copy()
        edges = data["edges"].copy()
        vertices = data["vertices"].copy()
        assert all(np.array_equal(board[k], data[k]) for k in
                   ["triangles", "pentagons", "polysmalls", "polylarges"])
        reconstructed = np.concatenate([
            vertices, vertices[edges].mean(axis=1),
            vertices[data["triangles"]].mean(axis=1),
            vertices[data["pentagons"]].mean(axis=1)])
    json_raw = np.array(board["cities"])
    assert raw.shape == (302, 3) and np.max(abs(raw-json_raw)) < 5.1e-10
    assert np.max(abs(raw-reconstructed)) < 1e-14
    assert all(board["encoder"][60+i] == sorted(edge.tolist())
               for i, edge in enumerate(edges))
    assert all(board["encoder"][210+i] == sorted(face)
               for i, face in enumerate(board["triangles"]))
    assert all(board["encoder"][290+i] == sorted(face)
               for i, face in enumerate(board["pentagons"]))
    sphere, json_sphere = unit(raw), unit(json_raw)
    projected = sphere @ BASIS.T
    json_projected = json_sphere @ BASIS.T
    search = search_latitudes(sphere)
    json_search = search_latitudes(json_sphere)
    assert [r["ids"] for r in search["rings_at_least_six"]] == [
        r["ids"] for r in json_search["rings_at_least_six"]]
    phi = (1+np.sqrt(5))/2
    ideals = {62: [0., 0., 1.], 83: [1., 0., 0.], 106: [0., 1., 0.],
              67: [.5, (1-phi)/2, phi/2],
              88: [phi/2, .5, (1-phi)/2],
              109: [(1-phi)/2, phi/2, .5]}
    cases = {}
    for name, points in [("npz_float64", projected), ("runtime_json_rounded", json_projected)]:
        cases[name] = {
            "conic": conic_metrics(points, ORDER),
            "pascal": pascal(points, ORDER),
            "counterexample": conic_metrics(points, COUNTEREXAMPLE),
            "latitude_rings": [
                conic_metrics(points, ids)
                for ring in search["rings_at_least_six"]
                for ids in map(list, itertools.combinations(ring["ids"], 6))
            ],
        }
        case = cases[name]
        assert case["conic"]["smin_over_smax"] < EPS_CONIC
        assert case["conic"]["snext_over_smax"] > EPS_NONDEGENERATE
        assert case["conic"]["Q_smin_over_smax"] > EPS_NONDEGENERATE
        assert case["conic"]["minimum_abs_triple_determinant"] > EPS_NONDEGENERATE
        assert case["pascal"]["smin_over_smax"] < EPS_CONIC
        assert case["pascal"]["max_normalized_incidence_residual"] < EPS_CONIC
        assert min(case["pascal"]["chord_cross_norms"]) > EPS_NONDEGENERATE
        assert min(case["pascal"]["paired_line_cross_norms"]) > EPS_NONDEGENERATE
        assert min(case["pascal"]["intersection_pair_cross_norms"]) > EPS_NONDEGENERATE
        assert case["counterexample"]["smin_over_smax"] > 1e-3
        assert case["counterexample"]["minimum_abs_triple_determinant"] > EPS_NONDEGENERATE
        assert all(c["smin_over_smax"] < EPS_CONIC and
                   c["Q_smin_over_smax"] > EPS_NONDEGENERATE
                   for c in case["latitude_rings"])
    results = {
        "versions": {"numpy": np.__version__},
        "source_sha256": provenance,
        "source_comparison": {
            "static_board_files_byte_identical": True,
            "npz_json_max_coordinate_difference": float(np.max(abs(raw-json_raw))),
            "npz_centroid_reconstruction_max_error": float(np.max(abs(raw-reconstructed))),
            "npz_sphere_ideal_six_max_coordinate_error": float(max(
                np.max(abs(sphere[i]-ideals[i])) for i in ideals)),
            "halfturn_symmetry_max_nearest_vertex_error": {
                str(i): float(np.max(np.min(np.linalg.norm(
                    (vertices @ (2*np.outer(sphere[i], sphere[i])-np.eye(3)).T)[:, None, :]
                    - vertices[None, :, :], axis=2), axis=1)))
                for i in ideals
            },
        },
        "settings": {
            "basis_rows_X_Y_Z_in_source_frame": BASIS.tolist(),
            "hemisphere": "normalized source point dot Z > 0",
            "equator_exclusion_abs_z_le": EPS_EQUATOR,
            "latitude_complete_span_tolerance": EPS_LATITUDE,
            "conic_svd_ratio_tolerance": EPS_CONIC,
            "nondegeneracy_ratio_floor": EPS_NONDEGENERATE,
        },
        "search": search,
        "runtime_json_search": json_search,
        "points": [
            {"label": letter, "id": i, "encoder": board["encoder"][i],
             "sphere_source_xyz_npz": sphere[i].tolist(),
             "sphere_fixed_XYZ_npz": projected[i].tolist(),
             "projection_XY_npz": (projected[i, :2]/projected[i, 2]).tolist(),
             "projection_XY_json": (json_projected[i, :2]/json_projected[i, 2]).tolist()}
            for letter, i in zip("ABCDEF", ORDER)
        ],
        "cases": cases,
        "exact_evidence": exact_evidence(board_path),
    }
    (out/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    plot(projected, cases["npz_float64"]["pascal"], out)
    print(json.dumps({"search": search, "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
