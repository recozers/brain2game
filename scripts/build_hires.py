#!/usr/bin/env python3
"""Build full-resolution fsaverage (FreeSurfer ico7: 163842 vertices per hemisphere) viewer assets plus an
fsaverage5 -> fsaverage interpolation table, so TRIBE v2's 20484-vertex predictions can be painted on the
fine mesh.

Outputs (app/static/assets/hires/; all little-endian binary, no headers).
N = 327684 vertices (left 0..163841, right 163842..327683), F = 655360 faces.

  pial.f32        Float32 N*3  pial surface, native FreeSurfer mm, no hemisphere shift (they touch at the midline)
  white.f32       Float32 N*3  white surface, same
  inflated.f32    Float32 N*3  inflated surface, hemispheres pulled apart in x with the same rule as
                               scripts/build_assets.py (left medial surface at -HEMI_GAP_MM/2, right at +HEMI_GAP_MM/2)
  faces.u32       Uint32  F*3  left faces then right faces (+163842); FreeSurfer winding kept
  curv.f32        Float32 N    curvature (?h.curv), left then right
  sulc.f32        Float32 N    sulcal depth (?h.sulc), left then right
  interp_idx.u32  Uint32  N*4  per fine vertex: its 4 nearest fsaverage5 vertices on the same hemisphere's sphere,
                               as indices into TRIBE's 20484-long vector (right hemisphere + 10242)
  interp_w.f32    Float32 N*4  inverse-distance-squared weights (rows sum to 1); one-hot where the fine vertex
                               coincides with a coarse vertex (the ico5 vertices are the first 10242 of ico7)
  mesh.json       sizes, dtypes, bbox, gap, validation summary

Viewer use:  fine[v] = sum_k interp_w[v, k] * coarse[interp_idx[v, k]]     (coarse = TRIBE's 20484 floats)

Run:  python3 scripts/build_hires.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "static" / "assets" / "hires"

FINE_MESH = "fsaverage"      # nilearn's name for FreeSurfer's fsaverage (ico7); "fsaverage7" is an alias
COARSE_MESH = "fsaverage5"   # TRIBE v2 output space
N_FINE = 163842              # vertices per hemisphere, fine mesh
N_COARSE = 10242             # vertices per hemisphere, coarse mesh
K = 4                        # coarse neighbours per fine vertex
HEMI_GAP_MM = 12.0           # same constant + rule as scripts/build_assets.py: gap between the two medial surfaces
COINCIDENT_MM = 1e-3         # nearest coarse vertex closer than this -> weight 1 on it, 0 on the rest
HEMIS = ("left", "right")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def load_mesh(fs, key: str):
    from nilearn import surface

    m = surface.load_surf_mesh(fs[key])
    return np.asarray(m.coordinates, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def load_data(fs, key: str) -> np.ndarray:
    from nilearn import surface

    return np.asarray(surface.load_surf_data(fs[key]), dtype=np.float32)


def pull_apart(hemi: str, xyz: np.ndarray):
    """scripts/build_assets.py rule: nilearn's inflated hemispheres are each centred at x~0 and overlap
    completely, so put the left medial surface (its max x) at -HEMI_GAP_MM/2 and the right medial surface
    (its min x) at +HEMI_GAP_MM/2.  Returns the shifted copy and the x shift applied."""
    xyz = xyz.copy()
    if hemi == "left":
        shift = -(xyz[:, 0].max() + HEMI_GAP_MM / 2)
    else:
        shift = -(xyz[:, 0].min() - HEMI_GAP_MM / 2)
    xyz[:, 0] += shift
    return xyz, float(shift)


def outward_fraction(xyz: np.ndarray, tri: np.ndarray) -> float:
    """Fraction of faces whose (b-a)x(c-a) normal points away from the mesh centroid.  On the sphere this is
    1.0 for FreeSurfer's counter-clockwise-from-outside winding and 0.0 if the winding were flipped."""
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    n = np.cross(b - a, c - a)
    out = (a + b + c) / 3.0 - xyz.mean(0)
    return float((np.einsum("ij,ij->i", n, out) > 0).mean())


def bbox(xyz: np.ndarray) -> list[float]:
    return np.concatenate([xyz.min(0), xyz.max(0)]).round(3).tolist()


# ----------------------------------------------------------------------------------------------
# surfaces
# ----------------------------------------------------------------------------------------------
def build_surfaces(fs7):
    coords = {"pial": [], "white": [], "inflated": []}
    faces, curv, sulc, shifts, winding = [], [], [], {}, {}
    for h, hemi in enumerate(HEMIS):
        ref_tri = None
        for name, key in (("pial", "pial"), ("white", "white"), ("inflated", "infl")):
            xyz, tri = load_mesh(fs7, f"{key}_{hemi}")
            assert xyz.shape == (N_FINE, 3), (key, hemi, xyz.shape)
            assert tri.ndim == 2 and tri.shape[1] == 3 and tri.min() == 0 and tri.max() == N_FINE - 1, (key, hemi)
            if ref_tri is None:
                ref_tri = tri
            assert np.array_equal(ref_tri, tri), f"{key}_{hemi}: faces differ from pial faces"
            if name == "inflated":
                xyz, shifts[hemi] = pull_apart(hemi, xyz)
            coords[name].append(xyz)
        sphere_xyz, sphere_tri = load_mesh(fs7, f"sphere_{hemi}")
        assert np.array_equal(ref_tri, sphere_tri), f"sphere_{hemi}: faces differ from pial faces"
        winding[hemi] = outward_fraction(sphere_xyz, ref_tri)
        faces.append(ref_tri + h * N_FINE)
        c, s = load_data(fs7, f"curv_{hemi}"), load_data(fs7, f"sulc_{hemi}")
        assert c.shape == (N_FINE,) and s.shape == (N_FINE,), (c.shape, s.shape)
        curv.append(c)
        sulc.append(s)
        print(f"  {hemi}: {N_FINE} vertices, {len(ref_tri)} faces, inflated x shift {shifts[hemi]:+.3f} mm, "
              f"sphere faces outward-wound: {winding[hemi]:.4f}")

    out = {name: np.concatenate(v).astype("<f4") for name, v in coords.items()}
    out["faces"] = np.concatenate(faces).astype("<u4")
    out["curv"] = np.concatenate(curv).astype("<f4")
    out["sulc"] = np.concatenate(sulc).astype("<f4")
    n = 2 * N_FINE
    for name in ("pial", "white", "inflated"):
        assert out[name].shape == (n, 3)
    assert out["faces"].shape == (2 * 327680, 3)
    assert out["faces"].min() == 0 and out["faces"].max() == n - 1
    assert out["curv"].shape == (n,) and out["sulc"].shape == (n,)
    # every vertex is used by at least one face
    used = np.zeros(n, dtype=bool)
    used[out["faces"].ravel()] = True
    assert used.all(), "unreferenced vertices"
    infl = out["inflated"]
    lx_max = infl[:N_FINE, 0].max()
    rx_min = infl[N_FINE:, 0].min()
    print(f"  inflated: left hemisphere x max {lx_max:+.3f}, right x min {rx_min:+.3f}  "
          f"-> medial gap {rx_min - lx_max:.3f} mm (HEMI_GAP_MM={HEMI_GAP_MM})")
    return out, shifts, winding


# ----------------------------------------------------------------------------------------------
# fsaverage5 -> fsaverage interpolation table
# ----------------------------------------------------------------------------------------------
def build_interp(fs7, fs5):
    from scipy.spatial import cKDTree

    n = 2 * N_FINE
    idx = np.empty((n, K), dtype=np.int64)
    w = np.empty((n, K), dtype=np.float64)
    fine_all, coarse_all = [], []
    report = {}
    for h, hemi in enumerate(HEMIS):
        fine, _ = load_mesh(fs7, f"sphere_{hemi}")
        coarse, _ = load_mesh(fs5, f"sphere_{hemi}")
        assert fine.shape == (N_FINE, 3) and coarse.shape == (N_COARSE, 3), (fine.shape, coarse.shape)
        rf, rc = np.linalg.norm(fine, axis=1), np.linalg.norm(coarse, axis=1)
        d, i = cKDTree(coarse).query(fine, k=K)
        assert d.shape == (N_FINE, K) and np.all(np.diff(d, axis=1) >= 0)

        # nesting: fine vertex j < N_COARSE should sit exactly on coarse vertex j
        nested = i[:N_COARSE, 0] == np.arange(N_COARSE)
        frac_nested = float(nested.mean())
        nest_maxd = float(np.linalg.norm(fine[:N_COARSE] - coarse, axis=1).max())
        coincident = d[:, 0] < COINCIDENT_MM
        n_coincident = int(coincident.sum())
        coincident_beyond = int(coincident[N_COARSE:].sum())

        wi = 1.0 / np.maximum(d, COINCIDENT_MM) ** 2   # inverse distance squared
        wi[coincident] = 0.0
        wi[coincident, 0] = 1.0                        # exact hit -> copy that coarse vertex
        wi /= wi.sum(axis=1, keepdims=True)

        sl = slice(h * N_FINE, (h + 1) * N_FINE)
        idx[sl] = i + h * N_COARSE                     # right hemisphere -> 10242..20483 of TRIBE's vector
        w[sl] = wi
        fine_all.append(fine)
        coarse_all.append(coarse)
        report[hemi] = {
            "nested_fraction": frac_nested,
            "nested_max_dist_mm": nest_maxd,
            "coincident_rows": n_coincident,
            "coincident_rows_beyond_first_10242": coincident_beyond,
            "nearest_dist_mm_nonnested": [float(d[N_COARSE:, 0].min()), float(d[N_COARSE:, 0].mean()), float(d[N_COARSE:, 0].max())],
            "kth_dist_mm_max": float(d[:, K - 1].max()),
        }
        print(f"  {hemi}: sphere radius fine [{rf.min():.3f},{rf.max():.3f}] coarse [{rc.min():.3f},{rc.max():.3f}]")
        print(f"  {hemi}: nearest-coarse==self for fine j<{N_COARSE}: {frac_nested * 100:.3f}%  "
              f"(max |fine[j]-coarse[j]| = {nest_maxd:.2e} mm); coincident rows (<{COINCIDENT_MM} mm): {n_coincident}"
              f" ({coincident_beyond} beyond the first {N_COARSE})")
        print(f"  {hemi}: non-nested fine vertices: nearest coarse dist min/mean/max = "
              f"{d[N_COARSE:, 0].min():.3f}/{d[N_COARSE:, 0].mean():.3f}/{d[N_COARSE:, 0].max():.3f} mm, "
              f"4th-nearest max {d[:, K - 1].max():.3f} mm")
        if frac_nested < 0.999 or n_coincident != N_COARSE:
            print("!!" * 40)
            print(f"!! {hemi}: NESTING ASSUMPTION DOES NOT HOLD (fsaverage5 vertex j != fsaverage vertex j).")
            print("!! The IDW table below is still correct (it never relies on nesting), but anything that")
            print("!! slices fsaverage data [:10242] to get fsaverage5 (e.g. the Glasser step in build_assets.py) is wrong.")
            print("!!" * 40)
        else:
            print(f"  {hemi}: nesting OK (fsaverage5 = first {N_COARSE} vertices of fsaverage on the sphere)")

    idx_u32 = idx.astype("<u4")
    w_f32 = w.astype("<f4")
    assert idx_u32.shape == (n, K) and w_f32.shape == (n, K)
    assert idx_u32.max() == 2 * N_COARSE - 1 and idx_u32[:N_FINE].max() < N_COARSE and idx_u32[N_FINE:].min() >= N_COARSE
    return idx_u32, w_f32, np.concatenate(fine_all), np.concatenate(coarse_all), report


def validate_interp(idx: np.ndarray, w: np.ndarray, fine_sphere: np.ndarray, coarse_sphere: np.ndarray) -> dict:
    """Weight sums + round trip of a smooth function (sphere z, then x and y) coarse -> fine."""
    wsum = w.astype(np.float64).sum(1)
    print(f"  weights: sum-1 max |err| = {np.abs(wsum - 1).max():.2e} (float32 table), min w = {w.min():.3g}, "
          f"rows with a weight == 1: {int((w == 1).any(1).sum())}")
    assert np.abs(wsum - 1).max() < 1e-5
    res = {"weight_sum_max_abs_err": float(np.abs(wsum - 1).max())}
    for axis, name in ((2, "z"), (0, "x"), (1, "y")):
        f = coarse_sphere[:, axis]                       # smooth function on the coarse sphere (20484,)
        pred = (w.astype(np.float64) * f[idx]).sum(1)     # interpolate with the table
        err = np.abs(pred - fine_sphere[:, axis])
        nested = np.zeros(len(err), dtype=bool)
        nested[:N_COARSE] = True
        nested[N_FINE:N_FINE + N_COARSE] = True
        line = (f"  round trip {name}: max abs err {err.max():.4f} mm, mean {err.mean():.4f} mm "
                f"(radius 100; nested vertices max {err[nested].max():.2e}, non-nested mean {err[~nested].mean():.4f})")
        if name == "z":
            line += f"  left max {err[:N_FINE].max():.4f} / right max {err[N_FINE:].max():.4f}"
        print(line)
        res[f"roundtrip_{name}_max_abs_err_mm"] = float(err.max())
        res[f"roundtrip_{name}_mean_abs_err_mm"] = float(err.mean())
    return res


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    from nilearn import datasets

    print(f"fetching {FINE_MESH} + {COARSE_MESH} via nilearn ...")
    fs7 = datasets.fetch_surf_fsaverage(mesh=FINE_MESH)
    fs5 = datasets.fetch_surf_fsaverage(mesh=COARSE_MESH)

    print("surfaces:")
    arrays, shifts, winding = build_surfaces(fs7)
    print("interpolation table:")
    idx, w, fine_sphere, coarse_sphere, interp_report = build_interp(fs7, fs5)
    arrays["interp_idx"], arrays["interp_w"] = idx, w

    print("validation:")
    n = 2 * N_FINE
    for name, a in arrays.items():
        print(f"  {name:11s} shape {tuple(a.shape)} dtype {a.dtype.str}")
    assert arrays["faces"].max() < n, "face index out of range"
    print(f"  faces: all indices < N={n}: {bool(arrays['faces'].max() < n)} (max {int(arrays['faces'].max())}); "
          f"left faces max {int(arrays['faces'][:327680].max())}, right faces min {int(arrays['faces'][327680:].min())}")
    vres = validate_interp(idx, w, fine_sphere, coarse_sphere)

    # write
    OUT.mkdir(parents=True, exist_ok=True)
    order = ["pial", "white", "inflated", "faces", "curv", "sulc", "interp_idx", "interp_w"]
    fname = {"pial": "pial.f32", "white": "white.f32", "inflated": "inflated.f32", "faces": "faces.u32",
             "curv": "curv.f32", "sulc": "sulc.f32", "interp_idx": "interp_idx.u32", "interp_w": "interp_w.f32"}
    files, total = {}, 0
    for name in order:
        a = np.ascontiguousarray(arrays[name])
        assert a.dtype.byteorder in ("<", "|") or (a.dtype.byteorder == "=" and np.little_endian)
        path = OUT / fname[name]
        a.tofile(path)
        size = path.stat().st_size
        assert size == a.nbytes, (name, size, a.nbytes)
        total += size
        files[name] = {"file": fname[name], "bytes": int(size),
                       "dtype": "uint32le" if a.dtype.kind == "u" else "float32le", "shape": list(a.shape)}
        print(f"  wrote {path.relative_to(ROOT)}  {size:,} bytes")

    # read-back check straight from disk (dtype/endianness/size) + the z round trip again from the files
    rb_idx = np.fromfile(OUT / "interp_idx.u32", dtype="<u4").reshape(n, K)
    rb_w = np.fromfile(OUT / "interp_w.f32", dtype="<f4").reshape(n, K)
    rb_faces = np.fromfile(OUT / "faces.u32", dtype="<u4").reshape(-1, 3)
    for name in ("pial", "white", "inflated"):
        rb = np.fromfile(OUT / fname[name], dtype="<f4").reshape(n, 3)
        assert np.array_equal(rb, arrays[name]), name
    assert rb_faces.shape == (2 * 327680, 3) and rb_faces.max() < n
    pred = (rb_w.astype(np.float64) * coarse_sphere[:, 2][rb_idx]).sum(1)
    rb_err = np.abs(pred - fine_sphere[:, 2])
    print(f"  read-back from disk: z round trip max {rb_err.max():.4f} mm, mean {rb_err.mean():.4f} mm; "
          f"faces {rb_faces.shape}, idx {rb_idx.shape}, w {rb_w.shape}")

    mesh = {
        "n_vertices": n,
        "n_faces": int(arrays["faces"].shape[0]),
        "hemi_offset": N_FINE,
        "k": K,
        "coarse_n": 2 * N_COARSE,
        "coarse_hemi_offset": N_COARSE,
        "hemi_gap_mm": HEMI_GAP_MM,
        "files": files,
        "bbox": {name: bbox(arrays[name]) for name in ("pial", "inflated", "white")},
        "source": f"nilearn fetch_surf_fsaverage(mesh='{FINE_MESH}')",
        "mesh": FINE_MESH,
        "coarse_mesh": COARSE_MESH,
        "coarse_source": f"nilearn fetch_surf_fsaverage(mesh='{COARSE_MESH}') sphere (TRIBE v2 vertex order: left 0..10241, right 10242..20483)",
        "units": "mm",
        "inflated_x_shift_mm": shifts,
        "curv_range": [float(arrays["curv"].min()), float(arrays["curv"].max())],
        "sulc_range": [float(arrays["sulc"].min()), float(arrays["sulc"].max())],
        "interp": {
            "method": f"inverse-distance-squared over the {K} nearest fsaverage5 vertices on the same hemisphere's "
                      f"sphere (radius 100); one-hot when the nearest is < {COINCIDENT_MM} mm away",
            "usage": "fine[v] = sum_k interp_w[v,k] * coarse[interp_idx[v,k]]",
            "nested": "fsaverage5 vertex j == fsaverage vertex j (j < 10242) per hemisphere on the sphere",
        },
        "validation": {"sphere_faces_outward_fraction": winding, "nesting": interp_report, **vres},
        "total_bytes": int(total),
    }
    (OUT / "mesh.json").write_text(json.dumps(mesh, indent=1))
    total += (OUT / "mesh.json").stat().st_size
    print(f"wrote {(OUT / 'mesh.json').relative_to(ROOT)}; total {total:,} bytes in {OUT.relative_to(ROOT)}/ "
          f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
