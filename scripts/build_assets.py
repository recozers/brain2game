#!/usr/bin/env python3
"""Build the viewer's mesh + atlas assets from nilearn's fsaverage5.

Outputs (app/static/assets/):
  positions.f32        Float32 LE, N*3   inflated surface, left then right, hemispheres pulled apart in x
  faces.u32            Uint32,     F*3   triangle indices into the combined vertex array (right offset by 10242)
  sulc.f32             Float32 LE, N     sulcal depth for base shading (positive = sulcus)
  mesh.json            {"n_vertices", "n_faces", "hemi_offset", "files", "bbox", ...}
  atlas.json           primary atlas (Glasser if available, otherwise Destrieux) — see CONTRACTS.md
  atlas_destrieux.json Destrieux atlas (always written; identical to atlas.json when Glasser is unavailable)

Vertex order matches TRIBE v2 output: index 0..10241 = left hemisphere, 10242..20483 = right.
Verified against tribev2/plotting/base.py: `get_stat_map` splits a vector as
    left = data[: len(data) // 2]; right = data[len(data) // 2 :]
and `get_mesh` concatenates left then right coordinates, offsetting right-hemisphere faces by
`left_faces.max() + 1` (= 10242 for fsaverage5).  Both hemispheres come from the same
nilearn `fetch_surf_fsaverage("fsaverage5")` files tribev2 uses, in the same order.

Run:  python3 scripts/build_assets.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "static" / "assets"
sys.path.insert(0, str(ROOT / "app"))
import regions  # noqa: E402  (app/regions.py — single source of truth for systems)

MESH = "fsaverage5"
N_PER_HEMI = 10242
HEMI_GAP_MM = 12.0  # gap between the two medial surfaces after pulling the hemispheres apart

# HCP-MMP1 (Glasser) on fsaverage, the same figshare files MNE's fetch_hcp_mmp_parcellation uses.
GLASSER_URLS = {
    "lh": "https://ndownloader.figshare.com/files/5528816",
    "rh": "https://ndownloader.figshare.com/files/5528819",
}
GLASSER_TIME_BUDGET_S = 15 * 60


# ----------------------------------------------------------------------------------------------
# Mesh
# ----------------------------------------------------------------------------------------------
def build_mesh() -> dict:
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage(mesh=MESH)
    coords, faces, sulc = [], [], []
    for hemi in ("left", "right"):
        xyz, tri = surface.load_surf_mesh(fs[f"infl_{hemi}"])
        xyz = np.asarray(xyz, dtype=np.float32).copy()
        tri = np.asarray(tri, dtype=np.int64)
        depth = np.asarray(surface.load_surf_data(fs[f"sulc_{hemi}"]), dtype=np.float32)
        assert xyz.shape == (N_PER_HEMI, 3), xyz.shape
        assert depth.shape == (N_PER_HEMI,), depth.shape
        assert tri.max() == N_PER_HEMI - 1
        # nilearn's inflated hemispheres are each centred at x~0 and overlap completely, so
        # (like tribev2.plotting.base.get_mesh) put the left medial surface at -gap/2 and the
        # right one at +gap/2.
        if hemi == "left":
            xyz[:, 0] -= xyz[:, 0].max() + HEMI_GAP_MM / 2
        else:
            xyz[:, 0] -= xyz[:, 0].min() - HEMI_GAP_MM / 2
            tri = tri + N_PER_HEMI
        coords.append(xyz)
        faces.append(tri)
        sulc.append(depth)

    positions = np.concatenate(coords).astype("<f4")
    faces_all = np.concatenate(faces).astype("<u4")
    sulc_all = np.concatenate(sulc).astype("<f4")
    assert positions.shape == (2 * N_PER_HEMI, 3)
    assert faces_all.max() == 2 * N_PER_HEMI - 1

    OUT.mkdir(parents=True, exist_ok=True)
    positions.tofile(OUT / "positions.f32")
    faces_all.tofile(OUT / "faces.u32")
    sulc_all.tofile(OUT / "sulc.f32")
    bbox = np.concatenate([positions.min(0), positions.max(0)]).round(3).tolist()
    mesh = {
        "n_vertices": int(positions.shape[0]),
        "n_faces": int(faces_all.shape[0]),
        "hemi_offset": N_PER_HEMI,
        "mesh": MESH,
        "surface": "inflated",
        "units": "mm",
        "hemi_gap_mm": HEMI_GAP_MM,
        "files": {"positions": "positions.f32", "faces": "faces.u32", "sulc": "sulc.f32"},
        "dtypes": {"positions": "float32le[N*3]", "faces": "uint32le[F*3]", "sulc": "float32le[N]"},
        "bbox": bbox,
        "sulc_range": [float(sulc_all.min()), float(sulc_all.max())],
    }
    (OUT / "mesh.json").write_text(json.dumps(mesh, indent=1))
    print(f"mesh: {mesh['n_vertices']} vertices, {mesh['n_faces']} faces, bbox {bbox}")
    return mesh


# ----------------------------------------------------------------------------------------------
# Atlases
# ----------------------------------------------------------------------------------------------
def combine_hemis(hemi_names: list[str], map_left: np.ndarray, map_right: np.ndarray, unknown: set[str]):
    """Build hemi-prefixed names + a 20484 label vector.  names[0] == "Unknown" (medial wall etc.)."""
    names = ["Unknown"]
    index = {}
    for prefix, m in (("L", map_left), ("R", map_right)):
        for i, n in enumerate(hemi_names):
            if n in unknown or i == 0 and n in ("Unknown", "???"):
                continue
            full = f"{prefix}_{n}"
            if full not in index:
                index[full] = len(names)
                names.append(full)
    labels = np.zeros(2 * N_PER_HEMI, dtype=np.int32)
    for prefix, m, off in (("L", map_left, 0), ("R", map_right, N_PER_HEMI)):
        m = np.asarray(m).astype(np.int64)
        assert m.shape == (N_PER_HEMI,), m.shape
        for v in np.unique(m):
            n = hemi_names[v] if 0 <= v < len(hemi_names) else None
            if n is None or n in unknown or v == 0:
                continue  # stays 0 == Unknown
            labels[off:off + N_PER_HEMI][m == v] = index[f"{prefix}_{n}"]
    return names, labels


def expand_systems(atlas_key: str, names: list[str], labels: np.ndarray, strict: bool):
    """regions.py lists -> {"id": {"label","blurb","game_target","regions","vertex_count"}}."""
    import difflib

    bare = sorted({n[2:] for n in names[1:]})
    name_to_idx = {n: i for i, n in enumerate(names)}
    systems = {}
    problems = []
    for s in regions.SYSTEMS:
        reg, idxs = [], []
        for bare_name in s[atlas_key]:
            for prefix in ("L", "R"):
                full = f"{prefix}_{bare_name}"
                if full in name_to_idx:
                    reg.append(full)
                    idxs.append(name_to_idx[full])
                else:
                    close = difflib.get_close_matches(bare_name, bare, n=3, cutoff=0.4)
                    problems.append((s["id"], bare_name, close))
        mask = np.isin(labels, idxs) if idxs else np.zeros_like(labels, dtype=bool)
        systems[s["id"]] = {
            "label": s["label"],
            "blurb": s["blurb"],
            "game_target": bool(s["game_target"]),
            "regions": reg,
            "vertex_count": int(mask.sum()),
            "vertex_count_left": int(mask[:N_PER_HEMI].sum()),
            "vertex_count_right": int(mask[N_PER_HEMI:].sum()),
        }
    if problems:
        # de-duplicate (each name is reported once per hemisphere)
        seen = []
        for p in problems:
            if p not in seen:
                seen.append(p)
        print(f"!! {atlas_key}: {len(seen)} region name(s) in app/regions.py are not in the atlas:")
        for sid, n, close in seen:
            print(f"   system={sid!r} name={n!r} closest={close}")
        if strict:
            print("   Fix the spelling in app/regions.py (the 'destrieux' lists) and re-run.")
            sys.exit(1)
    return systems


def sanity_table(atlas_name: str, names: list[str], labels: np.ndarray, systems: dict):
    print(f"\n{atlas_name}: {len(names)} names (incl. Unknown), 20484 labels")
    print(f"{'system':<16}{'label':<46}{'regions':>8}{'left':>7}{'right':>7}{'total':>7}")
    covered = np.zeros(labels.shape[0], dtype=bool)
    idx_of = {n: i for i, n in enumerate(names)}
    for sid, s in systems.items():
        print(f"{sid:<16}{s['label']:<46}{len(s['regions']):>8}{s['vertex_count_left']:>7}{s['vertex_count_right']:>7}{s['vertex_count']:>7}")
        covered |= np.isin(labels, [idx_of[r] for r in s["regions"]])
    unknown = int((labels == 0).sum())
    print(f"{'(not in any system)':<70}{int((~covered).sum()):>7}   of which Unknown/medial wall: {unknown}")
    # overlap check: a vertex belongs to exactly one atlas region, but a region can be listed in 2 systems
    multi = {}
    for sid, s in systems.items():
        for r in s["regions"]:
            multi.setdefault(r, []).append(sid)
    dup = {r: v for r, v in multi.items() if len(v) > 1}
    if dup:
        print(f"note: regions shared by several systems: {dup}")


def write_atlas(path: Path, atlas_name: str, names: list[str], labels: np.ndarray, systems: dict, extra: dict | None = None):
    doc = {
        "n_vertices": int(labels.shape[0]),
        "hemi_offset": N_PER_HEMI,
        "atlas": atlas_name,
        "labels": labels.astype(int).tolist(),
        "names": names,
        "systems": systems,
    }
    if extra:
        doc.update(extra)
    path.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} kB)")


def build_destrieux():
    from nilearn import datasets

    d = datasets.fetch_atlas_surf_destrieux()
    raw = [l.decode() if isinstance(l, bytes) else str(l) for l in d["labels"]]
    names, labels = combine_hemis(raw, d["map_left"], d["map_right"], unknown={"Unknown", "Medial_wall"})
    systems = expand_systems("destrieux", names, labels, strict=True)
    sanity_table("destrieux", names, labels, systems)
    write_atlas(OUT / "atlas_destrieux.json", "destrieux", names, labels, systems,
                {"source": "nilearn fetch_atlas_surf_destrieux (Destrieux 2010, aparc.a2009s on fsaverage5)"})
    return names, labels, systems


def build_glasser():
    """HCP-MMP1 (Glasser 2016) fsaverage .annot sliced to fsaverage5 (its first 10242 vertices per hemi)."""
    import nibabel.freesurfer as fsio

    t0 = time.time()
    cache = Path.home() / "nilearn_data" / "hcp_mmp1"
    cache.mkdir(parents=True, exist_ok=True)
    maps, raw_names = {}, None
    for hemi, url in GLASSER_URLS.items():
        f = cache / f"{hemi}.HCP-MMP1.annot"
        if not f.exists() or f.stat().st_size < 100_000:
            print(f"downloading {url} -> {f}")
            with urllib.request.urlopen(url, timeout=120) as r:
                f.write_bytes(r.read())
        vlabels, ctab, nms = fsio.read_annot(str(f))
        nms = [n.decode() if isinstance(n, bytes) else str(n) for n in nms]
        assert vlabels.shape[0] == 163842, vlabels.shape  # fsaverage (ico7)
        # fsaverage5 = first 10242 vertices of fsaverage per hemisphere (icosahedron subdivision nesting)
        v5 = vlabels[:N_PER_HEMI].copy()
        v5[v5 < 0] = 0  # unlabeled vertices -> "???"
        # strip the hemisphere prefix + "_ROI" so both hemis share bare names, e.g. "L_V1_ROI" -> "V1"
        bare = []
        for n in nms:
            b = n
            if b[:2] in ("L_", "R_"):
                b = b[2:]
            if b.endswith("_ROI"):
                b = b[:-4]
            bare.append(b)
        if raw_names is None:
            raw_names = bare
        else:
            assert raw_names == bare, "lh/rh annot name tables differ"
        maps[hemi] = v5
        if time.time() - t0 > GLASSER_TIME_BUDGET_S:
            raise TimeoutError("glasser time budget exceeded")
    names, labels = combine_hemis(raw_names, maps["lh"], maps["rh"], unknown={"???", "Unknown", "Medial_wall"})
    systems = expand_systems("glasser", names, labels, strict=False)
    sanity_table("glasser", names, labels, systems)
    return names, labels, systems


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_mesh()
    d_names, d_labels, d_systems = build_destrieux()

    glasser = None
    try:
        glasser = build_glasser()
    except Exception as e:  # optional: any failure -> Destrieux only
        print(f"glasser unavailable ({type(e).__name__}: {e}); using Destrieux as the primary atlas")

    if glasser is not None:
        g_names, g_labels, g_systems = glasser
        write_atlas(OUT / "atlas.json", "glasser", g_names, g_labels, g_systems,
                    {"source": "HCP-MMP1.0 (Glasser et al. 2016) projected on fsaverage, figshare 3498446, sliced to fsaverage5",
                     "fallback": "atlas_destrieux.json"})
    else:
        write_atlas(OUT / "atlas.json", "destrieux", d_names, d_labels, d_systems,
                    {"source": "nilearn fetch_atlas_surf_destrieux (Destrieux 2010, aparc.a2009s on fsaverage5)"})
    print("\ndone:", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
