#!/usr/bin/env python3
"""
Analyze thermal decomposition products from MD simulation dumps.
Parses a LAMMPS custom dump file with:
  ITEM: TIMESTEP
  ITEM: NUMBER OF ATOMS
  ITEM: BOX BOUNDS pp pp pp
  ITEM: ATOMS id type mass x y z xu yu zu

Generates:
  - species_timeseries.csv : counts of unique fragments vs time
  - fragments.json         : unique fragment composition & SMILES
  - events.csv             : appearance/disappearance of fragments
"""

import numpy as np
import json
import pandas as pd
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass
import os

# tqdm for progress bar (graceful fallback)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# optional RDKit support
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False
    print("Warning: RDKit not available. SMILES generation will be skipped.")

#to compute adaptive bond cutoffs
COVALENT_RADII = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "K": 2.03, "Ca": 1.76, "Fe": 1.32, "Cu": 1.32, "Zn": 1.22, "Br": 1.2, "I": 1.39
}

@dataclass
class Frame:
    natoms: int
    box: np.ndarray
    ids: np.ndarray
    types: np.ndarray
    masses: np.ndarray
    coords: np.ndarray
    timestep: int


# --- Parsing the LAMMPS dump file ---
def parse_lammps_dump(filename):
    """
    Generator yielding Frame objects for a LAMMPS custom dump with header:
    ITEM: TIMESTEP
    <int>
    ITEM: NUMBER OF ATOMS
    <int>
    ITEM: BOX BOUNDS ...
    <3 lines of floats>
    ITEM: ATOMS id type mass x y z xu yu zu
    <N lines of atom data>
    """
    with open(filename, "r") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep_line = fh.readline()
            if not timestep_line:
                break
            try:
                timestep = int(timestep_line.strip())
            except ValueError:
                continue

            # number of atoms
            l = fh.readline()
            if not l or not l.startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("Expected 'ITEM: NUMBER OF ATOMS'")
            natoms = int(fh.readline().strip())

            # box bounds
            l = fh.readline()
            if not l.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("Expected 'ITEM: BOX BOUNDS'")
            box = []
            for _ in range(3):
                parts = fh.readline().split()
                box.append([float(parts[0]), float(parts[1])])
            box = np.array(box)

            # atoms section
            header = fh.readline().strip()
            if not header.startswith("ITEM: ATOMS"):
                raise RuntimeError("Expected 'ITEM: ATOMS'")
            cols = header.split()[2:]
            col_idx = {c: i for i, c in enumerate(cols)}
            use_unwrapped = all(k in col_idx for k in ("xu", "yu", "zu"))
            coords_present = all(k in col_idx for k in ("x", "y", "z"))

            ids = np.empty(natoms, dtype=int)
            types = np.empty(natoms, dtype=int)
            masses = np.empty(natoms, dtype=float)
            coords = np.zeros((natoms, 3), dtype=float)

            for i in range(natoms):
                parts = fh.readline().split()
                ids[i] = int(parts[col_idx["id"]])
                types[i] = int(parts[col_idx["type"]])
                masses[i] = float(parts[col_idx["mass"]]) if "mass" in col_idx else np.nan
                if use_unwrapped:
                    coords[i] = [
                        float(parts[col_idx["xu"]]),
                        float(parts[col_idx["yu"]]),
                        float(parts[col_idx["zu"]]),
                    ]
                elif coords_present:
                    coords[i] = [
                        float(parts[col_idx["x"]]),
                        float(parts[col_idx["y"]]),
                        float(parts[col_idx["z"]]),
                    ]
                else:
                    raise RuntimeError("No coordinate columns found in dump.")

            order = np.argsort(ids)
            yield Frame(
                natoms=natoms,
                box=box,
                ids=ids[order],
                types=types[order],
                masses=masses[order],
                coords=coords[order],
                timestep=timestep,
            )


# --- Core analysis functions ---
#def detect_bonds(frame, cutoff=1.6):
#    """Builds a bond list from atomic positions using distance cutoff."""
#    coords = frame.coords
#    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
#    dist = np.linalg.norm(diff, axis=-1)
#    bonded = (dist < cutoff) & (dist > 0)
#    G = nx.Graph()
#    for i in range(frame.natoms):
#        G.add_node(i, type=int(frame.types[i]))
#    for i, j in zip(*np.where(bonded)):
#        if i < j:
#            G.add_edge(i, j)
#    return G

from scipy.spatial import cKDTree
import numpy as np
import networkx as nx

def detect_bonds_fast(frame, type_map, tol=0.4):
    """
    Detect bonds using covalent radii and neighbor search (KDTree).
    Much faster than O(N^2) for large systems.
    """
    coords = frame.coords
    types = frame.types
    n = frame.natoms
    G = nx.Graph()
    
    # Add nodes
    for i in range(n):
        G.add_node(i, type=int(types[i]))
    
    # Compute max cutoff per frame (largest covalent radius pair + tol)
    max_radii = max(COVALENT_RADII.get(type_map[str(t)], 0.8) for t in types)
    max_cutoff = 2*max_radii + tol
    
    # Build KDTree
    tree = cKDTree(coords)
    
    # Find pairs within max cutoff
    pairs = tree.query_pairs(r=max_cutoff, output_type='set')
    
    # Filter pairs using actual element-specific radii
    for i, j in pairs:
        elem_i = type_map[str(types[i])]
        elem_j = type_map[str(types[j])]
        cutoff = COVALENT_RADII.get(elem_i, 0.8) + COVALENT_RADII.get(elem_j, 0.8) + tol
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist <= cutoff:
            G.add_edge(i, j)
    
    return G

#def cluster_fragments(frame, cutoff=1.6):
#    """Cluster atoms into fragments based on bonding network."""
#    G = detect_bonds(frame, cutoff=cutoff)
#    fragments = []
#    for comp in nx.connected_components(G):
#        atoms = sorted(list(comp))
#        frag_types = tuple(sorted(frame.types[atoms]))
#        fragments.append(frag_types)
#    return fragments


def cluster_fragments(frame, type_map, tol=0.4):
    G = detect_bonds_fast(frame, type_map=type_map, tol=tol)
    fragments = []
    for comp in nx.connected_components(G):
        atoms = sorted(list(comp))
        frag_types = tuple(sorted(frame.types[atoms]))
        fragments.append(frag_types)
    return fragments

def smiles_from_fragment(frag_types):
    """Placeholder SMILES generation (mock by element types)."""
    if not RDKit_AVAILABLE:
        return None
    # Convert numeric types to element symbols if you have a map
    # For now just create a fake molecule with single atoms
    mol = Chem.RWMol()
    for _ in frag_types:
        mol.AddAtom(Chem.Atom("C"))
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# --- Main analysis driver ---
def analyze_dump(dumpfile, type_map, outdir="results", tol=1.6, min_lifetime_frames=2):
    os.makedirs(outdir, exist_ok=True)
    # Estimate number of frames
    with open(dumpfile) as f:
        nframes = sum(1 for line in f if line.startswith("ITEM: TIMESTEP"))
    print(f"Found {nframes} frames. Beginning analysis...")

    species_counts = []
    fragments_seen = {}
    events = []

    pbar = tqdm(total=nframes, desc="Parsing frames", dynamic_ncols=True)
    for i, fr in enumerate(parse_lammps_dump(dumpfile)):
        #frags = cluster_fragments(fr, cutoff=tol)
        frags = cluster_fragments(fr, type_map=type_map, tol=tol)
        frame_count = defaultdict(int)
        for frag in frags:
            frame_count[frag] += 1
            if frag not in fragments_seen:
                fragments_seen[frag] = {
                    "composition": {type_map.get(str(t), str(t)): frag.count(t) for t in set(frag)},
                    "first_seen": fr.timestep,
                    "smiles": smiles_from_fragment(frag),
                }

        species_counts.append({"timestep": fr.timestep, **frame_count})

        pbar.set_description(f"Frame {i+1}/{nframes} (t={fr.timestep}, frags={len(frags)})")
        pbar.update(1)

    pbar.close()

    # Convert species count time series to DataFrame
    df = pd.DataFrame(species_counts).fillna(0)
    df.to_csv(os.path.join(outdir, "species_timeseries.csv"), index=False)

    # Write fragments summary
    # Convert tuple keys to strings for JSON
    json_fragments = {str(k): v for k, v in fragments_seen.items()}

    with open(os.path.join(outdir, "fragments.json"), "w") as f:
         json.dump(json_fragments, f, indent=2)

# --- Command-line interface ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze decomposition products from MD dump.")
    parser.add_argument("--dump", required=True, help="LAMMPS dump file")
    parser.add_argument("--map", required=True, help="JSON file mapping type->element symbol")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--tol", type=float, default=1.6, help="Bond cutoff distance (Å)")
    parser.add_argument("--min_life", type=int, default=2, help="Minimum lifetime in frames")

    args = parser.parse_args()

    with open(args.map) as f:
        type_map = json.load(f)

    analyze_dump(args.dump, type_map, outdir=args.outdir, tol=args.tol, min_lifetime_frames=args.min_life)

