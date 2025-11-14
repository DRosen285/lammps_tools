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
#Covalent radii in A: bond_length=covalent_radius(A)+covalent_radius(B)
COVALENT_RADII = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "K": 2.03, "Ca": 1.76, "Fe": 1.32, "Cu": 1.32, "Zn": 1.22, "Br": 1.2, "I": 1.39
}

# Covalent radii in Å
COVALENT_RADII = {
    "H": 0.31, "Li": 1.28, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07
}

# Pair-specific tolerances (Å)
PAIR_EXTRA = {
    ("C", "H"): 0.4,
    ("C", "C"): 0.4,
    ("C", "O"): 0.5,
    ("O", "H"): 0.4,
    ("P", "F"): 0.5,
    ("Li", "O"): 0.6,
    ("Li", "F"): 0.6
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


def detect_bonds_fast(frame, type_map, default_tol=0.5):
    """
    Detect bonds using covalent radii + pair-specific extra tolerance.
    Uses KDTree for fast neighbor search.
    """
    coords = frame.coords  # Nx3 array
    types = frame.types    # list of atom types (integers)
    n = len(types)

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, type=int(types[i]))

    # Build KDTree
    tree = cKDTree(coords)

    # Determine max possible cutoff
    max_radii = max(COVALENT_RADII.get(type_map[str(t)], 0.8) for t in types)
    max_cutoff = 2*max_radii + max(PAIR_EXTRA.values(), default=default_tol)

    # Find all pairs within max_cutoff
    pairs = tree.query_pairs(r=max_cutoff)

    for i, j in pairs:
        elem_i = type_map[str(types[i])]
        elem_j = type_map[str(types[j])]

        # Determine pair-specific tolerance
        tol = PAIR_EXTRA.get((elem_i, elem_j),
                              PAIR_EXTRA.get((elem_j, elem_i), default_tol))

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



def cluster_fragments(frame, type_map, default_tol=0.5):
    """
    Cluster connected atoms into fragments using pair-specific cutoff.
    Returns a list of fragments, each as a tuple of sorted element symbols.
    """
    G = detect_bonds_fast(frame, type_map, default_tol=default_tol)
    fragments = []

    for comp in nx.connected_components(G):
        atoms = sorted(list(comp))
        frag_types = tuple(sorted([type_map[str(frame.types[i])] for i in atoms]))
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

def analyze_dump(dumpfile, type_map, outdir="results", tol=0.5, min_lifetime_frames=10):
    """
    Analyze LAMMPS trajectory and compute stable fragment lifetimes using
    pair-specific bond cutoffs, with a progress bar.
    """
    fragments_seen = defaultdict(list)  # fragment tuple -> list of frame indices

    # First, get total number of frames for tqdm (optional, for known-size trajectories)
    # If unknown, leave total=None
    total_frames = None
    try:
        with open(dumpfile, 'r') as fh:
            total_frames = sum(1 for line in fh if line.strip() == "ITEM: TIMESTEP")
    except:
        total_frames = None  # fallback if file is huge or unknown

    print("Parsing frames from dump...")
    for frame_idx, frame in enumerate(tqdm(parse_lammps_dump(dumpfile), total=total_frames, desc="Frames")):
        # Get fragments using pair-specific cutoffs
        frags = cluster_fragments(frame, type_map, default_tol=tol)

        for frag in frags:
            fragments_seen[frag].append(frame_idx)

    # Compute maximum lifetime for each fragment
    stable_fragments = {}
    for frag, frames in fragments_seen.items():
        frames = sorted(frames)
        max_run = 0
        run = 1
        for i in range(1, len(frames)):
            if frames[i] == frames[i-1] + 1:
                run += 1
            else:
                max_run = max(max_run, run)
                run = 1
        max_run = max(max_run, run)

        if max_run >= min_lifetime_frames:
            stable_fragments[frag] = max_run

    # Save results to JSON
    out_file = f"{outdir}/stable_fragments.json"
    with open(out_file, "w") as f:
        json.dump({",".join(frag): life for frag, life in stable_fragments.items()}, f, indent=2)

    print(f"Analysis complete. {len(stable_fragments)} stable fragments saved to {out_file}")
    return stable_fragments


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

