# -*- coding: utf-8 -*-
"""
Graph Edit Network (GEN) for metabolite prediction
===========================================================
"""

# ─────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────
import io, os, re, ast, math, json, random, warnings
from contextlib import nullcontext 
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set, Any

warnings.filterwarnings("ignore")

# Suppress all RDKit C++ layer warnings/errors
from rdkit import RDLogger as _rdlogger
_rdlogger.DisableLog("rdApp.*")

# ─────────────────────────────────────────────────────────────
# Third-party
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score as sk_f1
import copy 
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import autocast, GradScaler
    def _autocast_ctx(device_type: str, enabled: bool):
        return autocast(device_type=device_type, enabled=enabled)
    def _make_grad_scaler(enabled: bool):
        return GradScaler(device="cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _cuda_autocast, GradScaler as _CudaGradScaler
    def _autocast_ctx(device_type: str, enabled: bool):
        if device_type == "cuda":
            return _cuda_autocast(enabled=enabled)
        return nullcontext()
    def _make_grad_scaler(enabled: bool):
        return _CudaGradScaler(enabled=enabled)

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import (AllChem, rdMolDescriptors as rdMD, Crippen, Draw)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool

try:
    from PIL import Image, ImageDraw as PILDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import joblib
except ImportError:
    import sklearn.externals.joblib as joblib  # type: ignore


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
def enable_reproducibility(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
    if hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def write_log(path: str, line: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def detach_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def resolve_amp(device: torch.device, use_amp: bool) -> bool:
    return bool(use_amp and device.type == "cuda" and torch.cuda.is_available())

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def safe_literal_list(s: str) -> List[int]:
    if not isinstance(s, str) or not s.strip(): return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [int(v) for v in obj if str(v).strip().lstrip("-").isdigit()]
    except Exception: pass
    return []

def parse_symbol_sites(s: str) -> List[int]:
    if not isinstance(s, str) or not s.strip(): return []
    try:
        inside = s.strip().strip("[]")
        return [int(m.group(1))
                for tok in inside.split(",")
                for m in [re.search(r":(\d+)", tok)] if m]
    except Exception: return []

def bonds_touching(mol: Chem.Mol, atoms: List[int]) -> List[int]:
    if not atoms: return []
    aset = set(atoms)
    return [b.GetIdx() for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in aset or b.GetEndAtomIdx() in aset]

def mol_from_mapped_smiles(mapped_smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {mapped_smiles}")
    amap = [int(a.GetAtomMapNum()) for a in mol.GetAtoms()]
    if not any(amap): return mol
    pairs = sorted(enumerate(amap), key=lambda t: t[1])
    order = [int(i) for i, _ in pairs]
    mol2  = Chem.RenumberAtoms(mol, order)
    for a in mol2.GetAtoms(): a.SetAtomMapNum(0)
    return mol2

def add_atom_numbers(mol: Chem.Mol) -> Chem.Mol:
    em = Chem.RWMol(Chem.Mol(mol))
    for a in em.GetAtoms(): a.SetAtomMapNum(a.GetIdx())
    return em.GetMol()


# ─────────────────────────────────────────────────────────────
# Murcko scaffold splits
# ─────────────────────────────────────────────────────────────
def murcko_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return ""
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception: return ""

def build_parent_scaffold_df(df: pd.DataFrame,
                             parent_col: str = "Predecessor_SMILES") -> pd.DataFrame:
    """
    Build one-row-per-unique-parent table for scaffold-aware splitting.

    parent_id is the canonical parent SMILES so duplicate reaction rows for the same
    parent stay together. n_rows_parent tracks how many dataset rows belong to that
    parent so split diagnostics can report both parent and row counts.
    """
    tmp = df.copy().reset_index(drop=False).rename(columns={"index": "row_idx"})
    tmp["parent_id"] = tmp[parent_col].astype(str).str.strip()
    tmp["scaffold"] = tmp["parent_id"].apply(murcko_scaffold)
    parent_df = tmp.groupby(["parent_id", "scaffold"], as_index=False).agg(
        n_rows_parent=("row_idx", "count")
    )
    return parent_df


def optimize_scaffold_holdout(parent_df: pd.DataFrame, test_frac: float, seed: int) -> Tuple[set, set, Dict]:
    """
    Choose held-out scaffold groups using parent counts as the primary objective.
    This avoids the previous largest-first row-based split that could select one giant scaffold.
    """
    grp = parent_df.groupby('scaffold').agg(
        n_parents=('parent_id', 'nunique'),
        n_rows=('n_rows_parent', 'sum')
    ).reset_index()

    scaffolds = grp['scaffold'].tolist()
    parent_counts = grp['n_parents'].tolist()
    row_counts = grp['n_rows'].tolist()

    total_parents = int(sum(parent_counts))
    total_rows = int(sum(row_counts))
    target_parents = max(1, int(round(total_parents * test_frac)))
    target_rows = max(1, int(round(total_rows * test_frac)))
    max_parent = max(parent_counts) if parent_counts else 0
    max_sum = min(total_parents, target_parents + max_parent) if total_parents else 0

    rng = random.Random(seed)
    order = list(range(len(scaffolds)))
    rng.shuffle(order)

    # DP over parent counts. For each achievable parent-count sum, keep subset with:
    # 1) more scaffold groups
    # 2) row count closer to target_rows
    dp: Dict[int, Tuple[int, int, Tuple[int, ...]]] = {0: (0, 0, tuple())}
    for idx in order:
        pc = int(parent_counts[idx])
        rc = int(row_counts[idx])
        items = sorted(list(dp.items()), key=lambda x: x[0], reverse=True)
        for s, (ng, rows, chosen) in items:
            ns = s + pc
            if ns > max_sum:
                continue
            cand = (ng + 1, rows + rc, chosen + (idx,))
            keep = dp.get(ns)
            if keep is None:
                dp[ns] = cand
            else:
                better = False
                if cand[0] > keep[0]:
                    better = True
                elif cand[0] == keep[0]:
                    if abs(cand[1] - target_rows) < abs(keep[1] - target_rows):
                        better = True
                if better:
                    dp[ns] = cand

    # choose best achievable sum; prefer closeness to target_parents, then more groups, then row closeness
    candidates = []
    for s, (ng, rows, chosen) in dp.items():
        if s == 0:
            continue
        candidates.append((abs(s - target_parents), -ng, abs(rows - target_rows), s, rows, chosen))
    candidates.sort()

    if not candidates:
        # fallback random split by unique scaffold groups
        all_scaffolds = set(scaffolds)
        shuffled = scaffolds[:]
        rng.shuffle(shuffled)
        test_scaffolds = set(shuffled[: max(1, int(round(len(scaffolds) * test_frac)))])
        train_scaffolds = all_scaffolds - test_scaffolds
        info = {
            'target_parents': target_parents,
            'target_rows': target_rows,
            'selected_parents': int(parent_df[parent_df['scaffold'].isin(test_scaffolds)]['parent_id'].nunique()),
            'selected_rows': int(parent_df[parent_df['scaffold'].isin(test_scaffolds)]['n_rows_parent'].sum()),
            'n_test_scaffolds': len(test_scaffolds),
            'largest_scaffold_parents': max_parent,
            'method': 'fallback_random',
        }
        return train_scaffolds, test_scaffolds, info

    _, neg_ng, _, selected_parents, selected_rows, chosen_idx = candidates[0]
    test_scaffolds = {scaffolds[i] for i in chosen_idx}
    train_scaffolds = set(scaffolds) - test_scaffolds
    info = {
        'target_parents': target_parents,
        'target_rows': target_rows,
        'selected_parents': int(selected_parents),
        'selected_rows': int(selected_rows),
        'n_test_scaffolds': int(-neg_ng),
        'largest_scaffold_parents': int(max_parent),
        'method': 'parent_count_dp',
    }
    return train_scaffolds, test_scaffolds, info


def scaffold_test_split(df: pd.DataFrame,
                        test_frac: float = 0.10,
                        seed: int = 42,
                        parent_col: str = "Predecessor_SMILES") -> Tuple[np.ndarray, np.ndarray, Dict, pd.DataFrame]:
    """
    Held-out scaffold split on unique parents, then expanded back to row indices.
    """
    parent_df = build_parent_scaffold_df(df, parent_col=parent_col)
    train_scaffolds, test_scaffolds, info = optimize_scaffold_holdout(parent_df, test_frac, seed)

    tmp = df.copy().reset_index(drop=False).rename(columns={"index": "row_idx"})
    tmp["parent_id"] = tmp[parent_col].astype(str).str.strip()
    tmp["scaffold"] = tmp["parent_id"].apply(murcko_scaffold)

    tst_idx = tmp.loc[tmp["scaffold"].isin(test_scaffolds), "row_idx"].to_numpy(dtype=int)
    pre_idx = tmp.loc[tmp["scaffold"].isin(train_scaffolds), "row_idx"].to_numpy(dtype=int)
    return pre_idx, tst_idx, info, parent_df


def scaffold_kfold_split(parent_df: pd.DataFrame,
                         n_splits: int = 5,
                         seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Scaffold-grouped CV on unique parents. Folds are balanced primarily by number of
    unique parents and secondarily by row counts, then returned as parent-index arrays.
    """
    grp = parent_df.groupby('scaffold').agg(
        parent_indices=('parent_id', lambda s: tuple(parent_df.loc[s.index, 'parent_index'].tolist())),
        n_parents=('parent_id', 'nunique'),
        n_rows=('n_rows_parent', 'sum')
    ).reset_index()

    records = grp.to_dict('records')
    rng = random.Random(seed)
    rng.shuffle(records)
    records.sort(key=lambda r: (-int(r['n_parents']), -int(r['n_rows']), str(r['scaffold'])))

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    fold_parent_counts = [0] * n_splits
    fold_row_counts = [0] * n_splits

    for rec in records:
        target_fold = min(range(n_splits), key=lambda i: (fold_parent_counts[i], fold_row_counts[i], i))
        pidx = [int(v) for v in rec['parent_indices']]
        folds[target_fold].extend(pidx)
        fold_parent_counts[target_fold] += int(rec['n_parents'])
        fold_row_counts[target_fold] += int(rec['n_rows'])

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        val = np.array(sorted(folds[i]), dtype=int)
        train = np.array(sorted([idx for j, fold in enumerate(folds) for idx in fold if j != i]), dtype=int)
        splits.append((train, val))
    return splits


# ─────────────────────────────────────────────────────────────
# Atom features  (19-d per atom)
# ─────────────────────────────────────────────────────────────
VALENCE_ELECTRONS = {1:1, 5:3, 6:4, 7:5, 8:6, 9:7, 15:5, 16:6, 17:7, 35:7, 53:7}

def _benzylic_carbon(atom, mol):
    if atom.GetAtomicNum()!=6 or atom.GetHybridization()!=Chem.HybridizationType.SP3: return 0
    return int(any(nb.GetIsAromatic() and nb.GetAtomicNum()==6 for nb in atom.GetNeighbors()))

def _allylic_carbon(atom):
    if atom.GetAtomicNum()!=6 or atom.GetHybridization()!=Chem.HybridizationType.SP3: return 0
    return int(any(nb.GetHybridization() in (Chem.HybridizationType.SP2,
                                              Chem.HybridizationType.SP)
                   for nb in atom.GetNeighbors()))

def _tertiary_amine_n(atom):
    if atom.GetAtomicNum()==7 and atom.GetDegree()==3 and not atom.GetIsAromatic():
        return int(all(n.GetAtomicNum()!=7 for n in atom.GetNeighbors()))
    return 0

def _hetero_softspot(atom): return int(atom.GetAtomicNum() in (7,8,15,16))
def _is_linker(atom, mol):
    rn = [n for n in atom.GetNeighbors() if n.IsInRing()]
    return int((not atom.IsInRing() and len(rn)>=2) or (atom.IsInRing() and len(rn)>=3))
def _ring_count(atom, mol): return mol.GetRingInfo().NumAtomRings(atom.GetIdx())
def _arom_nb_frac(atom):
    d = atom.GetDegree()
    return sum(1 for n in atom.GetNeighbors() if n.GetIsAromatic()) / float(d) if d else 0.0

def _d2_heavy(atom, mol):
    idx=atom.GetIdx(); vis={idx}
    l1={n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum()>1}; vis|=l1
    l2=set()
    for i in l1:
        for n2 in mol.GetAtomWithIdx(i).GetNeighbors():
            j=n2.GetIdx()
            if j not in vis and n2.GetAtomicNum()>1: l2.add(j)
    return len(l2)

def _lone_pair(atom):
    ve=VALENCE_ELECTRONS.get(atom.GetAtomicNum(),0)
    bo=sum(1 if b.GetBondType()==Chem.BondType.SINGLE else
           2 if b.GetBondType()==Chem.BondType.DOUBLE else
           3 if b.GetBondType()==Chem.BondType.TRIPLE else
           1.5 if b.GetBondType()==Chem.BondType.AROMATIC else 0
           for b in atom.GetBonds())
    return float(0.5*max(0.0, ve-bo-atom.GetFormalCharge()))

def _is_thioether_s(atom, mol):
    """S in C–S–C with no =O neighbours → thioether, prime S-oxidation site."""
    if atom.GetAtomicNum() != 16: return 0
    c_nbrs = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 6]
    o_dbl  = [nb for nb in atom.GetNeighbors()
              if nb.GetAtomicNum() == 8 and
                 mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx()).GetBondType()
                 == Chem.BondType.DOUBLE]
    return int(len(c_nbrs) >= 2 and len(o_dbl) == 0)

def _alpha_to_n(atom, mol):
    """sp3 C directly bonded to N → typical N-dealkylation α-carbon."""
    if atom.GetAtomicNum() != 6: return 0
    if atom.GetHybridization() != Chem.HybridizationType.SP3: return 0
    return int(any(nb.GetAtomicNum() == 7 for nb in atom.GetNeighbors()))

def _benzylic_n(atom):
    """N bonded to at least one aromatic carbon → N-oxidation softspot."""
    if atom.GetAtomicNum() != 7: return 0
    return int(any(nb.GetIsAromatic() and nb.GetAtomicNum() == 6
                   for nb in atom.GetNeighbors()))

def _sp2_nonaromatic_ring(atom):
    """sp2 C in a non-aromatic ring → epoxidation candidate."""
    if atom.GetAtomicNum() != 6: return 0
    if atom.GetHybridization() != Chem.HybridizationType.SP2: return 0
    if not atom.IsInRing(): return 0
    return int(not atom.GetIsAromatic())
# ────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# ATOM FEATURE GROUPS
# ═════════════════════════════════════════════════════════════════
# The 19 atom descriptors fall into two groups:
#
#   GENERIC (10)     – general-purpose physicochemical / topological
#                      descriptors that any molecular GNN would use.
#   ENGINEERED (9)   – knowledge-driven metabolic soft-spot indicators
#                      hand-designed for this task (benzylic/allylic C,
#                      tertiary amine N, heteroatom soft spot, linker,
#                      thioether S, alpha-to-N, benzylic N, sp2
#                      non-aromatic ring).

ATOM_FEATURE_NAMES = [
    "gasteiger_charge",          # 0   generic
    "mean_neighbour_charge",     # 1   generic
    "valence_electrons",         # 2   generic
    "tpsa_contrib",              # 3   generic
    "lone_pair_count",           # 4   generic
    "d2_heavy_neighbours",       # 5   generic
    "benzylic_carbon",           # 6   ENGINEERED
    "allylic_carbon",            # 7   ENGINEERED
    "tertiary_amine_N",          # 8   ENGINEERED
    "heteroatom_softspot",       # 9   ENGINEERED
    "is_linker",                 # 10  ENGINEERED
    "ring_membership_count",     # 11  generic
    "crippen_logp_contrib",      # 12  generic
    "crippen_mr_contrib",        # 13  generic
    "aromatic_neighbour_frac",   # 14  generic
    "thioether_S",               # 15  ENGINEERED
    "alpha_to_N",                # 16  ENGINEERED
    "benzylic_N",                # 17  ENGINEERED
    "sp2_nonaromatic_ring",      # 18  ENGINEERED
]
N_ATOM_FEATURES        = len(ATOM_FEATURE_NAMES)
ENGINEERED_FEATURE_IDX = [6, 7, 8, 9, 10, 15, 16, 17, 18]
GENERIC_FEATURE_IDX    = [i for i in range(N_ATOM_FEATURES)
                          if i not in ENGINEERED_FEATURE_IDX]

# Module-level switch consulted by compute_atom_features(). Set ONCE at the
# start of a run via set_feature_mode(); every graph built afterwards (including
# augmented copies) uses the same feature set.
FEATURE_MODE = "full"          # "full" | "no_engineered"

def set_feature_mode(mode: str) -> None:
    global FEATURE_MODE
    if mode not in ("full", "no_engineered"):
        raise ValueError(f"Unknown feature mode: {mode}")
    FEATURE_MODE = mode

def active_feature_indices() -> List[int]:
    return (list(range(N_ATOM_FEATURES)) if FEATURE_MODE == "full"
            else list(GENERIC_FEATURE_IDX))

def active_feature_names() -> List[str]:
    return [ATOM_FEATURE_NAMES[i] for i in active_feature_indices()]

def active_feature_dim() -> int:
    return len(active_feature_indices())


def compute_atom_features(mol: Chem.Mol) -> np.ndarray:
    """
    Return an (N_atoms, D) float32 feature matrix.

    D = 19 when FEATURE_MODE == "full"
    D = 10 when FEATURE_MODE == "no_engineered"  (engineered soft-spot flags dropped)
    """
    try:
        AllChem.ComputeGasteigerCharges(mol)
        gc=[0.0 if (math.isnan(v:=a.GetDoubleProp("_GasteigerCharge")
                               if a.HasProp("_GasteigerCharge") else 0.0)
                   or math.isinf(v)) else v
            for a in mol.GetAtoms()]
    except Exception: gc=[0.0]*mol.GetNumAtoms()
    mn=[]
    for a in mol.GetAtoms():
        nbs=a.GetNeighbors()
        if not nbs: mn.append(0.0)
        else:
            vs=[]
            for n in nbs:
                try:
                    v=n.GetDoubleProp("_GasteigerCharge") if n.HasProp("_GasteigerCharge") else 0.0
                    vs.append(0.0 if (math.isnan(v) or math.isinf(v)) else v)
                except Exception: vs.append(0.0)
            mn.append(float(np.mean(vs)))
    try: tpsa=list(rdMD._CalcTPSAContribs(mol))
    except Exception: tpsa=[0.0]*mol.GetNumAtoms()
    try:
        lc,mc=[],[]
        for lp,mr in Crippen._GetAtomContribs(mol): lc.append(lp); mc.append(mr)
    except Exception: lc=[0.0]*mol.GetNumAtoms(); mc=[0.0]*mol.GetNumAtoms()
    feats=[]
    for a in mol.GetAtoms():
        i=a.GetIdx()
        feats.append([
            # original 15 features
            gc[i], mn[i],
            float(VALENCE_ELECTRONS.get(a.GetAtomicNum(),0)),
            tpsa[i] if i<len(tpsa) else 0.0,
            _lone_pair(a), float(_d2_heavy(a,mol)),
            float(_benzylic_carbon(a,mol)), float(_allylic_carbon(a)),
            float(_tertiary_amine_n(a)), float(_hetero_softspot(a)),
            float(_is_linker(a,mol)), float(_ring_count(a,mol)),
            lc[i] if i<len(lc) else 0.0, mc[i] if i<len(mc) else 0.0,
            _arom_nb_frac(a),
            # P3: 4 new minority-class-targeted features
            float(_is_thioether_s(a, mol)),
            float(_alpha_to_n(a, mol)),
            float(_benzylic_n(a)),
            float(_sp2_nonaromatic_ring(a)),
        ])
    arr = np.asarray(feats, dtype=np.float32).reshape(-1, N_ATOM_FEATURES)
    idx = active_feature_indices()
    if len(idx) != N_ATOM_FEATURES:
        arr = arr[:, idx]
    return np.ascontiguousarray(arr, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Bond-type edge features (4-d one-hot)
# ─────────────────────────────────────────────────────────────
_BOND_TYPE_MAP = {
    Chem.BondType.SINGLE:   0,
    Chem.BondType.DOUBLE:   1,
    Chem.BondType.TRIPLE:   2,
    Chem.BondType.AROMATIC: 3,
}
EDGE_FEAT_DIM = 4  # single / double / triple / aromatic

def _edge_index_and_attr(mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (edge_index [2, 2E], edge_attr [2E, 4]) for the molecule."""
    src, dst, attrs = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt_idx = _BOND_TYPE_MAP.get(b.GetBondType(), 0)
        one_hot = [0.0] * EDGE_FEAT_DIM
        one_hot[bt_idx] = 1.0
        # both directions
        src += [i, j]; dst += [j, i]
        attrs += [one_hot, one_hot]
    ei   = torch.tensor([src, dst], dtype=torch.long)
    attr = torch.tensor(attrs, dtype=torch.float) if attrs else torch.zeros(0, EDGE_FEAT_DIM)
    return ei, attr

def mol_to_som_graph(mol, y_atoms):
    x  = torch.tensor(compute_atom_features(mol), dtype=torch.float)
    ei, ea = _edge_index_and_attr(mol)
    y  = torch.zeros((mol.GetNumAtoms(),1), dtype=torch.float)
    for idx in y_atoms:
        if 0<=idx<mol.GetNumAtoms(): y[idx,0]=1.0
    mm = Chem.Mol(mol)
    for i,a in enumerate(mm.GetAtoms()): a.SetAtomMapNum(i)
    d = Data(x=x, edge_index=ei, edge_attr=ea, y=y)
    d.smiles=Chem.MolToSmiles(mol,canonical=True)
    d.smiles_mapped=Chem.MolToSmiles(mm,canonical=False)
    d.successor_smiles=""
    d.gt_metabolites_json="[]"
    d.entry_id=-1
    return d


def mol_to_rxn_graph(mol, rxn_label):
    x  = torch.tensor(compute_atom_features(mol), dtype=torch.float)
    ei, ea = _edge_index_and_attr(mol)
    d  = Data(x=x, edge_index=ei, edge_attr=ea,
              y=torch.tensor([rxn_label], dtype=torch.long))
    d.smiles=Chem.MolToSmiles(mol,canonical=True)
    d.successor_smiles=""
    d.gt_metabolites_json="[]"
    d.entry_id=-1
    return d


# ─────────────────────────────────────────────────────────────
# SMILES augmentation for minority classes
# ─────────────────────────────────────────────────────────────
def random_smiles(mol: Chem.Mol, n: int = 4, seed: int = 0) -> List[str]:
    """
    Generate n distinct randomized, non-canonical isomeric SMILES.

    RDKit releases differ in whether ``MolToSmiles`` accepts a ``randomSeed``
    keyword.  v2 seeds RDKit's RNG through ``rdBase.SeedRandomNumberGenerator``
    instead, which is compatible with both older and newer releases.
    """
    seen: Set[str] = set()
    results: List[str] = []
    attempts = 0
    try:
        rdBase.SeedRandomNumberGenerator(int(seed) & 0x7FFFFFFF)
    except Exception:
        pass
    while len(results) < n and attempts < n * 20:
        attempts += 1
        try:
            smi = Chem.MolToSmiles(
                mol,
                isomericSmiles=True,
                canonical=False,
                doRandom=True,
            )
            if smi and smi not in seen:
                seen.add(smi)
                results.append(smi)
        except Exception:
            pass
    return results


def _randomized_mol_with_remapped_som(mol: Chem.Mol,
                                       y_atoms: List[int],
                                       random_smi: str) -> Tuple[Optional[Chem.Mol], List[int]]:

    if mol is None or not random_smi:
        return None, []

    rand_mol = Chem.MolFromSmiles(random_smi)
    if rand_mol is None or rand_mol.GetNumAtoms() != mol.GetNumAtoms():
        return None, []

    original_som = {int(i) for i in y_atoms if 0 <= int(i) < mol.GetNumAtoms()}
    new_som: List[int] = []
    seen_original: Set[int] = set()

    for atom in rand_mol.GetAtoms():
        map_num = int(atom.GetAtomMapNum())
        if map_num <= 0:
            # Every atom must carry the persistent map ID generated below.
            return None, []
        old_idx = map_num - 1
        if old_idx < 0 or old_idx >= mol.GetNumAtoms() or old_idx in seen_original:
            return None, []
        seen_original.add(old_idx)
        if old_idx in original_som:
            new_som.append(int(atom.GetIdx()))

    if len(seen_original) != mol.GetNumAtoms() or len(new_som) != len(original_som):
        return None, []

    for atom in rand_mol.GetAtoms():
        atom.SetAtomMapNum(0)

    try:
        src_can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        rnd_can = Chem.MolToSmiles(rand_mol, canonical=True, isomericSmiles=True)
        if src_can != rnd_can:
            return None, []
    except Exception:
        return None, []

    return rand_mol, sorted(new_som)


def random_smiles_with_atom_maps(mol: Chem.Mol,
                                  n: int = 4,
                                  seed: int = 0) -> List[str]:
    mapped = Chem.Mol(mol)
    for atom in mapped.GetAtoms():
        atom.SetAtomMapNum(int(atom.GetIdx()) + 1)
    return random_smiles(mapped, n=n, seed=seed)


def augment_minority_classes(sg_list: List[Data],
                              rg_list: List[Data],
                              raw_df: pd.DataFrame,
                              le: LabelEncoder,
                              aug_threshold: int = 60,
                              aug_factor: int = 4) -> Tuple[List[Data], List[Data]]:
    class_counts = raw_df["reaction_superclass"].value_counts().to_dict()
    aug_sg, aug_rg = list(sg_list), list(rg_list)
    n_rejected = 0

    for i, (sg, rg) in enumerate(zip(sg_list, rg_list)):
        rxn_cls = raw_df.iloc[i]["reaction_superclass"]
        if class_counts.get(rxn_cls, 999) >= aug_threshold:
            continue

        try:
            mapped_source = getattr(sg, "smiles_mapped", "")
            if not mapped_source:
                raise ValueError("SOM graph has no smiles_mapped atom-identity record")
            mol = mol_from_mapped_smiles(mapped_source)
        except Exception:
            n_rejected += max(int(aug_factor), 1)
            continue
        if mol is None:
            n_rejected += max(int(aug_factor), 1)
            continue

        succ_smi  = sg.successor_smiles
        gt_json   = getattr(sg, "gt_metabolites_json", "[]")
        entry_id  = sg.entry_id
        rxn_label = int(rg.y.item())
        y_atoms = [int(idx) for idx in
                   (sg.y.view(-1) > 0.5).nonzero(as_tuple=False).squeeze(-1).tolist()]
        mapped_random_smis = random_smiles_with_atom_maps(
            mol, n=aug_factor, seed=int(i)
        )

        for rand_smi in mapped_random_smis:
            try:
                rand_mol, remapped_y_atoms = _randomized_mol_with_remapped_som(
                    mol, y_atoms, rand_smi
                )
                if rand_mol is None:
                    n_rejected += 1
                    continue

                new_sg = mol_to_som_graph(rand_mol, remapped_y_atoms)
                new_sg.successor_smiles    = succ_smi
                new_sg.gt_metabolites_json = gt_json
                new_sg.entry_id            = entry_id

                new_rg = mol_to_rxn_graph(rand_mol, rxn_label)
                new_rg.successor_smiles    = succ_smi
                new_rg.gt_metabolites_json = gt_json
                new_rg.entry_id            = entry_id

                aug_sg.append(new_sg)
                aug_rg.append(new_rg)
            except Exception:
                n_rejected += 1
                continue

    added = len(aug_sg) - len(sg_list)
    if added > 0:
        msg = (f"    Randomized-SMILES oversampling: +{added} graphs for minority classes "
               f"(threshold={aug_threshold}, factor={aug_factor}); SOM labels remapped by atom identity")
        if n_rejected:
            msg += f"; rejected={n_rejected}"
        print(msg)
    elif n_rejected:
        print(f"    [warn] Randomized-SMILES oversampling added no graphs; rejected={n_rejected}")
    return aug_sg, aug_rg


# ─────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    required = {"Predecessor_SMILES", "atom_index", "SOM_index",
                "Successor_SMILES",   "reaction_superclass"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Normalise types
    df["atom_index"]  = df["atom_index"].astype(int)
    df["SOM_index"]   = df["SOM_index"].astype(int)
    df["Predecessor_SMILES"] = df["Predecessor_SMILES"].astype(str).str.strip()
    df["Successor_SMILES"]   = df["Successor_SMILES"].astype(str).str.strip()
    df["reaction_superclass"]= df["reaction_superclass"].astype(str).str.strip()

    # Build LabelEncoder from actual reaction classes (excluding No_reaction)
    rxn_classes = sorted(df.loc[df["reaction_superclass"] != "No_reaction",
                                "reaction_superclass"].unique().tolist())
    le = LabelEncoder()
    le.fit(rxn_classes)

    # Group by predecessor — collect SOM indices + reaction class per molecule
    sg_list, rg_list, repr_rows = [], [], []
    skipped = 0

    for pred_smi, grp in df.groupby("Predecessor_SMILES", sort=False):
        mol = Chem.MolFromSmiles(pred_smi)
        if mol is None:
            skipped += 1
            continue

        # SOM rows for this molecule
        som_rows = grp[grp["SOM_index"] == 1].copy()

        # 0-based atom indices that are SOMs
        y_sites = sorted(som_rows["atom_index"].unique().tolist())

        # Reaction class: most frequent non-No_reaction class
        rxn_counts = (som_rows["reaction_superclass"]
                      .value_counts()
                      .drop(labels=["No_reaction"], errors="ignore"))
        if rxn_counts.empty:
            # No labelled SOM rows — skip (molecule has only No_reaction entries)
            skipped += 1
            continue
        rxn_sc = rxn_counts.idxmax()

        try:
            rxn_label = int(le.transform([rxn_sc])[0])
        except Exception:
            skipped += 1
            continue

        # Ground-truth metabolite panels for imaging / reporting
        gt_metabolites = []
        valid_som_rows = som_rows.loc[
            (som_rows["Successor_SMILES"] != "") &
            (som_rows["Successor_SMILES"] != pred_smi) &
            (som_rows["reaction_superclass"] != "No_reaction")
        ].copy()

        if not valid_som_rows.empty:
            grouped_gt = (valid_som_rows
                          .groupby(["Successor_SMILES", "reaction_superclass"], sort=False)["atom_index"]
                          .apply(lambda s: sorted(set(int(v) for v in s.tolist())))
                          .reset_index())
            for _, row in grouped_gt.iterrows():
                gt_metabolites.append({
                    "successor_smiles": str(row["Successor_SMILES"]),
                    "ground_truth_som": row["atom_index"],
                    "ground_truth_reaction_superclass": str(row["reaction_superclass"]),
                })

        gt_metabolites_json = json.dumps(gt_metabolites, ensure_ascii=False)

        # Successor SMILES: first real metabolite from SOM rows
        succ_smis = [m["successor_smiles"] for m in gt_metabolites]
        succ_smi  = succ_smis[0] if succ_smis else ""

        # entry_id: index of the first SOM row (for image filenames)
        entry_id = int(grp.index[0])

        # Build graphs (same as before)
        sg = mol_to_som_graph(mol, y_sites)
        sg.successor_smiles = succ_smi
        sg.gt_metabolites_json = gt_metabolites_json
        sg.entry_id = entry_id

        rg = mol_to_rxn_graph(mol, rxn_label)
        rg.successor_smiles = succ_smi
        rg.gt_metabolites_json = gt_metabolites_json
        rg.entry_id = entry_id

        sg_list.append(sg)
        rg_list.append(rg)

        # Representative row for this molecule (used by augmentation + splits)
        repr_rows.append({
            "Predecessor_SMILES" : pred_smi,
            "Successor_SMILES"   : succ_smi,
            "reaction_superclass": rxn_sc,
            "y_sites"            : str(y_sites),
            "gt_metabolites_json": gt_metabolites_json,
            "entry_id"           : entry_id,
        })

    if skipped:
        print(f"  [warn] Skipped {skipped} predecessors "
              f"(invalid SMILES or no labelled SOM).")
    repr_df = pd.DataFrame(repr_rows).reset_index(drop=True)
    return sg_list, rg_list, le, repr_df


# ─────────────────────────────────────────────────────────────
#  Compute inverse-frequency class weights from a DataFrame
# ─────────────────────────────────────────────────────────────
def compute_class_weights(raw_df: pd.DataFrame,
                           le: LabelEncoder,
                           device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency weighting: w_c = N / (C * n_c)
    Returned as a tensor on `device` aligned with le.classes_ ordering.
    """
    counts = torch.zeros(len(le.classes_))
    for cls_name in le.classes_:
        idx = int(le.transform([cls_name])[0])
        counts[idx] = float((raw_df["reaction_superclass"] == cls_name).sum())
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (len(le.classes_) * counts)
    return weights.to(device)


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────
def drop_edge(edge_index: torch.Tensor,
              edge_attr:  Optional[torch.Tensor],
              p: float,
              training: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not training or p <= 0.0 or edge_index.size(1) == 0:
        return edge_index, edge_attr
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > p
    # Always keep at least one edge to avoid degenerate graphs
    if mask.sum() == 0:
        mask[0] = True
    ei   = edge_index[:, mask]
    ea   = edge_attr[mask] if edge_attr is not None else None
    return ei, ea


class GATNodeClassifier(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=3, heads=4,
                 dropout=0.35, edge_dim=EDGE_FEAT_DIM, drop_edge_p=0.15):
        super().__init__()
        self.dropout     = dropout
        self.drop_edge_p = drop_edge_p
        self.convs = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            self.convs.append(GATv2Conv(last, hidden, heads=heads,
                                        dropout=dropout, add_self_loops=True,
                                        edge_dim=edge_dim))
            last = hidden * heads
        self.lin = nn.Linear(last, 1)

    def forward(self, x, edge_index, edge_attr=None):
        # drop_edge only fires when self.training is True
        ei, ea = drop_edge(edge_index, edge_attr, self.drop_edge_p, self.training)
        for conv in self.convs:
            x = F.dropout(F.elu(conv(x, ei, edge_attr=ea)),
                          p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


class GATGraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=3, heads=4,
                 dropout=0.35, num_classes=12,
                 edge_dim=EDGE_FEAT_DIM, drop_edge_p=0.15):
        super().__init__()
        self.dropout     = dropout
        self.drop_edge_p = drop_edge_p
        self.convs = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            self.convs.append(GATv2Conv(last, hidden, heads=heads,
                                        dropout=dropout, add_self_loops=True,
                                        edge_dim=edge_dim))
            last = hidden * heads
        self.cls = nn.Sequential(
            nn.Linear(last, hidden), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes))

    def forward(self, x, edge_index, batch_vec, edge_attr=None):
        # drop_edge only fires when self.training is True
        ei, ea = drop_edge(edge_index, edge_attr, self.drop_edge_p, self.training)
        for conv in self.convs:
            x = F.dropout(F.elu(conv(x, ei, edge_attr=ea)),
                          p=self.dropout, training=self.training)
        return self.cls(global_mean_pool(x, batch_vec))


# ═════════════════════════════════════════════════════════════════
# SIMPLIFIED GNN BASELINES  
# ═════════════════════════════════════════════════════════════════
class SimpleGCNNodeClassifier(nn.Module):
    """Plain GCN node classifier for SOM prediction (simplified baseline)."""
    def __init__(self, in_dim, hidden=64, layers=2, heads=1,
                 dropout=0.35, edge_dim=EDGE_FEAT_DIM, drop_edge_p=0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            self.convs.append(GCNConv(last, hidden, add_self_loops=True))
            last = hidden
        self.lin = nn.Linear(last, 1)

    def forward(self, x, edge_index, edge_attr=None):
        # edge_attr intentionally ignored: GCNConv is not edge-conditioned
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)),
                          p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


class SimpleGCNGraphClassifier(nn.Module):
    """Plain GCN graph classifier for reaction class prediction."""
    def __init__(self, in_dim, hidden=64, layers=2, heads=1,
                 dropout=0.35, num_classes=12,
                 edge_dim=EDGE_FEAT_DIM, drop_edge_p=0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            self.convs.append(GCNConv(last, hidden, add_self_loops=True))
            last = hidden
        self.cls = nn.Linear(last, num_classes)

    def forward(self, x, edge_index, batch_vec, edge_attr=None):
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)),
                          p=self.dropout, training=self.training)
        return self.cls(global_mean_pool(x, batch_vec))


class GENReranker(nn.Module):
    def __init__(self, fp_dim: int = 2048, hidden: int = 256, dropout: float = 0.35):
        super().__init__()
        in_dim = 2 + fp_dim + 3   # som_prob + rxn_prob + ECFP4 + 3 diversity/rank features
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__(); self.alpha=alpha; self.gamma=gamma
    def forward(self, logits, targets):
        p = torch.sigmoid(logits).clamp(1e-8, 1-1e-8); t = targets.float()
        return (-self.alpha*(1-p)**self.gamma*torch.log(p)*t
                -(1-self.alpha)*p**self.gamma*torch.log(1-p)*(1-t)).mean()


class MultiClassFocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma  = gamma
        # Register as buffer so it moves to the correct device with .to(device)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # log-softmax for numerical stability
        log_p = F.log_softmax(logits, dim=-1)                      # (B, C)
        p     = log_p.exp()                                        # (B, C)
        # p_t: probability of the ground-truth class for each sample
        p_t   = p.gather(1, targets.view(-1, 1)).squeeze(1)       # (B,)
        # focal weight
        focal = (1.0 - p_t) ** self.gamma                         # (B,)
        # class weight for each sample's GT class
        if self.weight is not None:
            w = self.weight[targets]                               # (B,)
        else:
            w = torch.ones_like(p_t)
        # per-sample loss
        loss = -w * focal * p_t.log()
        return loss.mean()


# ═════════════════════════════════════════════════════════════════
# REACTION CHEMISTRY ENGINE  –  rule-gated, 12-class  (unchanged)
# ═════════════════════════════════════════════════════════════════
def _san(em):
    try: prod=em.GetMol(); Chem.SanitizeMol(prod); return prod
    except Exception: return None

def _other(bond, idx):
    return bond.GetBeginAtomIdx() if bond.GetEndAtomIdx()==idx else bond.GetEndAtomIdx()

def _has_h(a): return a.GetTotalNumHs()>0

def _dbl_partners(atom, mol, an):
    return [nb.GetIdx() for nb in atom.GetNeighbors()
            if nb.GetAtomicNum()==an and
               mol.GetBondBetweenAtoms(atom.GetIdx(),nb.GetIdx()).GetBondType()==Chem.BondType.DOUBLE]

def _add_oh(em, idx):
    o=em.AddAtom(Chem.Atom(8)); em.AddBond(idx,o,Chem.BondType.SINGLE)
    em.GetAtomWithIdx(o).SetNoImplicit(False)

def _heavy_deg(atom, mol):
    return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum()>1)


def _alicyclic_OH(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=6: return []
    if a.GetHybridization()!=Chem.HybridizationType.SP3: return []
    if not a.IsInRing() or a.GetIsAromatic(): return []
    if not _has_h(a): return []
    em=Chem.RWMol(Chem.Mol(mol)); _add_oh(em,idx); p=_san(em); return [p] if p else []

def _aliphatic_OH(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=6: return []
    if a.GetHybridization()!=Chem.HybridizationType.SP3: return []
    if a.IsInRing(): return []
    if not _has_h(a): return []
    em=Chem.RWMol(Chem.Mol(mol)); _add_oh(em,idx); p=_san(em); return [p] if p else []

def _aromatic_OH(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=6 or not a.GetIsAromatic() or not _has_h(a): return []
    em=Chem.RWMol(Chem.Mol(mol)); _add_oh(em,idx); p=_san(em); return [p] if p else []

def _c_oxidation(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=6: return []
    results=[]
    if _dbl_partners(a,mol,8) and _has_h(a):
        em=Chem.RWMol(Chem.Mol(mol)); _add_oh(em,idx); p=_san(em)
        if p: results.append(p)
    oh_nbs=[nb.GetIdx() for nb in a.GetNeighbors()
            if nb.GetAtomicNum()==8 and _has_h(nb) and
               mol.GetBondBetweenAtoms(idx,nb.GetIdx()).GetBondType()==Chem.BondType.SINGLE]
    for oi in oh_nbs:
        em=Chem.RWMol(Chem.Mol(mol))
        em.RemoveBond(idx,oi); em.AddBond(idx,oi,Chem.BondType.DOUBLE)
        em.GetAtomWithIdx(oi).SetNoImplicit(True); em.GetAtomWithIdx(oi).SetNumExplicitHs(0)
        p=_san(em)
        if p: results.append(p)
    if not results and a.GetHybridization()==Chem.HybridizationType.SP3 and _has_h(a):
        em=Chem.RWMol(Chem.Mol(mol))
        oi=em.AddAtom(Chem.Atom(8)); em.AddBond(idx,oi,Chem.BondType.DOUBLE)
        p=_san(em)
        if p: results.append(p)
    return results

def _dehydrogenation(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=6 or a.GetHybridization()!=Chem.HybridizationType.SP3 or not _has_h(a):
        return []
    for nb in a.GetNeighbors():
        if nb.GetAtomicNum() not in (6,7) or not _has_h(nb): continue
        bond=mol.GetBondBetweenAtoms(idx,nb.GetIdx())
        if bond is None or bond.GetBondType()!=Chem.BondType.SINGLE: continue
        em=Chem.RWMol(Chem.Mol(mol))
        em.RemoveBond(idx,nb.GetIdx()); em.AddBond(idx,nb.GetIdx(),Chem.BondType.DOUBLE)
        p=_san(em)
        if p: return [p]
    return []

def _epoxidation(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=6 or a.GetHybridization()!=Chem.HybridizationType.SP2 or a.GetIsAromatic():
        return []
    for bond in a.GetBonds():
        if bond.GetBondType()!=Chem.BondType.DOUBLE: continue
        oi=_other(bond,idx); ob=mol.GetAtomWithIdx(oi)
        if ob.GetAtomicNum()!=6 or ob.GetIsAromatic(): continue
        em=Chem.RWMol(Chem.Mol(mol))
        em.RemoveBond(idx,oi); em.AddBond(idx,oi,Chem.BondType.SINGLE)
        eo=em.AddAtom(Chem.Atom(8))
        em.AddBond(idx,eo,Chem.BondType.SINGLE); em.AddBond(oi,eo,Chem.BondType.SINGLE)
        p=_san(em)
        if p: return [p]
    return []

def _n_dealkylation(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=7 or a.GetIsAromatic(): return []
    alkyl=[nb for nb in a.GetNeighbors()
           if nb.GetAtomicNum()==6 and nb.GetHybridization()==Chem.HybridizationType.SP3 and
              mol.GetBondBetweenAtoms(idx,nb.GetIdx()).GetBondType()==Chem.BondType.SINGLE]
    if not alkyl: return []
    alkyl.sort(key=lambda x: x.GetDegree())
    results=[]; seen=set()
    for c in alkyl:
        em=Chem.RWMol(Chem.Mol(mol))
        em.RemoveBond(idx,c.GetIdx()); em.GetAtomWithIdx(idx).SetNoImplicit(False)
        p=_san(em)
        if p:
            smi=Chem.MolToSmiles(p,canonical=True)
            if smi not in seen: seen.add(smi); results.append(p)
    return results

def _n_hydroxylation(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=7 or not _has_h(a): return []
    em=Chem.RWMol(Chem.Mol(mol)); _add_oh(em,idx); p=_san(em); return [p] if p else []

def _n_oxidation(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=7 or _has_h(a): return []
    is_tert = (not a.GetIsAromatic() and
               all(nb.GetAtomicNum()==6 for nb in a.GetNeighbors()))
    if not (is_tert or a.GetIsAromatic()): return []
    em=Chem.RWMol(Chem.Mol(mol)); na=em.GetAtomWithIdx(idx)
    oi=em.AddAtom(Chem.Atom(8)); em.AddBond(idx,oi,Chem.BondType.SINGLE)
    em.GetAtomWithIdx(oi).SetFormalCharge(-1); na.SetFormalCharge(1)
    p=_san(em); return [p] if p else []

def _o_dealkylation(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=8 or _has_h(a): return []
    alkyl=[nb for nb in a.GetNeighbors()
           if nb.GetAtomicNum()==6 and nb.GetHybridization()==Chem.HybridizationType.SP3 and
              mol.GetBondBetweenAtoms(idx,nb.GetIdx()).GetBondType()==Chem.BondType.SINGLE]
    if not alkyl: return []
    alkyl.sort(key=lambda x: x.GetDegree())
    results=[]; seen=set()
    for c in alkyl:
        em=Chem.RWMol(Chem.Mol(mol))
        em.RemoveBond(idx,c.GetIdx()); em.GetAtomWithIdx(idx).SetNoImplicit(False)
        p=_san(em)
        if p:
            smi=Chem.MolToSmiles(p,canonical=True)
            if smi not in seen: seen.add(smi); results.append(p)
    return results

def _s_oxidation(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=16: return []
    if len(_dbl_partners(a,mol,8))>=2: return []
    em=Chem.RWMol(Chem.Mol(mol))
    oi=em.AddAtom(Chem.Atom(8)); em.AddBond(idx,oi,Chem.BondType.DOUBLE)
    p=_san(em); return [p] if p else []

def _terminal_OH(mol, idx):
    a=mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum()!=6: return []
    if a.GetHybridization()!=Chem.HybridizationType.SP3: return []
    if a.IsInRing() or not _has_h(a): return []
    if _heavy_deg(a,mol) not in (1,2): return []
    em=Chem.RWMol(Chem.Mol(mol)); _add_oh(em,idx); p=_san(em); return [p] if p else []


REACTION_TRANSFORMS: Dict[str, object] = {
    "alicyclic hydroxylation":            _alicyclic_OH,
    "aliphatic hydroxylation":            _aliphatic_OH,
    "aromatic hydroxylation":             _aromatic_OH,
    "c-oxidation":                        _c_oxidation,
    "dehydrogenation":                    _dehydrogenation,
    "epoxidation":                        _epoxidation,
    "n-dealkylation":                     _n_dealkylation,
    "n-hydroxylation":                    _n_hydroxylation,
    "n-oxidation":                        _n_oxidation,
    "o-dealkylation":                     _o_dealkylation,
    "s-oxidation":                        _s_oxidation,
    "terminal/penultimate hydroxylation": _terminal_OH,
}

ELIGIBILITY_NOTES: Dict[str, str] = {
    "alicyclic hydroxylation":            "sp3 C in non-aromatic ring + ≥1 H",
    "aliphatic hydroxylation":            "sp3 acyclic C (incl. benzylic/allylic) + ≥1 H",
    "aromatic hydroxylation":             "aromatic C + ≥1 H",
    "c-oxidation":                        "C: aldehyde+H→acid; C-OH→C=O; sp3 C+H→add=O",
    "dehydrogenation":                    "sp3 C+H adjacent to sp3/sp2 C or N with ≥1 H",
    "epoxidation":                        "non-aromatic sp2 C in C=C",
    "n-dealkylation":                     "aliphatic N + ≥1 sp3-C alkyl via single bond",
    "n-hydroxylation":                    "N (any) + ≥1 H → N-OH hydroxylamine",
    "n-oxidation":                        "tert. aliphatic N (no H, all-C nbrs) OR arom. N (no H) → N-oxide",
    "o-dealkylation":                     "ether O (no H) + ≥1 sp3-C alkyl via single bond",
    "s-oxidation":                        "S only: thioether→sulfoxide; sulfoxide→sulfone",
    "terminal/penultimate hydroxylation": "sp3 acyclic C + ≥1 H + heavy-degree 1 (ω) or 2 (ω-1)",
}

def apply_reaction(mol, idx, rxn_cls):
    key=rxn_cls.strip().lower()
    fn=REACTION_TRANSFORMS.get(key)
    if fn is None: return []
    if not (0<=idx<mol.GetNumAtoms()): return []
    return fn(mol, idx)  # type: ignore


# ─────────────────────────────────────────────────────────────
# Fingerprint utilities for similarity

FRAGMENT_POLICY = "all"        # "all" | "largest" | "none"
MIN_FRAGMENT_ATOMS = 3         # heavy-atom floor for a fragment to be a candidate


def set_fragment_policy(policy: str = "all", min_heavy: int = 3) -> None:
    global FRAGMENT_POLICY, MIN_FRAGMENT_ATOMS
    if policy not in ("all", "largest", "none"):
        raise ValueError(f"Unknown fragment policy: {policy}")
    FRAGMENT_POLICY = policy
    MIN_FRAGMENT_ATOMS = int(min_heavy)


def split_product_fragments(prod: Chem.Mol) -> List[Chem.Mol]:
    if prod is None:
        return []
    if FRAGMENT_POLICY == "none":
        return [prod]
    try:
        frags = Chem.GetMolFrags(prod, asMols=True, sanitizeFrags=True)
    except Exception:
        return [prod]
    if len(frags) <= 1:
        return [prod]
    if FRAGMENT_POLICY == "largest":
        return [max(frags, key=lambda m: m.GetNumHeavyAtoms())]
    keep = [f for f in frags if f.GetNumHeavyAtoms() >= MIN_FRAGMENT_ATOMS]
    if not keep:                       # everything below threshold: keep the biggest
        keep = [max(frags, key=lambda m: m.GetNumHeavyAtoms())]
    return sorted(keep, key=lambda m: -m.GetNumHeavyAtoms())


def count_multifragment_ground_truth(df: pd.DataFrame) -> Dict[str, int]:
    n_multi = n_total = 0
    for s in df.get("Successor_SMILES", pd.Series([], dtype=str)).astype(str):
        s = s.strip()
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        n_total += 1
        try:
            if len(Chem.GetMolFrags(m)) > 1:
                n_multi += 1
        except Exception:
            pass
    return {"n_ground_truth": n_total, "n_multifragment": n_multi}


def ecfp4(mol, n_bits=2048):
    try: return np.array(AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=n_bits),dtype=np.float32)
    except Exception: return np.zeros(n_bits,dtype=np.float32)

def tanimoto_fp(ma, mb):
    if ma is None or mb is None: return 0.0
    try:
        fa=AllChem.GetMorganFingerprintAsBitVect(ma,2,2048)
        fb=AllChem.GetMorganFingerprintAsBitVect(mb,2,2048)
        return float(DataStructs.TanimotoSimilarity(fa,fb))
    except Exception: return 0.0


# ─────────────────────────────────────────────────────────────
# Scaler / loader helpers
# ─────────────────────────────────────────────────────────────
def build_scaler(graphs):
    X=torch.cat([g.x for g in graphs],dim=0).cpu().numpy()
    return StandardScaler().fit(X)

def apply_scaler(graphs, scaler):
    for g in graphs:
        g.x=torch.tensor(scaler.transform(g.x.cpu().numpy()),dtype=torch.float)

def make_loader(graphs, bs, shuffle, seed, num_workers=0, pin_memory=False):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(graphs,
                      batch_size=bs,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      worker_init_fn=seed_worker,
                      generator=generator)

def scale_graphs(graphs, scaler):
    out=[]
    for g in graphs:
        g2=Data(x=torch.tensor(scaler.transform(g.x.cpu().numpy()),dtype=torch.float),
                edge_index=g.edge_index,
                edge_attr=getattr(g,"edge_attr",None),
                y=g.y)
        g2.smiles=g.smiles
        g2.smiles_mapped=getattr(g,"smiles_mapped",g.smiles)
        g2.successor_smiles=getattr(g,"successor_smiles","")
        g2.gt_metabolites_json=getattr(g,"gt_metabolites_json","[]")
        g2.entry_id=getattr(g,"entry_id",-1)
        out.append(g2)
    return out


# ─────────────────────────────────────────────────────────────
# Prediction collectors (pass edge_attr through)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_som_predictions(model, loader, device, top_k=5, use_amp: bool = False):
    model.eval(); results=[]
    amp_enabled = resolve_amp(device, use_amp)
    for batch in loader:
        batch=batch.to(device, non_blocking=(device.type=="cuda"))
        ea=batch.edge_attr if hasattr(batch,"edge_attr") else None
        with _autocast_ctx(device.type, amp_enabled):
            logits = model(batch.x,batch.edge_index,ea)
            probs = torch.sigmoid(logits)
        probs=detach_to_numpy(probs)
        bidx=detach_to_numpy(batch.batch)
        num_g=int(bidx.max())+1 if bidx.size>0 else 0
        y_all=detach_to_numpy(batch.y.view(-1))
        smi_l=batch.smiles if isinstance(batch.smiles,list) else [batch.smiles]
        map_l=batch.smiles_mapped if hasattr(batch,"smiles_mapped") else smi_l
        suc_l=batch.successor_smiles if hasattr(batch,"successor_smiles") else [""]*num_g
        gtm_l=batch.gt_metabolites_json if hasattr(batch,"gt_metabolites_json") else ["[]"]*num_g
        eid_l=batch.entry_id if hasattr(batch,"entry_id") else [-1]*num_g
        for g in range(num_g):
            mask=(bidx==g); pg=probs[mask]; yg=y_all[mask]
            gt=[int(i) for i,v in enumerate(yg) if v>=0.5]
            order=np.argsort(-pg)
            gt_json = gtm_l[g] if g < len(gtm_l) else "[]"
            try:
                gt_metabolites = json.loads(gt_json) if isinstance(gt_json, str) else []
            except Exception:
                gt_metabolites = []
            rec={"smiles":smi_l[g] if g<len(smi_l) else "",
                 "smiles_mapped":map_l[g] if g<len(map_l) else "",
                 "successor_smiles":suc_l[g] if g<len(suc_l) else "",
                 "gt_metabolites":gt_metabolites,
                 "entry_id":int(eid_l[g]) if g<len(eid_l) else -1,
                 "ground_truth_SOM":gt,"probs":pg,"order":order}
            for k in range(1,top_k+1):
                rec[f"top{k}_SOM"]=int(order[k-1]) if (k-1)<len(order) else None
            results.append(rec)
    return results


def collect_rxn_predictions(model, loader, device, le, top_k=5, use_amp: bool = False):
    model.eval(); results=[]
    amp_enabled = resolve_amp(device, use_amp)
    for batch in loader:
        batch=batch.to(device, non_blocking=(device.type=="cuda"))
        ea=batch.edge_attr if hasattr(batch,"edge_attr") else None
        with _autocast_ctx(device.type, amp_enabled):
            logits = model(batch.x,batch.edge_index,batch.batch,ea)
            probs = torch.softmax(logits,dim=-1)
        probs=detach_to_numpy(probs)
        trues=detach_to_numpy(batch.y.view(-1))
        smi_l=batch.smiles if isinstance(batch.smiles,list) else [batch.smiles]
        eid_l=batch.entry_id if hasattr(batch,"entry_id") else [-1]*len(trues)
        for i in range(len(trues)):
            order=np.argsort(-probs[i])
            gt_lbl=le.inverse_transform([int(trues[i])])[0]
            rec={"smiles":smi_l[i] if i<len(smi_l) else "","entry_id":int(eid_l[i]) if i<len(eid_l) else -1,
                 "ground_truth_rxn":gt_lbl,"probs":probs[i],"order":order}
            for k in range(1,top_k+1):
                rec[f"top{k}_rxn"]=le.inverse_transform([int(order[k-1])])[0] if (k-1)<len(order) else None
            results.append(rec)
    return results



# ─────────────────────────────────────────────────────────────
# GEN: generate + rerank candidates
# ─────────────────────────────────────────────────────────────
def generate_candidates(pred_smi, som_res, rxn_res, le, top_som=5, top_rxn=5):
    """
    Generate rule-based metabolite candidates and attach reranker supervision.

    IMPORTANT: ``tanimoto_gt`` is the maximum ECFP4 Tanimoto similarity to
    ANY annotated metabolite for the parent, not only the legacy
    ``successor_smiles`` field.  This keeps reranker training aligned with
    evaluation, which also credits any member of ``gt_metabolites``.

    ``successor_smiles`` is retained as a backward-compatible fallback for
    old datasets/results that do not carry the full ``gt_metabolites`` list.
    """
    mol=Chem.MolFromSmiles(pred_smi)
    if mol is None: return []
    som_atoms=[som_res[f"top{k}_SOM"] for k in range(1,top_som+1)
               if som_res.get(f"top{k}_SOM") is not None]
    rxn_labels=[rxn_res[f"top{k}_rxn"] for k in range(1,top_rxn+1)
                if rxn_res.get(f"top{k}_rxn") is not None]

    gt_smis: List[str] = []
    for meta in (som_res.get("gt_metabolites", []) or []):
        s = str(meta.get("successor_smiles", "")).strip()
        if s and s not in gt_smis:
            gt_smis.append(s)
    legacy_smi = str(som_res.get("successor_smiles", "")).strip()
    if legacy_smi and legacy_smi not in gt_smis:
        gt_smis.append(legacy_smi)

    gt_mols: List[Chem.Mol] = []
    for s in gt_smis:
        gm = Chem.MolFromSmiles(s)
        if gm is not None:
            gt_mols.append(gm)

    seen: Set[str]=set(); cands=[]
    for aidx in som_atoms:
        if aidx is None or not (0<=aidx<mol.GetNumAtoms()): continue
        sp=float(som_res["probs"][aidx]) if aidx<len(som_res["probs"]) else 0.0
        som_rank = list(som_res["order"]).index(aidx) if aidx in som_res["order"] else top_som
        for rxn in rxn_labels:
            if rxn is None or rxn.strip().lower() not in REACTION_TRANSFORMS: continue
            try: ri=int(le.transform([rxn])[0])
            except Exception: continue
            rp=float(rxn_res["probs"][ri]) if ri<len(rxn_res["probs"]) else 0.0
            rxn_rank = list(rxn_res["order"]).index(ri) if ri in rxn_res["order"] else top_rxn
            for prod in apply_reaction(mol,aidx,rxn):
                try:
                    pieces = split_product_fragments(prod)
                    n_frag = len(Chem.GetMolFrags(prod))
                except Exception:
                    pieces, n_frag = [prod], 1
                for frag in pieces:
                    try:
                        smi=Chem.MolToSmiles(frag,canonical=True)
                        if not smi or smi in seen: continue
                        seen.add(smi)
                        fp = ecfp4(frag)
                        cands.append({"candidate_smiles":smi,"atom_idx":aidx,"rxn_class":rxn,
                                      "som_prob":sp,"rxn_prob":rp,
                                      "som_rank":som_rank,"rxn_rank":rxn_rank,
                                      "tanimoto_gt":max((tanimoto_fp(frag, gm)
                                                          for gm in gt_mols),
                                                         default=0.0),
                                      "n_product_fragments":int(n_frag),
                                      "ecfp4_vec":fp})
                    except Exception: pass
    return cands


def build_reranker_features(cands: List[Dict], fp_dim: int = 2048) -> torch.Tensor:
    if not cands:
        return torch.zeros(0, 2 + fp_dim + 3)

    n = len(cands)
    # Pre-compute all ECFP4 vectors
    fps = np.stack([c["ecfp4_vec"] for c in cands], 0)  # (n, fp_dim)

    # Mean pairwise Tanimoto between each candidate and all others
    inter_tan = np.zeros(n, dtype=np.float32)
    if n > 1:
        dot = fps @ fps.T                             # (n, n)
        sum_fp = fps.sum(axis=1, keepdims=True)       # (n, 1)
        union = sum_fp + sum_fp.T - dot               # (n, n) element-wise
        union = np.clip(union, 1e-8, None)
        tan_mat = dot / union                         # (n, n) pairwise Tanimoto
        # exclude self (diagonal)
        np.fill_diagonal(tan_mat, 0.0)
        inter_tan = tan_mat.sum(axis=1) / (n - 1)    # mean excluding self

    # Normalised rank features (0 = rank 1 = best)
    max_rank = max(max(c["som_rank"] for c in cands), 1)
    som_rank_norm = np.array([c["som_rank"] / max_rank for c in cands], dtype=np.float32)
    rxn_rank_norm = np.array([c["rxn_rank"] / max_rank for c in cands], dtype=np.float32)

    rows = []
    for i, c in enumerate(cands):
        # tanimoto_gt intentionally omitted — not available at real inference time
        base = np.array([c["som_prob"], c["rxn_prob"]], dtype=np.float32)
        div  = np.array([inter_tan[i], som_rank_norm[i], rxn_rank_norm[i]], dtype=np.float32)
        rows.append(np.concatenate([base, c["ecfp4_vec"], div]))
    return torch.tensor(np.stack(rows, 0), dtype=torch.float)


def rerank_candidates(reranker: GENReranker,
                       cands: List[Dict],
                       device: torch.device,
                       use_amp: bool = False) -> List[Dict]:
    if not cands: return []
    feats = build_reranker_features(cands).to(device)
    amp_enabled = resolve_amp(device, use_amp)
    reranker.eval()
    with torch.no_grad():
        with _autocast_ctx(device.type, amp_enabled):
            scores = reranker(feats)
        scores = detach_to_numpy(scores)
    for i, c in enumerate(cands):
        c["reranker_score"] = float(scores[i])
    return sorted(cands, key=lambda x: -x["reranker_score"])


def reranker_loss(reranker: GENReranker,
                   som_pv: torch.Tensor,
                   rxn_pv: torch.Tensor,
                   tan_v:  torch.Tensor,
                   fp_m:   torch.Tensor,
                   inter_tan_v: torch.Tensor,
                   som_rank_v:  torch.Tensor,
                   rxn_rank_v:  torch.Tensor,
                   device: torch.device,
                   diversity_weight: float = 0.15) -> torch.Tensor:
    if tan_v.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    feats = torch.cat([
        som_pv.unsqueeze(1).detach(),        # col 0  — detached: no grad to SOM model
        rxn_pv.unsqueeze(1).detach(),        # col 1  — detached: no grad to Rxn model
        fp_m.detach(),                       # cols 2–2049
        inter_tan_v.unsqueeze(1).detach(),   # col 2050
        som_rank_v.unsqueeze(1).detach(),    # col 2051
        rxn_rank_v.unsqueeze(1).detach(),    # col 2052
    ], dim=1).to(device)

    scores = reranker(feats)

    # --- Standard listwise term ---
    lp     = F.log_softmax(scores, dim=0)
    target = F.softmax(tan_v * 10.0, dim=0).detach()
    loss_rank = -(target * lp).sum()

    # --- Diversity bonus term 
    diversity_bonus = F.softmax((1.0 - inter_tan_v) * 5.0, dim=0).detach()
    loss_div = -(diversity_bonus * lp).sum()

    return loss_rank + diversity_weight * loss_div


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def calibrate_threshold(y_true, y_prob):
    bt,bf=0.5,-1.0
    for t in [i/100.0 for i in range(5,96)]:
        f=sk_f1(y_true,(y_prob>=t).astype(int),zero_division=0)
        if f>bf: bf,bt=f,t
    return bt

@torch.no_grad()
def som_ranking_metrics(model, loader, device, criterion=None, use_amp: bool = False):
    model.eval(); tl=tn=t1=t3=t5=0; rr=mol_n=0
    amp_enabled = resolve_amp(device, use_amp)
    for batch in loader:
        batch=batch.to(device, non_blocking=(device.type=="cuda"))
        ea=batch.edge_attr if hasattr(batch,"edge_attr") else None
        with _autocast_ctx(device.type, amp_enabled):
            logits=model(batch.x,batch.edge_index,ea)
            probs=torch.sigmoid(logits)
            if criterion is not None:
                loss = criterion(logits,batch.y.view(-1))
            else:
                loss = None
        if loss is not None:
            tl+=float(loss.item())*batch.num_nodes; tn+=batch.num_nodes
        bidx=batch.batch
        num_g=int(bidx.max().item())+1 if batch.num_graphs > 0 else 0
        y_all=detach_to_numpy(batch.y.view(-1)).astype(int); p_all=detach_to_numpy(probs)
        bidx_np=detach_to_numpy(bidx).astype(int)
        for g in range(num_g):
            mask=(bidx_np==g)
            yg=y_all[mask]; pg=p_all[mask]
            if yg.sum()==0: continue
            order=np.argsort(-pg)
            t1+=int(yg[order[:1]].max()==1)
            t3+=int(yg[order[:min(3,len(pg))]].max()==1)
            t5+=int(yg[order[:min(5,len(pg))]].max()==1)
            ranks=np.empty_like(order); ranks[order]=np.arange(1,len(order)+1)
            rr+=1.0/float(np.min(ranks[np.where(yg==1)[0]])); mol_n+=1
    al=tl/tn if tn else float("nan"); n=mol_n if mol_n else 1
    return {"loss":al,"top1":t1/n,"top3":t3/n,"top5":t5/n,"mrr":rr/n}

@torch.no_grad()
def rxn_evaluate(model, loader, criterion, device, le, use_amp: bool = False):
    model.eval(); tl=tn=0; ap,at=[],[]
    amp_enabled = resolve_amp(device, use_amp)
    for batch in loader:
        batch=batch.to(device, non_blocking=(device.type=="cuda"))
        ea=batch.edge_attr if hasattr(batch,"edge_attr") else None
        with _autocast_ctx(device.type, amp_enabled):
            logits=model(batch.x,batch.edge_index,batch.batch,ea)
            y=batch.y.view(-1)
            loss = criterion(logits,y)
        tl+=float(loss.item())*y.size(0); tn+=y.size(0)
        ap.extend(detach_to_numpy(logits.argmax(1)).tolist())
        at.extend(detach_to_numpy(y).tolist())
    ap=np.array(ap); at=np.array(at)
    # weighted overall F1
    f1_weighted = float(sk_f1(at, ap, average="weighted", zero_division=0))
    # per-class F1 (array of length num_classes)
    f1_per_class = sk_f1(at, ap, average=None, zero_division=0,
                         labels=list(range(len(le.classes_))))
    per_class_dict = {le.classes_[i]: float(f1_per_class[i])
                      for i in range(len(le.classes_))}
    return {"loss":   tl/tn if tn else float("nan"),
            "acc":    float((ap==at).mean()),
            "f1":     f1_weighted,
            "f1_per": per_class_dict}


# ─────────────────────────────────────────────────────────────
#  validation metric
# ─────────────────────────────────────────────────────────────
def val_reranked_metrics(som_model, rxn_model, reranker,
                          val_som_loader, val_rxn_loader,
                          device, le,
                          top_som: int = 5,
                          top_rxn: int = 5,
                          tanimoto_threshold: float = 0.95,
                          use_amp: bool = False) -> Dict:
    """
    Run the FULL pipeline on the validation set and return end-to-end metrics.

    Pipeline per molecule
    ---------------------
    SOM model  → atom probability scores
    Rxn model  → reaction class probability scores
    generate_candidates → candidate SMILES pool (may be empty)
    reranker   → sorted ranked list
    Compare rank-1 … rank-k SMILES against every ground-truth successor for
    that molecule: a hit is declared when either
      (a) canonical SMILES match exactly, OR
      (b) Tanimoto(ECFP4) ≥ tanimoto_threshold

    Molecules with an empty candidate pool count as rank ∞:
      MRR contribution = 0, top-k miss.
    This ensures the metric is not inflated by skipping hard cases.

    Returns
    -------
    dict with keys:
      top1        fraction of molecules where rank-1 candidate is a hit
      mrr         mean reciprocal rank of first hit (0 for empty pools)
      joint       0.6 * top1 + 0.4 * mrr   (scalar used for checkpointing)
      n_molecules total validation molecules evaluated
      n_empty     molecules with empty candidate pool (counted as rank ∞)
    """
    som_model.eval(); rxn_model.eval(); reranker.eval()

    som_res = collect_som_predictions(som_model, val_som_loader, device, use_amp=use_amp)
    rxn_res = collect_rxn_predictions(rxn_model, val_rxn_loader, device, le, use_amp=use_amp)

    top1_hits = 0; mrr_sum = 0.0; n_mol = 0; n_empty = 0

    for sr, rr_res in zip(som_res, rxn_res):
        # Ground-truth successor SMILES for this molecule (may be multiple)
        gt_smiles_raw: List[str] = []
        gt_metabolites = sr.get("gt_metabolites", []) or []
        for meta in gt_metabolites:
            s = str(meta.get("successor_smiles", "")).strip()
            if s:
                gt_smiles_raw.append(s)
        # Also include the legacy single successor_smiles field as fallback
        legacy = str(sr.get("successor_smiles", "")).strip()
        if legacy and legacy not in gt_smiles_raw:
            gt_smiles_raw.append(legacy)

        if not gt_smiles_raw:
            # No ground truth recorded — skip (can't evaluate)
            continue

        n_mol += 1

        # Canonical SMILES and fingerprints of all GT metabolites
        gt_canon: List[str] = []
        gt_fps = []
        for s in gt_smiles_raw:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            gt_canon.append(Chem.MolToSmiles(mol, canonical=True))
            gt_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

        if not gt_canon:
            n_mol -= 1   # no valid GT mol — skip
            continue

        # Generate and rerank candidates
        cands = generate_candidates(sr["smiles"], sr, rr_res, le, top_som, top_rxn)

        if not cands:
            # Empty pool → rank ∞: MRR=0, top-k miss
            n_empty += 1
            continue

        ranked = rerank_candidates(reranker, cands, device, use_amp=use_amp)

        # Find rank of first hit in the ranked list
        first_hit_rank: Optional[int] = None
        for rank, cand in enumerate(ranked, start=1):
            csmi = str(cand.get("candidate_smiles", "")).strip()
            if not csmi:
                continue
            cmol = Chem.MolFromSmiles(csmi)
            if cmol is None:
                continue
            ccanon = Chem.MolToSmiles(cmol, canonical=True)
            # (a) exact canonical match
            if ccanon in gt_canon:
                first_hit_rank = rank
                break
            # (b) Tanimoto ≥ threshold against any GT
            cfp = AllChem.GetMorganFingerprintAsBitVect(cmol, 2, 2048)
            if any(float(DataStructs.TanimotoSimilarity(cfp, gfp)) >= tanimoto_threshold
                   for gfp in gt_fps):
                first_hit_rank = rank
                break

        if first_hit_rank is not None:
            mrr_sum += 1.0 / first_hit_rank
            if first_hit_rank == 1:
                top1_hits += 1
        # else: no hit in ranked list → MRR contribution 0, top-1 miss

    n = n_mol if n_mol > 0 else 1
    top1 = top1_hits / n
    mrr  = mrr_sum   / n
    joint = 0.6 * top1 + 0.4 * mrr
    return {"top1": top1, "mrr": mrr, "joint": joint,
            "n_molecules": n_mol, "n_empty": n_empty}


# ─────────────────────────────────────────────────────────────
# CSV savers
# ─────────────────────────────────────────────────────────────
def save_som_csv(results, path, top_k=5):
    rows=[{"Predecessor_SMILES":r["smiles"],"ground_truth_SOM":str(r["ground_truth_SOM"]),
           **{f"top{k}_SOM":r.get(f"top{k}_SOM") for k in range(1,top_k+1)}}
          for r in results]
    pd.DataFrame(rows).to_csv(path,index=False)

def save_rxn_csv(results, path, top_k=5):
    rows=[{"Predecessor_SMILES":r["smiles"],"ground_truth_reaction_class":r["ground_truth_rxn"],
           **{f"top{k}_rxn":r.get(f"top{k}_rxn") for k in range(1,top_k+1)}}
          for r in results]
    pd.DataFrame(rows).to_csv(path,index=False)

def save_metabolites_csv(all_cands, path, top_n=5):
    wide=[]; long=[]
    for pred_smi,succ_smi,cands in all_cands:
        top=cands[:top_n]
        w={"Predecessor_SMILES":pred_smi,"Successor_SMILES_GT":succ_smi}
        for rank in range(1,top_n+1):
            c=top[rank-1] if (rank-1)<len(top) else None
            w[f"rank{rank}_smiles"]        =c["candidate_smiles"] if c else ""
            w[f"rank{rank}_rxn_class"]     =c["rxn_class"]        if c else ""
            w[f"rank{rank}_SOM_atom"]      =c["atom_idx"]         if c else ""
            w[f"rank{rank}_tanimoto_GT"]   =c["tanimoto_gt"]      if c else ""
            w[f"rank{rank}_reranker_score"]=c.get("reranker_score","") if c else ""
        wide.append(w)
        for rank,c in enumerate(top,1):
            long.append({"Predecessor_SMILES":pred_smi,"Successor_SMILES_GT":succ_smi,
                         "rank":rank,"candidate_smiles":c["candidate_smiles"],
                         "rxn_class":c["rxn_class"],"SOM_atom_idx":c["atom_idx"],
                         "tanimoto_to_GT":c["tanimoto_gt"],
                         "reranker_score":c.get("reranker_score",float("nan"))})
    pd.DataFrame(wide).to_csv(path,index=False)
    pd.DataFrame(long).to_csv(path.replace(".csv","_long.csv"),index=False)


# ─────────────────────────────────────────────────────────────
# Image rendering
# ─────────────────────────────────────────────────────────────
_ORANGE    = (1.0, 0.55, 0.0)   # RDKit float RGB for GT-SOM highlight
_LEGEND_H  = 52                  # pixel height of caption strip
_LEGEND_BG = (248, 248, 248)     # caption strip background


def _prep_mol_indexed(mol: Chem.Mol) -> Chem.Mol:
    """
    Return a copy with:
      - atom-map numbers set to atom index  (displayed as index labels)
      - 2D coordinates computed
    Used ONLY for the Predecessor panel.
    """
    m = Chem.RWMol(Chem.Mol(mol))
    for a in m.GetAtoms():
        a.SetAtomMapNum(a.GetIdx())
    try:
        AllChem.Compute2DCoords(m)
    except Exception:
        pass
    return m.GetMol()


def _prep_mol_clean(mol: Chem.Mol) -> Chem.Mol:
    """
    Return a copy with:
      - ALL atom-map numbers cleared (no index labels rendered)
      - 2D coordinates computed
    Used for Successor and all generated-metabolite panels.
    """
    m = Chem.RWMol(Chem.Mol(mol))
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    try:
        AllChem.Compute2DCoords(m)
    except Exception:
        pass
    return m.GetMol()


def _render_mol_cairo(mol: Chem.Mol, w: int, h: int,
                       hl_atoms: List[int],
                       hl_bonds: List[int]) -> Optional["Image.Image"]:
    """
    Render mol (w×h) via MolDraw2DCairo.
    hl_atoms / hl_bonds are coloured orange when non-empty.
    addAtomIndices=False always; atom-map numbers (set on the mol beforehand)
    are what produce the index labels on the predecessor.
    Returns None on failure.
    """
    if not _PIL_OK:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
        opts = drawer.drawOptions()
        opts.addAtomIndices      = False   # indices come from atom-map numbers
        opts.addStereoAnnotation = True
        opts.padding             = 0.12

        ha = list(hl_atoms) if hl_atoms else []
        hb = list(hl_bonds) if hl_bonds else []
        ac = {a: _ORANGE for a in ha}
        bc = {b: _ORANGE for b in hb}

        rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer.DrawMolecule(mol,
                            highlightAtoms=ha,
                            highlightAtomColors=ac if ha else {},
                            highlightBonds=hb,
                            highlightBondColors=bc if hb else {})
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    except Exception:
        return None


def _add_legend_strip(mol_img: "Image.Image", lines: List[str]) -> "Image.Image":
    """Append a text caption strip below mol_img with dynamic height."""
    if not _PIL_OK:
        return mol_img
    wrapped_lines: List[str] = []
    for line in lines:
        txt = "" if line is None else str(line)
        if len(txt) <= 42:
            wrapped_lines.append(txt)
        else:
            start = 0
            while start < len(txt):
                wrapped_lines.append(txt[start:start+42])
                start += 42
    wrapped_lines = wrapped_lines[:8]
    line_h = 16
    pad_y  = 4
    strip_h = max(_LEGEND_H, pad_y * 2 + line_h * max(1, len(wrapped_lines)))
    w     = mol_img.width
    strip = Image.new("RGB", (w, strip_h), _LEGEND_BG)
    draw  = PILDraw.Draw(strip)
    try:   font = ImageFont.load_default()
    except Exception: font = None
    y = pad_y
    for line in wrapped_lines:
        draw.text((6, y), line, fill=(30, 30, 30), font=font)
        y += line_h
    out = Image.new("RGB", (w, mol_img.height + strip_h), (255, 255, 255))
    out.paste(mol_img, (0, 0))
    out.paste(strip,   (0, mol_img.height))
    return out


def _mol_panel(mol: Chem.Mol, w: int, h: int,
               hl_atoms: List[int], hl_bonds: List[int],
               legend_lines: List[str]) -> "Image.Image":
    """Full panel = rendered mol + caption strip. Falls back to grey on error."""
    if not _PIL_OK:
        return Image.new("RGB", (w, h + _LEGEND_H), (255, 255, 255))
    img = _render_mol_cairo(mol, w, h, hl_atoms, hl_bonds)
    if img is None:
        img = Image.new("RGB", (w, h), (245, 245, 245))
        if _PIL_OK:
            d = PILDraw.Draw(img)
            try:   font = ImageFont.load_default()
            except Exception: font = None
            d.text((8, h // 2 - 8), "render error", fill=(180, 100, 100), font=font)
    return _add_legend_strip(img, legend_lines)


def _blank_panel(w: int, h: int, text: str = "") -> "Image.Image":
    total_h = h + _LEGEND_H
    img = Image.new("RGB", (w, total_h), (255, 255, 255))
    if text and _PIL_OK:
        d = PILDraw.Draw(img)
        try:   font = ImageFont.load_default()
        except Exception: font = None
        d.text((6, h // 2 - 8), text, fill=(160, 160, 160), font=font)
    return img


def _hstack(imgs: List["Image.Image"], pad: int = 6) -> "Image.Image":
    if not imgs: return Image.new("RGB", (1, 1), (255, 255, 255))
    W = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    H = max(im.height for im in imgs)
    c = Image.new("RGB", (W, H), (255, 255, 255)); x = 0
    for im in imgs: c.paste(im, (x, 0)); x += im.width + pad
    return c


def _vstack(imgs: List["Image.Image"], pad: int = 0) -> "Image.Image":
    if not imgs: return Image.new("RGB", (1, 1), (255, 255, 255))
    W = max(im.width for im in imgs)
    H = sum(im.height for im in imgs) + pad * (len(imgs) - 1)
    c = Image.new("RGB", (W, H), (255, 255, 255)); y = 0
    for im in imgs: c.paste(im, (0, y)); y += im.height + pad
    return c


def _row_label(img: "Image.Image", label: str, lw: int = 72) -> "Image.Image":
    if not _PIL_OK: return img
    h   = img.height
    out = Image.new("RGB", (lw + img.width, h), (220, 220, 220))
    txt = Image.new("RGB", (h, lw), (200, 200, 200))
    td  = PILDraw.Draw(txt)
    try:   font = ImageFont.load_default()
    except Exception: font = None
    td.text((max(4, h // 2 - len(label) * 3), 6), label, fill=(40, 40, 40), font=font)
    out.paste(txt.rotate(90, expand=True), (0, 0))
    out.paste(img, (lw, 0))
    return out


def _pad_to_width(img: "Image.Image", target_w: int) -> "Image.Image":
    if img.width >= target_w: return img
    out = Image.new("RGB", (target_w, img.height), (255, 255, 255))
    out.paste(img, (0, 0))
    return out


def save_compound_image(entry_id: int,
                         som_res: Dict,
                         ranked_cands: List[Dict],
                         img_dir: str,
                         sw: int = 350,
                         sh: int = 260,
                         top_n: int = 5):
    """
    Three-row PNG per molecule.

    ROW 1 – Parent
      Single predecessor panel with atom indices visible.
      Legend lines:
        Parent
        GT SOM : [...]

    ROW 2 – Ground-truth successors
      One panel per GT metabolite.
      Legend lines:
        Metabolite 1 / Metabolite 2 / ...
        GT SOM : [...]
        GT reaction : ...

    ROW 3 – Predicted/generated metabolites
      One panel per generated candidate (top_n).
      Legend lines:
        Pred 1 / Pred 2 / ...
        Pred SOM : [...]
        Pred reaction: ...
        Tanimoto : ...
    """
    if not _PIL_OK:
        return
    ensure_dir(img_dir)

    # ── Parent row ──────────────────────────────────────────────
    try:
        mp_raw = mol_from_mapped_smiles(som_res.get("smiles_mapped", som_res["smiles"]))
    except Exception:
        mp_raw = Chem.MolFromSmiles(som_res["smiles"])
    if mp_raw is None:
        return

    mp  = _prep_mol_indexed(mp_raw)
    gt_parent = som_res.get("ground_truth_SOM", [])
    parent_panel = _mol_panel(
        mp, sw, sh,
        hl_atoms=gt_parent,
        hl_bonds=bonds_touching(mp, gt_parent),
        legend_lines=[
            "Parent",
            f"GT SOM : {gt_parent}",
        ],
    )
    row1 = _row_label(parent_panel, "Parent")

    # ── Ground-truth metabolite row ─────────────────────────────
    gt_metabolites = som_res.get("gt_metabolites", []) or []
    gt_imgs: List["Image.Image"] = []
    for i, meta in enumerate(gt_metabolites, start=1):
        succ_smi = str(meta.get("successor_smiles", ""))
        gt_som   = meta.get("ground_truth_som", [])
        gt_rxn   = str(meta.get("ground_truth_reaction_superclass", ""))
        ms_raw   = Chem.MolFromSmiles(succ_smi) if succ_smi else None
        legend   = [
            f"Metabolite {i}",
            f"GT SOM : {gt_som}",
            f"GT reaction: {gt_rxn}",
        ]
        if ms_raw is not None:
            ms = _prep_mol_clean(ms_raw)
            gt_imgs.append(_mol_panel(ms, sw, sh, [], [], legend))
        else:
            gt_imgs.append(_blank_panel(sw, sh, "Ground-truth metabolite render error"))

    if not gt_imgs:
        gt_imgs = [_blank_panel(sw, sh, "No ground-truth metabolite recorded")]
    row2 = _row_label(_hstack(gt_imgs), "Actual Metabolites")

    # ── Predicted/generated metabolite row ──────────────────────
    pred_imgs: List["Image.Image"] = []
    for i, cand in enumerate(ranked_cands[:top_n], start=1):
        csmi   = str(cand.get("candidate_smiles", ""))
        mc_raw = Chem.MolFromSmiles(csmi) if csmi else None
        pred_som = [cand.get("atom_idx")] if cand.get("atom_idx", None) is not None else []
        pred_rxn = str(cand.get("rxn_class", ""))
        tan      = cand.get("tanimoto_gt", float("nan"))
        legend   = [
            f"Pred {i}",
            f"Pred SOM : {pred_som}",
            f"Pred reaction: {pred_rxn}",
            f"Tanimoto : {tan:.4f}" if isinstance(tan, (int, float, np.floating)) and not math.isnan(float(tan)) else "Tanimoto : NA",
        ]
        if mc_raw is not None:
            mc = _prep_mol_clean(mc_raw)
            pred_imgs.append(_mol_panel(mc, sw, sh, [], [], legend))
        else:
            pred_imgs.append(_blank_panel(sw, sh, "Predicted metabolite render error"))

    while len(pred_imgs) < top_n:
        pred_imgs.append(_blank_panel(sw, sh, "—"))
    row3 = _row_label(_hstack(pred_imgs), "Predicted Metabolites")

    # ── Combine rows ────────────────────────────────────────────
    mw   = max(row1.width, row2.width, row3.width)
    row1 = _pad_to_width(row1, mw)
    row2 = _pad_to_width(row2, mw)
    row3 = _pad_to_width(row3, mw)
    div1 = Image.new("RGB", (mw, 6), (160, 160, 160))
    div2 = Image.new("RGB", (mw, 6), (160, 160, 160))
    final = _vstack([row1, div1, row2, div2, row3])

    try:
        final.save(os.path.join(img_dir, f"Entry_{entry_id}.png"))
    except Exception as e:
        print(f"  [warn] Image Entry_{entry_id}: {e}")


def save_all_images(som_results: List[Dict],
                     ranked_cands_list: List[List[Dict]],
                     img_dir: str,
                     sw: int = 350,
                     sh: int = 260,
                     top_n: int = 5):
    ensure_dir(img_dir)
    if not _PIL_OK:
        print("  [warn] Pillow not found – images skipped.")
        return
    for sr, cands in zip(som_results, ranked_cands_list):
        try:
            save_compound_image(sr.get("entry_id", -1), sr, cands,
                                img_dir, sw, sh, top_n)
        except Exception as e:
            print(f"  [warn] Image Entry_{sr.get('entry_id', -1)}: {e}")


# ─────────────────────────────────────────────────────────────
# Train one fold  
# ─────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════
# VARIANT CONFIGURATION
# ═════════════════════════════════════════════════════════════════
@dataclass
class VariantConfig:
    """
    Declarative description of one ablation variant.

    Every variant shares the SAME data loading, scaffold test split, scaffold
    5-fold CV split, reaction rule engine, evaluation code and output layout.
    Only the flags below differ, so any performance gap between two variants is
    attributable to the flag that changed.
    """
    name: str
    description: str
    gnn: str = "gatv2"                 # "gatv2" (full) | "simple_gcn"
    feature_mode: str = "full"         # "full" (19-d) | "no_engineered" (10-d)
    train_som: bool = True             # train the SOM head
    train_rxn: bool = True             # train the reaction head
    use_reranker: bool = True          # learned reranker (False -> heuristic ranking)
    use_diversity: bool = True         # diversity-aware term in the ranking loss
    generate_metabolites: bool = True  # run the rule engine + ranking stage
    selection_metric: str = "joint"    # "joint" | "som_mrr" | "rxn_f1"

    def check(self):
        if self.gnn not in ("gatv2", "simple_gcn"):
            raise ValueError(f"{self.name}: bad gnn '{self.gnn}'")
        if self.feature_mode not in ("full", "no_engineered"):
            raise ValueError(f"{self.name}: bad feature_mode '{self.feature_mode}'")
        if self.selection_metric not in ("joint", "som_mrr", "rxn_f1"):
            raise ValueError(f"{self.name}: bad selection_metric '{self.selection_metric}'")
        if self.selection_metric == "joint" and not (self.train_som and self.train_rxn):
            raise ValueError(f"{self.name}: 'joint' selection needs both heads")
        if self.selection_metric == "som_mrr" and not self.train_som:
            raise ValueError(f"{self.name}: 'som_mrr' selection needs the SOM head")
        if self.selection_metric == "rxn_f1" and not self.train_rxn:
            raise ValueError(f"{self.name}: 'rxn_f1' selection needs the reaction head")
        if self.generate_metabolites and not (self.train_som and self.train_rxn):
            raise ValueError(f"{self.name}: metabolite generation needs both heads")
        return self


# ═════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═════════════════════════════════════════════════════════════════
def build_som_model(variant: VariantConfig, in_dim: int, args, device,
                    drop_edge_p: Optional[float] = None) -> nn.Module:
    dep = args.drop_edge_p if drop_edge_p is None else drop_edge_p
    if variant.gnn == "simple_gcn":
        m = SimpleGCNNodeClassifier(in_dim, args.simple_hidden, args.simple_layers,
                                    dropout=args.dropout)
    else:
        m = GATNodeClassifier(in_dim, args.hidden_som, args.layers, args.heads,
                              dropout=args.dropout, drop_edge_p=dep)
    return m.to(device)


def build_rxn_model(variant: VariantConfig, in_dim: int, num_classes: int, args, device,
                    drop_edge_p: Optional[float] = None) -> nn.Module:
    dep = args.drop_edge_p if drop_edge_p is None else drop_edge_p
    if variant.gnn == "simple_gcn":
        m = SimpleGCNGraphClassifier(in_dim, args.simple_hidden, args.simple_layers,
                                     dropout=args.dropout, num_classes=num_classes)
    else:
        m = GATGraphClassifier(in_dim, args.hidden_rxn, args.layers, args.heads,
                               dropout=args.dropout, num_classes=num_classes,
                               drop_edge_p=dep)
    return m.to(device)


def build_reranker(args, device) -> GENReranker:
    return GENReranker(2048, args.hidden_rer, args.dropout).to(device)


# ═════════════════════════════════════════════════════════════════
# HEURISTIC RANKING  (used when the learned reranker is disabled)
# ═════════════════════════════════════════════════════════════════
def rank_candidates_heuristic(cands: List[Dict]) -> List[Dict]:
    """
    Rank the candidate pool WITHOUT the learned reranker (ablation R1-3,
    "a model without the reranker").

    Score = P(SOM atom) x P(reaction class), i.e. the pipeline's own upstream
    confidence, which is the natural no-reranker control: the candidate pool
    and the rule engine are unchanged, only the learned ordering is removed.
    """
    if not cands:
        return []
    for c in cands:
        c["reranker_score"] = float(c["som_prob"]) * float(c["rxn_prob"])
    return sorted(cands, key=lambda x: (-x["reranker_score"], x["som_rank"], x["rxn_rank"]))


def rank_candidates(reranker: Optional[GENReranker],
                    cands: List[Dict],
                    device: torch.device,
                    use_reranker: bool,
                    use_amp: bool = False) -> List[Dict]:
    """Single entry point for ranking, honouring the variant's reranker flag."""
    if not cands:
        return []
    if use_reranker and reranker is not None:
        return rerank_candidates(reranker, cands, device, use_amp=use_amp)
    return rank_candidates_heuristic(cands)


# ═════════════════════════════════════════════════════════════════
# END-TO-END METABOLITE METRICS  (variant-aware wrapper of P7 metric)
# ═════════════════════════════════════════════════════════════════
def _gt_smiles_for(sr: Dict) -> List[str]:
    out: List[str] = []
    for meta in (sr.get("gt_metabolites", []) or []):
        s = str(meta.get("successor_smiles", "")).strip()
        if s and s not in out:
            out.append(s)
    legacy = str(sr.get("successor_smiles", "")).strip()
    if legacy and legacy not in out:
        out.append(legacy)
    return out


def reranked_metrics_from_predictions(som_res: List[Dict],
                                      rxn_res: List[Dict],
                                      reranker: Optional[GENReranker],
                                      device, le,
                                      top_som: int = 5,
                                      top_rxn: int = 5,
                                      tanimoto_threshold: float = 0.95,
                                      use_reranker: bool = True,
                                      use_amp: bool = False,
                                      collect_rows: bool = False) -> Dict:
    """
    End-to-end metabolite metrics computed from ALREADY-COLLECTED SOM and
    reaction predictions (so it can be reused for single models and ensembles).

    Reports exact-match and similarity-match accuracy SEPARATELY, which is what
    Reviewer 1 asked for in comment 9, plus the combined 'hit' definition used
    for checkpointing (exact OR Tanimoto >= threshold).

    Molecules with an empty candidate pool count as rank infinity (MRR 0, top-k miss).
    """
    top1 = top3 = top5 = 0
    exact1 = sim_only1 = 0
    mrr_sum = 0.0
    n_mol = 0
    n_empty = 0
    n_pool_hit = 0                  # correct metabolite present anywhere in the pool
    pool_sizes: List[int] = []
    rows: List[Dict] = []

    for sr, rr_res in zip(som_res, rxn_res):
        gt_raw = _gt_smiles_for(sr)
        if not gt_raw:
            continue

        gt_canon, gt_fps = [], []
        for s in gt_raw:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            gt_canon.append(Chem.MolToSmiles(m, canonical=True))
            gt_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
        if not gt_canon:
            continue

        n_mol += 1
        cands = generate_candidates(sr["smiles"], sr, rr_res, le, top_som, top_rxn)
        pool_sizes.append(len(cands))

        if not cands:
            n_empty += 1
            if collect_rows:
                rows.append({"Predecessor_SMILES": sr["smiles"], "n_candidates": 0,
                             "first_hit_rank": "", "hit_type": "empty_pool",
                             "best_tanimoto_in_pool": 0.0})
            continue

        ranked = rank_candidates(reranker, cands, device, use_reranker, use_amp=use_amp)

        first_hit_rank: Optional[int] = None
        first_hit_type = ""
        best_tan = 0.0
        pool_hit = False

        for rank, cand in enumerate(ranked, start=1):
            csmi = str(cand.get("candidate_smiles", "")).strip()
            if not csmi:
                continue
            cmol = Chem.MolFromSmiles(csmi)
            if cmol is None:
                continue
            ccanon = Chem.MolToSmiles(cmol, canonical=True)
            cfp = AllChem.GetMorganFingerprintAsBitVect(cmol, 2, 2048)
            tan = max((float(DataStructs.TanimotoSimilarity(cfp, g)) for g in gt_fps),
                      default=0.0)
            best_tan = max(best_tan, tan)
            is_exact = ccanon in gt_canon
            is_sim = tan >= tanimoto_threshold
            if is_exact or is_sim:
                pool_hit = True
                if first_hit_rank is None:
                    first_hit_rank = rank
                    first_hit_type = "exact" if is_exact else "similar"

        if pool_hit:
            n_pool_hit += 1
        if first_hit_rank is not None:
            mrr_sum += 1.0 / first_hit_rank
            if first_hit_rank <= 1:
                top1 += 1
                if first_hit_type == "exact":
                    exact1 += 1
                else:
                    sim_only1 += 1
            if first_hit_rank <= 3:
                top3 += 1
            if first_hit_rank <= 5:
                top5 += 1

        if collect_rows:
            rows.append({"Predecessor_SMILES": sr["smiles"],
                         "n_candidates": len(ranked),
                         "first_hit_rank": first_hit_rank if first_hit_rank else "",
                         "hit_type": first_hit_type or "miss",
                         "best_tanimoto_in_pool": round(best_tan, 4)})

    n = n_mol if n_mol > 0 else 1
    out = {
        "top1": top1 / n,
        "top3": top3 / n,
        "top5": top5 / n,
        "mrr": mrr_sum / n,
        "top1_exact_match": exact1 / n,
        "top1_similarity_only": sim_only1 / n,
        "candidate_pool_recall": n_pool_hit / n,
        "mean_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "n_molecules": n_mol,
        "n_empty": n_empty,
    }
    out["joint"] = 0.6 * out["top1"] + 0.4 * out["mrr"]
    if collect_rows:
        out["_rows"] = rows
    return out


# ═════════════════════════════════════════════════════════════════
# SINGLE-EPOCH TRAINING STEPS  (shared by CV folds and the full model)
# ═════════════════════════════════════════════════════════════════
def _train_som_epoch(sm, loader, optimizer, criterion, scaler, device, amp_enabled):
    sm.train()
    for b in loader:
        b = b.to(device, non_blocking=(device.type == "cuda"))
        ea = b.edge_attr if hasattr(b, "edge_attr") else None
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device.type, amp_enabled):
            loss = criterion(sm(b.x, b.edge_index, ea), b.y.view(-1))
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(sm.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(sm.parameters(), 5.0)
            optimizer.step()


def _train_rxn_epoch(rm, loader, optimizer, criterion, scaler, device, amp_enabled):
    rm.train()
    for b in loader:
        b = b.to(device, non_blocking=(device.type == "cuda"))
        ea = b.edge_attr if hasattr(b, "edge_attr") else None
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device.type, amp_enabled):
            loss = criterion(rm(b.x, b.edge_index, b.batch, ea), b.y.view(-1))
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(rm.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(rm.parameters(), 5.0)
            optimizer.step()


def _train_reranker_epoch(sm, rm, rr, som_eval_loader, rxn_eval_loader,
                          optimizer, scaler, device, amp_enabled, args,
                          diversity_weight: float, le) -> float:
    """
    One reranker epoch.

    NOTE (robustness fix vs. v3): the SOM and reaction predictions consumed here
    are collected with DEDICATED shuffle=False loaders, so the two result lists
    are guaranteed to refer to the same molecule at the same position. v3 relied
    on two independently-seeded shuffling loaders happening to produce identical
    permutations, which is true only as long as both are drawn the same number of
    times per epoch. The ranking loss is computed per molecule, so ordering has
    no effect on the result -- this only removes a silent misalignment risk.
    """
    sm.eval(); rm.eval(); rr.train()
    s_res = collect_som_predictions(sm, som_eval_loader, device, use_amp=amp_enabled)
    r_res = collect_rxn_predictions(rm, rxn_eval_loader, device, le, use_amp=amp_enabled)

    total = torch.tensor(0.0, device=device); nr = 0
    for sr, rres in zip(s_res, r_res):
        cands = generate_candidates(sr["smiles"], sr, rres, le, args.top_som, args.top_rxn)
        if not cands:
            continue
        n = len(cands)
        tan_v = torch.tensor([c["tanimoto_gt"] for c in cands], dtype=torch.float, device=device)
        som_pv = torch.tensor([c["som_prob"] for c in cands], dtype=torch.float, device=device)
        rxn_pv = torch.tensor([c["rxn_prob"] for c in cands], dtype=torch.float, device=device)
        fps_np = np.stack([c["ecfp4_vec"] for c in cands], 0)
        fp_m = torch.tensor(fps_np, dtype=torch.float, device=device)
        dot_np = fps_np @ fps_np.T
        sum_np = fps_np.sum(axis=1, keepdims=True)
        union_np = np.clip(sum_np + sum_np.T - dot_np, 1e-8, None)
        tan_mat = dot_np / union_np
        np.fill_diagonal(tan_mat, 0.0)
        inter = tan_mat.sum(axis=1) / (max(n - 1, 1))
        inter_v = torch.tensor(inter, dtype=torch.float, device=device)
        max_rank = max(max(c["som_rank"] for c in cands), 1)
        srn_v = torch.tensor([c["som_rank"] / max_rank for c in cands], dtype=torch.float, device=device)
        rrn_v = torch.tensor([c["rxn_rank"] / max_rank for c in cands], dtype=torch.float, device=device)

        total = total + reranker_loss(rr, som_pv, rxn_pv, tan_v, fp_m,
                                      inter_v, srn_v, rrn_v, device,
                                      diversity_weight=diversity_weight)
        nr += 1

    if nr > 0:
        optimizer.zero_grad(set_to_none=True)
        loss = total / nr
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(rr.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(rr.parameters(), 5.0)
            optimizer.step()
        return float(total.item() / nr)
    return float("nan")


# ═════════════════════════════════════════════════════════════════
# CROSS-VALIDATION: ONE FOLD
# ═════════════════════════════════════════════════════════════════
def train_one_fold(variant: VariantConfig,
                   fold_id, train_som, val_som, train_rxn, val_rxn,
                   raw_train_df, args, device, cv_dir, le, num_classes):
    """
    Train one CV fold.

    IMPORTANT (protocol change, R1-5 / R2-1):
    Cross-validation is now used ONLY for (a) hyperparameter / epoch-budget
    selection and (b) stability reporting. The model evaluated on the held-out
    test set is the one retrained on the COMPLETE development set in
    train_full_model(). No fold is cherry-picked for the headline result.
    """
    fold_seed = int(args.seed) + int(fold_id)
    enable_reproducibility(fold_seed)
    train_som = [copy.deepcopy(g) for g in train_som]
    val_som   = [copy.deepcopy(g) for g in val_som]
    train_rxn = [copy.deepcopy(g) for g in train_rxn]
    val_rxn   = [copy.deepcopy(g) for g in val_rxn]

    fold_dir = os.path.join(cv_dir, f"fold_{fold_id}")
    img_dir  = os.path.join(fold_dir, "images")
    ensure_dir(fold_dir)
    log_fp = os.path.join(fold_dir, "training_log.txt")

    if args.aug_factor > 0:
        train_som_aug, train_rxn_aug = augment_minority_classes(
            train_som, train_rxn, raw_train_df, le,
            aug_threshold=args.aug_threshold, aug_factor=args.aug_factor)
    else:
        print("    Augmentation: disabled (--aug-factor 0)")
        train_som_aug, train_rxn_aug = list(train_som), list(train_rxn)

    sc_som = build_scaler(train_som_aug)
    apply_scaler(train_som_aug, sc_som); apply_scaler(val_som, sc_som)
    sc_rxn = build_scaler(train_rxn_aug)
    apply_scaler(train_rxn_aug, sc_rxn); apply_scaler(val_rxn, sc_rxn)

    bs = args.batch_size; sd = fold_seed
    pin = (device.type == "cuda")
    tsl  = make_loader(train_som_aug, bs, True,  sd, args.num_workers, pin)
    vsl  = make_loader(val_som,       bs, False, sd, args.num_workers, pin)
    trl  = make_loader(train_rxn_aug, bs, True,  sd, args.num_workers, pin)
    vrl  = make_loader(val_rxn,       bs, False, sd, args.num_workers, pin)
    # dedicated deterministic loaders for the reranker collection pass
    tsl_e = make_loader(train_som_aug, bs, False, sd, args.num_workers, pin)
    trl_e = make_loader(train_rxn_aug, bs, False, sd, args.num_workers, pin)

    id_som = train_som_aug[0].x.size(-1)
    id_rxn = train_rxn_aug[0].x.size(-1)

    sm = build_som_model(variant, id_som, args, device) if variant.train_som else None
    rm = build_rxn_model(variant, id_rxn, num_classes, args, device) if variant.train_rxn else None
    rr = build_reranker(args, device) if variant.use_reranker else None

    amp_enabled = resolve_amp(device, bool(args.amp))
    sc_a = _make_grad_scaler(amp_enabled)
    sc_b = _make_grad_scaler(amp_enabled)
    sc_c = _make_grad_scaler(amp_enabled)

    focal = BinaryFocalLoss(args.alpha, args.gamma)
    cls_w = compute_class_weights(raw_train_df, le, device)
    ce = MultiClassFocalLoss(weight=cls_w, gamma=args.focal_gamma_rxn)

    os_m = torch.optim.AdamW(sm.parameters(), lr=args.lr, weight_decay=1e-4) if sm else None
    os_r = torch.optim.AdamW(rm.parameters(), lr=args.lr, weight_decay=1e-4) if rm else None
    os_rr = torch.optim.AdamW(rr.parameters(), lr=args.lr_rer, weight_decay=1e-4) if rr else None
    ss_m = torch.optim.lr_scheduler.CosineAnnealingLR(os_m, T_max=args.epochs, eta_min=args.lr_min) if os_m else None
    ss_r = torch.optim.lr_scheduler.CosineAnnealingLR(os_r, T_max=args.epochs, eta_min=args.lr_min) if os_r else None

    div_w = args.diversity_weight if variant.use_diversity else 0.0

    best_ss = best_rs = best_rrs = None
    best_score = -1.0; best_ep = -1; nim = 0
    best_som_mrr = -1.0; best_rxn_f1 = -1.0; best_joint = -1.0
    history: List[Dict] = []

    for ep in range(1, args.epochs + 1):
        if sm is not None:
            _train_som_epoch(sm, tsl, os_m, focal, sc_a, device, amp_enabled); ss_m.step()
        if rm is not None:
            _train_rxn_epoch(rm, trl, os_r, ce, sc_b, device, amp_enabled); ss_r.step()
        rer_loss = float("nan")
        if rr is not None:
            rer_loss = _train_reranker_epoch(sm, rm, rr, tsl_e, trl_e, os_rr, sc_c,
                                             device, amp_enabled, args, div_w, le)

        vm = som_ranking_metrics(sm, vsl, device, focal, use_amp=amp_enabled) if sm else {}
        vr = rxn_evaluate(rm, vrl, ce, device, le, use_amp=amp_enabled) if rm else {}
        if vm:
            mrr = float(vm["mrr"]) if not math.isnan(vm["mrr"]) else -1.0
            best_som_mrr = max(best_som_mrr, mrr)
        if vr:
            best_rxn_f1 = max(best_rxn_f1, float(vr["f1"]))

        jm = {}
        if variant.generate_metabolites:
            s_res = collect_som_predictions(sm, vsl, device, use_amp=amp_enabled)
            r_res = collect_rxn_predictions(rm, vrl, device, le, use_amp=amp_enabled)
            jm = reranked_metrics_from_predictions(
                s_res, r_res, rr, device, le,
                top_som=args.top_som, top_rxn=args.top_rxn,
                tanimoto_threshold=args.tanimoto_threshold,
                use_reranker=variant.use_reranker, use_amp=amp_enabled)
            best_joint = max(best_joint, float(jm["joint"]))

        if variant.selection_metric == "joint":
            score = float(jm["joint"])
        elif variant.selection_metric == "som_mrr":
            score = float(vm["mrr"]) if not math.isnan(vm["mrr"]) else -1.0
        else:
            score = float(vr["f1"])

        parts = [f"[{variant.name}|Fold {fold_id}] Ep {ep:03d}"]
        if ss_m is not None:
            parts.append(f"lr={ss_m.get_last_lr()[0]:.2e}")
        elif ss_r is not None:
            parts.append(f"lr={ss_r.get_last_lr()[0]:.2e}")
        if vm:
            parts.append(f"| SOM top1={vm['top1']:.3f} mrr={vm['mrr']:.3f}")
        if vr:
            parts.append(f"| Rxn acc={vr['acc']:.3f} f1={vr['f1']:.3f}")
        if rr is not None:
            parts.append(f"| Rer={rer_loss:.4f}")
        if jm:
            parts.append(f"| Met top1={jm['top1']:.3f} mrr={jm['mrr']:.3f} "
                         f"joint={jm['joint']:.4f} (n={jm['n_molecules']} empty={jm['n_empty']})")
        parts.append(f"| sel[{variant.selection_metric}]={score:.4f}")
        ln = " ".join(parts)
        print(ln); write_log(log_fp, ln)

        history.append({"epoch": ep,
                        "som_top1": vm.get("top1"), "som_mrr": vm.get("mrr"),
                        "rxn_acc": vr.get("acc"), "rxn_f1": vr.get("f1"),
                        "met_top1": jm.get("top1"), "met_mrr": jm.get("mrr"),
                        "met_joint": jm.get("joint"),
                        "reranker_loss": None if math.isnan(rer_loss) else rer_loss,
                        "selection_score": score})

        if score > best_score:
            best_score = score; best_ep = ep; nim = 0
            best_ss  = {k: v.cpu().clone() for k, v in sm.state_dict().items()} if sm else None
            best_rs  = {k: v.cpu().clone() for k, v in rm.state_dict().items()} if rm else None
            best_rrs = {k: v.cpu().clone() for k, v in rr.state_dict().items()} if rr else None
        else:
            nim += 1

        if nim >= args.patience:
            msg = (f"[{variant.name}|Fold {fold_id}] Early stop @ epoch {ep} "
                   f"({variant.selection_metric} stale {nim} epochs; "
                   f"best={best_score:.4f} @ ep {best_ep})")
            print(msg); write_log(log_fp, msg); break

    def _ld(m, st):
        if m is not None and st:
            m.load_state_dict({k: v.to(device) for k, v in st.items()})
    _ld(sm, best_ss); _ld(rm, best_rs); _ld(rr, best_rrs)

    if sm is not None:
        torch.save({"model": sm.state_dict(), "in_dim": id_som, "gnn": variant.gnn,
                    "hidden": args.hidden_som, "layers": args.layers, "heads": args.heads,
                    "dropout": args.dropout, "drop_edge_p": args.drop_edge_p,
                    "feature_mode": variant.feature_mode,
                    "feature_names": active_feature_names()},
                   os.path.join(fold_dir, "som_gat.pt"))
        joblib.dump(sc_som, os.path.join(fold_dir, "feature_scaler_som.pkl"))
    if rm is not None:
        torch.save({"model": rm.state_dict(), "in_dim": id_rxn, "gnn": variant.gnn,
                    "hidden": args.hidden_rxn, "layers": args.layers, "heads": args.heads,
                    "dropout": args.dropout, "num_classes": num_classes,
                    "drop_edge_p": args.drop_edge_p,
                    "feature_mode": variant.feature_mode},
                   os.path.join(fold_dir, "rxn_gat.pt"))
        joblib.dump(sc_rxn, os.path.join(fold_dir, "feature_scaler_rxn.pkl"))
    if rr is not None:
        torch.save({"model": rr.state_dict(), "fp_dim": 2048,
                    "hidden": args.hidden_rer, "dropout": args.dropout,
                    "diversity_weight": div_w},
                   os.path.join(fold_dir, "reranker.pt"))

    pd.DataFrame(history).to_csv(os.path.join(fold_dir, "epoch_history.csv"), index=False)

    # validation-set artefacts
    vs_res = vr_res = None
    if sm is not None:
        yt, yp = [], []
        with torch.no_grad():
            for b in vsl:
                b = b.to(device, non_blocking=(device.type == "cuda"))
                ea = b.edge_attr if hasattr(b, "edge_attr") else None
                with _autocast_ctx(device.type, amp_enabled):
                    prob = torch.sigmoid(sm(b.x, b.edge_index, ea))
                yt.append(detach_to_numpy(b.y.view(-1)))
                yp.append(detach_to_numpy(prob))
        thr = calibrate_threshold(np.concatenate(yt), np.concatenate(yp))
        open(os.path.join(fold_dir, "best_threshold.txt"), "w").write(str(thr))
        vs_res = collect_som_predictions(sm, vsl, device, use_amp=amp_enabled)
        save_som_csv(vs_res, os.path.join(fold_dir, "val_som_predictions.csv"))
    if rm is not None:
        vr_res = collect_rxn_predictions(rm, vrl, device, le, use_amp=amp_enabled)
        save_rxn_csv(vr_res, os.path.join(fold_dir, "val_rxn_predictions.csv"))
        save_rxn_report(rm, vrl, device, le, ce, fold_dir, prefix="val", use_amp=amp_enabled)

    final_val = {}
    if variant.generate_metabolites and vs_res is not None and vr_res is not None:
        final_val = reranked_metrics_from_predictions(
            vs_res, vr_res, rr, device, le,
            top_som=args.top_som, top_rxn=args.top_rxn,
            tanimoto_threshold=args.tanimoto_threshold,
            use_reranker=variant.use_reranker, use_amp=amp_enabled,
            collect_rows=True)
        rows = final_val.pop("_rows", [])
        pd.DataFrame(rows).to_csv(os.path.join(fold_dir, "val_error_analysis.csv"), index=False)
        v_mc, v_rl = [], []
        for sr, rres in zip(vs_res, vr_res):
            cands = generate_candidates(sr["smiles"], sr, rres, le, args.top_som, args.top_rxn)
            ranked = rank_candidates(rr, cands, device, variant.use_reranker, use_amp=amp_enabled)
            v_mc.append((sr["smiles"], sr.get("successor_smiles", ""), ranked))
            v_rl.append(ranked)
        save_metabolites_csv(v_mc, os.path.join(fold_dir, "val_metabolites.csv"), top_n=5)
        if args.save_images:
            ensure_dir(img_dir)
            save_all_images(vs_res, v_rl, img_dir, args.subimg_w, args.subimg_h, top_n=5)

    final_som = som_ranking_metrics(sm, vsl, device, focal, use_amp=amp_enabled) if sm else {}
    final_rxn = rxn_evaluate(rm, vrl, ce, device, le, use_amp=amp_enabled) if rm else {}

    fold_metrics = {
        "fold": int(fold_id),
        "selection_metric": variant.selection_metric,
        "best_selection_score": float(best_score),
        "best_epoch": int(best_ep),
        "val_som": {k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in final_som.items()},
        "val_rxn": {k: v for k, v in final_rxn.items() if k != "f1_per"},
        "val_rxn_f1_per_class": final_rxn.get("f1_per", {}),
        "val_metabolite": final_val,
        "best_som_mrr_seen": float(best_som_mrr),
        "best_rxn_f1_seen": float(best_rxn_f1),
        "best_joint_seen": float(best_joint),
        "n_train_graphs": int(len(train_som_aug)),
        "n_val_graphs": int(len(val_som)),
    }
    with open(os.path.join(fold_dir, "fold_metrics.json"), "w") as f:
        json.dump(fold_metrics, f, indent=2)

    return {"metrics": fold_metrics,
            "best_epoch": int(best_ep),
            "best_selection_score": float(best_score),
            "scaler_som": sc_som, "scaler_rxn": sc_rxn,
            "som_state": best_ss, "rxn_state": best_rs, "rer_state": best_rrs,
            "in_dim_som": id_som, "in_dim_rxn": id_rxn}


# ═════════════════════════════════════════════════════════════════
# PER-CLASS REACTION REPORT
# ═════════════════════════════════════════════════════════════════
@torch.no_grad()
def save_rxn_report(model, loader, device, le, criterion, outdir,
                    prefix="test", use_amp: bool = False):
    """Per-class precision / recall / F1 / support + confusion matrix (R1-7)."""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    model.eval()
    amp_enabled = resolve_amp(device, use_amp)
    yp, yt = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=(device.type == "cuda"))
        ea = batch.edge_attr if hasattr(batch, "edge_attr") else None
        with _autocast_ctx(device.type, amp_enabled):
            logits = model(batch.x, batch.edge_index, batch.batch, ea)
        yp.extend(detach_to_numpy(logits.argmax(1)).tolist())
        yt.extend(detach_to_numpy(batch.y.view(-1)).tolist())
    yp = np.array(yp); yt = np.array(yt)
    labels = list(range(len(le.classes_)))
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    rep = pd.DataFrame({"reaction_class": list(le.classes_),
                        "precision": np.round(p, 4), "recall": np.round(r, 4),
                        "f1": np.round(f, 4), "support": s})
    rep.to_csv(os.path.join(outdir, f"{prefix}_rxn_per_class_report.csv"), index=False)
    cm = confusion_matrix(yt, yp, labels=labels)
    pd.DataFrame(cm, index=list(le.classes_), columns=list(le.classes_)).to_csv(
        os.path.join(outdir, f"{prefix}_rxn_confusion_matrix.csv"))
    summary = {"accuracy": float((yp == yt).mean()) if len(yt) else float("nan"),
               "f1_weighted": float(sk_f1(yt, yp, average="weighted", zero_division=0)),
               "f1_macro": float(sk_f1(yt, yp, average="macro", zero_division=0)),
               "n": int(len(yt))}
    with open(os.path.join(outdir, f"{prefix}_rxn_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# ═════════════════════════════════════════════════════════════════
# SOM-ONLY ATOM RANKING METRICS FROM COLLECTED PREDICTIONS
# ═════════════════════════════════════════════════════════════════
def som_metrics_from_predictions(som_res: List[Dict]) -> Dict:
    t1 = t3 = t5 = 0; rr_s = 0.0; mn = 0
    for r in som_res:
        yg = np.zeros(len(r["probs"]), dtype=int)
        for idx in r["ground_truth_SOM"]:
            if 0 <= idx < len(yg):
                yg[idx] = 1
        if yg.sum() == 0:
            continue
        order = r["order"]
        t1 += int(yg[order[:1]].max() == 1)
        t3 += int(yg[order[:min(3, len(order))]].max() == 1)
        t5 += int(yg[order[:min(5, len(order))]].max() == 1)
        ranks = np.empty_like(order); ranks[order] = np.arange(1, len(order) + 1)
        rr_s += 1.0 / float(np.min(ranks[np.where(yg == 1)[0]])); mn += 1
    n = mn if mn else 1
    return {"top1": t1 / n, "top3": t3 / n, "top5": t5 / n,
            "mrr": rr_s / n, "mol_count": mn}


# ═════════════════════════════════════════════════════════════════
# FULL MODEL: RETRAIN ON THE COMPLETE DEVELOPMENT SET
# ═════════════════════════════════════════════════════════════════
def resolve_final_epochs(fold_res: List[Dict], args) -> int:
    """
    Epoch budget for the final model.

    The final model is trained on the COMPLETE development set, so there is no
    held-out validation split left for early stopping. The budget is therefore
    transferred from cross-validation: the epoch at which folds stopped
    improving, aggregated by --final-epochs-rule. This is the standard
    'refit on all data' protocol and is what keeps the held-out test set from
    influencing model selection in any way.
    """
    if int(args.final_epochs) > 0:
        return int(args.final_epochs)
    eps = [int(r["best_epoch"]) for r in fold_res if int(r["best_epoch"]) > 0]
    if not eps:
        return int(args.epochs)
    rule = args.final_epochs_rule
    if rule == "median":
        v = int(round(float(np.median(eps))))
    elif rule == "max":
        v = int(max(eps))
    else:
        v = int(round(float(np.mean(eps))))
    return max(1, v)


def train_full_model(variant: VariantConfig,
                     dev_som, dev_rxn, raw_dev_df,
                     n_epochs: int, args, device, out_dir, le, num_classes):
    """
    Train the final model on the complete development set (all CV folds pooled),
    for a fixed epoch budget, with no validation split and no early stopping.
    """
    enable_reproducibility(int(args.seed))
    dev_som = [copy.deepcopy(g) for g in dev_som]
    dev_rxn = [copy.deepcopy(g) for g in dev_rxn]
    ensure_dir(out_dir)
    log_fp = os.path.join(out_dir, "training_log.txt")

    hdr = (f"[{variant.name}|FULL] Training final model on the complete development set: "
           f"{len(dev_som)} molecules, {n_epochs} epochs (budget from CV).")
    print(hdr); write_log(log_fp, hdr)

    if args.aug_factor > 0:
        dev_som_aug, dev_rxn_aug = augment_minority_classes(
            dev_som, dev_rxn, raw_dev_df, le,
            aug_threshold=args.aug_threshold, aug_factor=args.aug_factor)
    else:
        dev_som_aug, dev_rxn_aug = list(dev_som), list(dev_rxn)

    sc_som = build_scaler(dev_som_aug); apply_scaler(dev_som_aug, sc_som)
    sc_rxn = build_scaler(dev_rxn_aug); apply_scaler(dev_rxn_aug, sc_rxn)

    bs = args.batch_size; sd = int(args.seed)
    pin = (device.type == "cuda")
    tsl  = make_loader(dev_som_aug, bs, True,  sd, args.num_workers, pin)
    trl  = make_loader(dev_rxn_aug, bs, True,  sd, args.num_workers, pin)
    tsl_e = make_loader(dev_som_aug, bs, False, sd, args.num_workers, pin)
    trl_e = make_loader(dev_rxn_aug, bs, False, sd, args.num_workers, pin)

    id_som = dev_som_aug[0].x.size(-1)
    id_rxn = dev_rxn_aug[0].x.size(-1)

    sm = build_som_model(variant, id_som, args, device) if variant.train_som else None
    rm = build_rxn_model(variant, id_rxn, num_classes, args, device) if variant.train_rxn else None
    rr = build_reranker(args, device) if variant.use_reranker else None

    amp_enabled = resolve_amp(device, bool(args.amp))
    sc_a = _make_grad_scaler(amp_enabled)
    sc_b = _make_grad_scaler(amp_enabled)
    sc_c = _make_grad_scaler(amp_enabled)

    focal = BinaryFocalLoss(args.alpha, args.gamma)
    cls_w = compute_class_weights(raw_dev_df, le, device)
    ce = MultiClassFocalLoss(weight=cls_w, gamma=args.focal_gamma_rxn)

    os_m = torch.optim.AdamW(sm.parameters(), lr=args.lr, weight_decay=1e-4) if sm else None
    os_r = torch.optim.AdamW(rm.parameters(), lr=args.lr, weight_decay=1e-4) if rm else None
    os_rr = torch.optim.AdamW(rr.parameters(), lr=args.lr_rer, weight_decay=1e-4) if rr else None
    ss_m = torch.optim.lr_scheduler.CosineAnnealingLR(os_m, T_max=n_epochs, eta_min=args.lr_min) if os_m else None
    ss_r = torch.optim.lr_scheduler.CosineAnnealingLR(os_r, T_max=n_epochs, eta_min=args.lr_min) if os_r else None

    div_w = args.diversity_weight if variant.use_diversity else 0.0
    history: List[Dict] = []

    for ep in range(1, n_epochs + 1):
        if sm is not None:
            _train_som_epoch(sm, tsl, os_m, focal, sc_a, device, amp_enabled); ss_m.step()
        if rm is not None:
            _train_rxn_epoch(rm, trl, os_r, ce, sc_b, device, amp_enabled); ss_r.step()
        rer_loss = float("nan")
        if rr is not None:
            rer_loss = _train_reranker_epoch(sm, rm, rr, tsl_e, trl_e, os_rr, sc_c,
                                             device, amp_enabled, args, div_w, le)
        # training-set diagnostics only -- NOT used for any selection decision
        tm = som_ranking_metrics(sm, tsl_e, device, focal, use_amp=amp_enabled) if sm else {}
        tr = rxn_evaluate(rm, trl_e, ce, device, le, use_amp=amp_enabled) if rm else {}
        ln = (f"[{variant.name}|FULL] Ep {ep:03d}/{n_epochs} "
              + (f"SOM top1={tm['top1']:.3f} mrr={tm['mrr']:.3f} " if tm else "")
              + (f"| Rxn acc={tr['acc']:.3f} f1={tr['f1']:.3f} " if tr else "")
              + (f"| Rer={rer_loss:.4f}" if rr is not None else "")
              + "  [train-set diagnostics]")
        print(ln); write_log(log_fp, ln)
        history.append({"epoch": ep, "train_som_top1": tm.get("top1"),
                        "train_som_mrr": tm.get("mrr"), "train_rxn_acc": tr.get("acc"),
                        "train_rxn_f1": tr.get("f1"),
                        "reranker_loss": None if math.isnan(rer_loss) else rer_loss})

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "epoch_history.csv"), index=False)

    if sm is not None:
        torch.save({"model": sm.state_dict(), "in_dim": id_som, "gnn": variant.gnn,
                    "hidden": args.hidden_som, "layers": args.layers, "heads": args.heads,
                    "dropout": args.dropout, "drop_edge_p": args.drop_edge_p,
                    "feature_mode": variant.feature_mode,
                    "feature_names": active_feature_names(),
                    "n_epochs": n_epochs},
                   os.path.join(out_dir, "som_model_full.pt"))
        joblib.dump(sc_som, os.path.join(out_dir, "feature_scaler_som.pkl"))
    if rm is not None:
        torch.save({"model": rm.state_dict(), "in_dim": id_rxn, "gnn": variant.gnn,
                    "hidden": args.hidden_rxn, "layers": args.layers, "heads": args.heads,
                    "dropout": args.dropout, "num_classes": num_classes,
                    "drop_edge_p": args.drop_edge_p,
                    "feature_mode": variant.feature_mode, "n_epochs": n_epochs},
                   os.path.join(out_dir, "rxn_model_full.pt"))
        joblib.dump(sc_rxn, os.path.join(out_dir, "feature_scaler_rxn.pkl"))
    if rr is not None:
        torch.save({"model": rr.state_dict(), "fp_dim": 2048, "hidden": args.hidden_rer,
                    "dropout": args.dropout, "diversity_weight": div_w,
                    "n_epochs": n_epochs},
                   os.path.join(out_dir, "reranker_full.pt"))

    return {"som_model": sm, "rxn_model": rm, "reranker": rr,
            "scaler_som": sc_som, "scaler_rxn": sc_rxn,
            "in_dim_som": id_som, "in_dim_rxn": id_rxn,
            "criterion_som": focal, "criterion_rxn": ce,
            "n_epochs": n_epochs}


# ═════════════════════════════════════════════════════════════════
# EVALUATION OF A TRAINED MODEL ON AN ARBITRARY SPLIT
# ═════════════════════════════════════════════════════════════════
def evaluate_split(variant: VariantConfig, bundle: Dict,
                   sg, rg, args, device, le, out_dir, split_name: str,
                   save_images: bool = False) -> Dict:
    """
    Run the trained model on one split and write every prediction artefact.
    Used for both the development ('train') split and the held-out test split.
    """
    ensure_dir(out_dir)
    amp_enabled = resolve_amp(device, bool(args.amp))
    sm, rm, rr = bundle["som_model"], bundle["rxn_model"], bundle["reranker"]
    pin = (device.type == "cuda")

    metrics: Dict[str, Any] = {"split": split_name, "n_molecules": len(sg)}

    som_res = rxn_res = None
    if sm is not None:
        sg_s = scale_graphs(sg, bundle["scaler_som"])
        sl = make_loader(sg_s, args.batch_size, False, args.seed, args.num_workers, pin)
        som_res = collect_som_predictions(sm, sl, device, use_amp=amp_enabled)
        save_som_csv(som_res, os.path.join(out_dir, f"{split_name}_som_predictions.csv"))
        metrics["som_only"] = som_metrics_from_predictions(som_res)

    if rm is not None:
        rg_s = scale_graphs(rg, bundle["scaler_rxn"])
        rl = make_loader(rg_s, args.batch_size, False, args.seed, args.num_workers, pin)
        rxn_res = collect_rxn_predictions(rm, rl, device, le, use_amp=amp_enabled)
        save_rxn_csv(rxn_res, os.path.join(out_dir, f"{split_name}_rxn_predictions.csv"))
        metrics["reaction"] = save_rxn_report(rm, rl, device, le,
                                              bundle["criterion_rxn"], out_dir,
                                              prefix=split_name, use_amp=amp_enabled)

    if variant.generate_metabolites and som_res is not None and rxn_res is not None:
        met = reranked_metrics_from_predictions(
            som_res, rxn_res, rr, device, le,
            top_som=args.top_som, top_rxn=args.top_rxn,
            tanimoto_threshold=args.tanimoto_threshold,
            use_reranker=variant.use_reranker, use_amp=amp_enabled,
            collect_rows=True)
        rows = met.pop("_rows", [])
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, f"{split_name}_error_analysis.csv"), index=False)
        metrics["metabolite"] = met

        mc, rl_all = [], []
        for sr, rres in zip(som_res, rxn_res):
            cands = generate_candidates(sr["smiles"], sr, rres, le, args.top_som, args.top_rxn)
            ranked = rank_candidates(rr, cands, device, variant.use_reranker, use_amp=amp_enabled)
            mc.append((sr["smiles"], sr.get("successor_smiles", ""), ranked))
            rl_all.append(ranked)
        save_metabolites_csv(mc, os.path.join(out_dir, f"{split_name}_metabolites.csv"), top_n=5)
        if save_images:
            img_dir = os.path.join(out_dir, f"{split_name}_images")
            ensure_dir(img_dir)
            save_all_images(som_res, rl_all, img_dir, args.subimg_w, args.subimg_h, top_n=5)

    with open(os.path.join(out_dir, f"{split_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ═════════════════════════════════════════════════════════════════
# CV FOLD MODELS AND ENSEMBLE, EVALUATED ON THE TEST SET (R2-1c)
# ═════════════════════════════════════════════════════════════════
def _fold_test_predictions(variant, fold, tst_sg, tst_rg, args, device, le, nc):
    """Collect raw SOM / reaction probabilities of one CV fold model on the test set."""
    amp_enabled = resolve_amp(device, bool(args.amp))
    pin = (device.type == "cuda")
    som_res = rxn_res = None
    if fold["som_state"] is not None:
        sm = build_som_model(variant, fold["in_dim_som"], args, device, drop_edge_p=0.0)
        sm.load_state_dict({k: v.to(device) for k, v in fold["som_state"].items()})
        sl = make_loader(scale_graphs(tst_sg, fold["scaler_som"]),
                         args.batch_size, False, args.seed, args.num_workers, pin)
        som_res = collect_som_predictions(sm, sl, device, use_amp=amp_enabled)
    if fold["rxn_state"] is not None:
        rm = build_rxn_model(variant, fold["in_dim_rxn"], nc, args, device, drop_edge_p=0.0)
        rm.load_state_dict({k: v.to(device) for k, v in fold["rxn_state"].items()})
        rl = make_loader(scale_graphs(tst_rg, fold["scaler_rxn"]),
                         args.batch_size, False, args.seed, args.num_workers, pin)
        rxn_res = collect_rxn_predictions(rm, rl, device, le, use_amp=amp_enabled)
    return som_res, rxn_res


def _average_som_results(per_fold: List[List[Dict]], top_k=5) -> List[Dict]:
    """Average per-atom SOM probabilities across fold models (loaders are shuffle=False)."""
    base = [dict(r) for r in per_fold[0]]
    for i, rec in enumerate(base):
        stack = np.stack([pf[i]["probs"] for pf in per_fold], 0)
        probs = stack.mean(0)
        order = np.argsort(-probs)
        rec["probs"] = probs
        rec["order"] = order
        for k in range(1, top_k + 1):
            rec[f"top{k}_SOM"] = int(order[k - 1]) if (k - 1) < len(order) else None
    return base


def _average_rxn_results(per_fold: List[List[Dict]], le, top_k=5) -> List[Dict]:
    """Average class-probability vectors across fold models."""
    base = [dict(r) for r in per_fold[0]]
    for i, rec in enumerate(base):
        stack = np.stack([pf[i]["probs"] for pf in per_fold], 0)
        probs = stack.mean(0)
        order = np.argsort(-probs)
        rec["probs"] = probs
        rec["order"] = order
        for k in range(1, top_k + 1):
            rec[f"top{k}_rxn"] = (le.inverse_transform([int(order[k - 1])])[0]
                                  if (k - 1) < len(order) else None)
    return base


def evaluate_cv_models_on_test(variant, fold_res, tst_sg, tst_rg,
                               args, device, le, nc, cv_dir) -> Dict:
    """
    Evaluate every CV fold model on the held-out test set, plus a 5-model
    probability-averaging ensemble.

    This is the analysis Reviewer 2 asked for (comment 1c): it shows how much of
    the headline test result depends on which fold is used, and gives the reader
    a fold-selection-free reference point. These numbers are REPORTED, not used
    for any selection decision.
    """
    amp_enabled = resolve_amp(device, bool(args.amp))
    per_fold_som, per_fold_rxn = [], []
    rows = []

    for i, fold in enumerate(fold_res, 1):
        som_res, rxn_res = _fold_test_predictions(variant, fold, tst_sg, tst_rg,
                                                  args, device, le, nc)
        entry: Dict[str, Any] = {"fold": i}
        if som_res is not None:
            per_fold_som.append(som_res)
            entry["som_only"] = som_metrics_from_predictions(som_res)
        if rxn_res is not None:
            per_fold_rxn.append(rxn_res)
        if variant.generate_metabolites and som_res is not None and rxn_res is not None:
            rr = None
            if variant.use_reranker and fold["rer_state"] is not None:
                rr = build_reranker(args, device)
                rr.load_state_dict({k: v.to(device) for k, v in fold["rer_state"].items()})
            entry["metabolite"] = reranked_metrics_from_predictions(
                som_res, rxn_res, rr, device, le,
                top_som=args.top_som, top_rxn=args.top_rxn,
                tanimoto_threshold=args.tanimoto_threshold,
                use_reranker=variant.use_reranker, use_amp=amp_enabled)
        rows.append(entry)

    out: Dict[str, Any] = {"per_fold": rows}

    # ── ensemble ──────────────────────────────────────────────────
    ens: Dict[str, Any] = {}
    ens_som = _average_som_results(per_fold_som) if per_fold_som else None
    ens_rxn = _average_rxn_results(per_fold_rxn, le) if per_fold_rxn else None
    if ens_som is not None:
        ens["som_only"] = som_metrics_from_predictions(ens_som)
        save_som_csv(ens_som, os.path.join(cv_dir, "ensemble_test_som_predictions.csv"))
    if ens_rxn is not None:
        save_rxn_csv(ens_rxn, os.path.join(cv_dir, "ensemble_test_rxn_predictions.csv"))
    if variant.generate_metabolites and ens_som is not None and ens_rxn is not None:
        rerankers = []
        if variant.use_reranker:
            for fold in fold_res:
                if fold["rer_state"] is None:
                    continue
                m = build_reranker(args, device)
                m.load_state_dict({k: v.to(device) for k, v in fold["rer_state"].items()})
                m.eval(); rerankers.append(m)

        class _MeanReranker(nn.Module):
            """Score = mean of the fold rerankers' scores."""
            def __init__(self, models): super().__init__(); self.models = nn.ModuleList(models)
            def forward(self, feat):
                return torch.stack([m(feat) for m in self.models], 0).mean(0)

        mean_rr = _MeanReranker(rerankers).to(device) if rerankers else None
        ens["metabolite"] = reranked_metrics_from_predictions(
            ens_som, ens_rxn, mean_rr, device, le,
            top_som=args.top_som, top_rxn=args.top_rxn,
            tanimoto_threshold=args.tanimoto_threshold,
            use_reranker=(mean_rr is not None), use_amp=amp_enabled)
    out["ensemble"] = ens

    # ── mean +/- SD across folds (R2-1b) ──────────────────────────
    def _agg(path: List[str]):
        vals = []
        for r in rows:
            cur: Any = r
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    cur = None; break
                cur = cur[p]
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)), "max": float(np.max(vals)), "n_folds": len(vals)}

    agg: Dict[str, Any] = {}
    for key in ["top1", "top3", "top5", "mrr"]:
        a = _agg(["som_only", key])
        if a: agg[f"som_{key}"] = a
    for key in ["top1", "top3", "top5", "mrr", "top1_exact_match", "candidate_pool_recall"]:
        a = _agg(["metabolite", key])
        if a: agg[f"metabolite_{key}"] = a
    out["fold_mean_sd_on_test"] = agg

    with open(os.path.join(cv_dir, "fold_models_on_test.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ═════════════════════════════════════════════════════════════════
# FOLD COMPOSITION DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════
def fold_composition_report(pre_df, fold_row_indices: List[Tuple[np.ndarray, np.ndarray]],
                            le, path: str):
    """
    Describe what is actually in each validation fold, so fold-to-fold
    variability can be related to scaffold composition, class distribution and
    molecular complexity rather than left unexplained.
    """
    recs = []
    for fid, (tri, vli) in enumerate(fold_row_indices, 1):
        vdf = pre_df.iloc[vli]
        smis = vdf["Predecessor_SMILES"].astype(str).tolist()
        scafs = [murcko_scaffold(s) for s in smis]
        heavy, n_som = [], []
        for _, row in vdf.iterrows():
            m = Chem.MolFromSmiles(str(row["Predecessor_SMILES"]))
            heavy.append(m.GetNumHeavyAtoms() if m else np.nan)
            try:
                n_som.append(len(safe_literal_list(str(row["y_sites"]))))
            except Exception:
                n_som.append(np.nan)
        counts = vdf["reaction_superclass"].value_counts(normalize=True)
        entropy = float(-(counts * np.log(counts.clip(lower=1e-12))).sum()) if len(counts) else 0.0
        rec = {"fold": fid,
               "n_train_molecules": int(len(tri)),
               "n_val_molecules": int(len(vli)),
               "n_unique_val_scaffolds": int(len(set(s for s in scafs if s))),
               "scaffold_diversity": round(len(set(s for s in scafs if s)) / max(len(scafs), 1), 4),
               "n_val_reaction_classes": int(vdf["reaction_superclass"].nunique()),
               "reaction_class_entropy": round(entropy, 4),
               "mean_heavy_atoms": round(float(np.nanmean(heavy)), 2) if heavy else np.nan,
               "mean_SOM_per_molecule": round(float(np.nanmean(n_som)), 3) if n_som else np.nan}
        for cls in le.classes_:
            rec[f"n_{cls}"] = int((vdf["reaction_superclass"] == cls).sum())
        recs.append(rec)
    pd.DataFrame(recs).to_csv(path, index=False)
    return recs


# ═════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════
def build_argparser(variant: VariantConfig):
    import argparse
    p = argparse.ArgumentParser(
        description=f"{variant.name}: {variant.description}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv",               type=str,   required=True)
    p.add_argument("--outdir",            type=str,   default=f"runs/{variant.name}")
    p.add_argument("--epochs",            type=int,   default=120)
    p.add_argument("--batch-size",        type=int,   default=32)
    p.add_argument("--lr",                type=float, default=2e-3)
    p.add_argument("--lr-min",            type=float, default=1e-5)
    p.add_argument("--lr-rer",            type=float, default=5e-4)
    p.add_argument("--hidden-som",        type=int,   default=128)
    p.add_argument("--hidden-rxn",        type=int,   default=128)
    p.add_argument("--hidden-rer",        type=int,   default=256)
    p.add_argument("--layers",            type=int,   default=3)
    p.add_argument("--heads",             type=int,   default=4)
    p.add_argument("--simple-hidden",     type=int,   default=64,
                   help="Hidden width of the simplified GCN baseline")
    p.add_argument("--simple-layers",     type=int,   default=2,
                   help="Number of layers in the simplified GCN baseline")
    p.add_argument("--dropout",           type=float, default=0.35)
    p.add_argument("--drop-edge-p",       type=float, default=0.10)
    p.add_argument("--alpha",             type=float, default=0.75)
    p.add_argument("--gamma",             type=float, default=2.0)
    p.add_argument("--focal-gamma-rxn",   type=float, default=2.0)
    p.add_argument("--diversity-weight",  type=float, default=0.15)
    p.add_argument("--aug-threshold",     type=int,   default=80)
    p.add_argument("--aug-factor",        type=int,   default=2,
                   help="Randomised-SMILES copies per minority-class molecule; 0 disables")
    p.add_argument("--kfolds",            type=int,   default=5)
    p.add_argument("--patience",          type=int,   default=20)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--device",            type=str,   default="cuda")
    p.add_argument("--amp",               type=int,   default=1)
    p.add_argument("--num-workers",       type=int,   default=0)
    p.add_argument("--top-som",           type=int,   default=5)
    p.add_argument("--top-rxn",           type=int,   default=5)
    p.add_argument("--tanimoto-threshold", type=float, default=0.95)
    p.add_argument("--fragment-policy",   type=str,   default="all",
                   choices=["all", "largest", "none"],
                   help="How to handle multi-fragment products from the N-/O-dealkylation "
                        "rules. 'all' scores every fragment above --min-fragment-atoms as "
                        "its own candidate; 'largest' keeps only the biggest; 'none' is the "
                        "pre-fix behaviour and is provided only for reproducing old runs")
    p.add_argument("--min-fragment-atoms", type=int, default=3,
                   help="Heavy-atom floor for a product fragment to become a candidate")
    p.add_argument("--subimg-w",          type=int,   default=350)
    p.add_argument("--subimg-h",          type=int,   default=280)
    p.add_argument("--save-images",       type=int,   default=0,
                   help="Render metabolite panels (slow); test set of the full model only")
    # ── final-model protocol ──────────────────────────────────────
    p.add_argument("--final-epochs",      type=int,   default=0,
                   help="Fixed epoch budget for the full-development-set model; "
                        "0 = derive from CV via --final-epochs-rule")
    p.add_argument("--final-epochs-rule", type=str,   default="mean",
                   choices=["mean", "median", "max"],
                   help="How to aggregate the per-fold best epoch into the final budget")
    p.add_argument("--skip-cv",           type=int,   default=0,
                   help="Skip 5-fold CV (requires --final-epochs > 0)")
    p.add_argument("--skip-full",         type=int,   default=0,
                   help="Run cross-validation only, do not train the final model")
    p.add_argument("--disable-reranker",  type=int,   default=0,
                   help="Ablate the learned reranker; candidates are ordered by "
                        "P(SOM) x P(reaction) instead")
    return p


# ═════════════════════════════════════════════════════════════════
# PIPELINE
# ═════════════════════════════════════════════════════════════════
def run_pipeline(variant: VariantConfig, argv=None):
    variant.check()
    args = build_argparser(variant).parse_args(argv)

    if int(args.disable_reranker):
        variant = VariantConfig(**{**asdict(variant), "use_reranker": False})
        variant.check()
    if int(args.skip_cv) and int(args.final_epochs) <= 0:
        raise SystemExit("--skip-cv requires --final-epochs > 0")

    set_feature_mode(variant.feature_mode)
    set_fragment_policy(args.fragment_policy, args.min_fragment_atoms)
    enable_reproducibility(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device.startswith("cuda") else "cpu")
    amp_enabled = resolve_amp(device, bool(args.amp))

    outdir   = args.outdir
    cv_dir   = os.path.join(outdir, "five_fold_CV")
    full_dir = os.path.join(outdir, "full_model_results")
    ensure_dir(outdir); ensure_dir(cv_dir); ensure_dir(full_dir)

    print("=" * 78)
    print(f"  VARIANT : {variant.name}")
    print(f"  {variant.description}")
    print(f"  encoder={variant.gnn}  features={variant.feature_mode} "
          f"({active_feature_dim()}-d)  reranker={variant.use_reranker}  "
          f"diversity={variant.use_diversity}  selection={variant.selection_metric}")
    print(f"  product fragments: policy={args.fragment_policy} "
          f"(min {args.min_fragment_atoms} heavy atoms)")
    print(f"  device={device}  amp={'on' if amp_enabled else 'off'}")
    print(f"  outdir={outdir}")
    print("=" * 78)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump({"variant": asdict(variant), "args": vars(args),
                   "atom_features_used": active_feature_names(),
                   "fragment_policy": {"policy": args.fragment_policy,
                                       "min_heavy_atoms": args.min_fragment_atoms}},
                  f, indent=2)

    # ── data ──────────────────────────────────────────────────────
    print("Loading dataset ...")
    sg, rg, le, df = load_dataset(args.csv)
    nc = len(le.classes_)
    print(f"  Molecules   : {len(sg)}")
    print(f"  Rxn classes : {nc}")
    print(f"  Atom feats  : {active_feature_dim()} ({variant.feature_mode})")
    cls_rows = []
    for i, cls in enumerate(le.classes_):
        cnt = int((df["reaction_superclass"] == cls).sum())
        flag = " *** minority" if cnt < args.aug_threshold else ""
        print(f"    [{i:2d}] {cls:48s} n={cnt:4d}{flag}")
        cls_rows.append({"index": i, "reaction_class": cls, "n_molecules": cnt,
                         "minority": cnt < args.aug_threshold})
    pd.DataFrame(cls_rows).to_csv(os.path.join(outdir, "reaction_class_distribution.csv"),
                                  index=False)
    with open(os.path.join(outdir, "label_mapping.json"), "w") as f:
        json.dump({int(i): c for i, c in enumerate(le.classes_)}, f, indent=2)

    gt_frag = count_multifragment_ground_truth(df)
    if gt_frag["n_multifragment"]:
        print(f"  [note] {gt_frag['n_multifragment']} of {gt_frag['n_ground_truth']} "
              f"annotated metabolites are multi-fragment (salts/counterions); these "
              f"cannot exact-match a single-fragment candidate.")
    with open(os.path.join(outdir, "ground_truth_fragment_check.json"), "w") as f:
        json.dump(gt_frag, f, indent=2)

    # ── scaffold hold-out test split (identical to v3) ────────────
    pre_idx, tst_idx, holdout_info, parent_df = scaffold_test_split(df, 0.20, args.seed)
    pre_sg = [sg[i] for i in pre_idx]; pre_rg = [rg[i] for i in pre_idx]
    tst_sg = [sg[i] for i in tst_idx]; tst_rg = [rg[i] for i in tst_idx]
    pre_df = df.iloc[pre_idx].reset_index(drop=True)
    print(f"  Development molecules : {len(pre_sg)}")
    print(f"  Held-out test         : {len(tst_sg)}")
    print(f"  Holdout split         : method={holdout_info['method']} "
          f"test_scaffolds={holdout_info['n_test_scaffolds']}")
    with open(os.path.join(outdir, "split_info.json"), "w") as f:
        json.dump({"holdout": {k: (int(v) if isinstance(v, (np.integer,)) else v)
                               for k, v in holdout_info.items()},
                   "n_development": len(pre_sg), "n_test": len(tst_sg)}, f, indent=2, default=str)

    pre_parent_df = build_parent_scaffold_df(pre_df).reset_index(drop=True)
    pre_parent_df["parent_index"] = np.arange(len(pre_parent_df), dtype=int)
    parent_to_rows = (pre_df.assign(parent_id=pre_df["Predecessor_SMILES"].astype(str).str.strip())
                      .groupby("parent_id").indices)

    fold_res: List[Dict] = []
    fold_row_idx: List[Tuple[np.ndarray, np.ndarray]] = []

    # ── STAGE 1: 5-fold CV -> five_fold_CV/ ───────────────────────
    if not int(args.skip_cv):
        splits = scaffold_kfold_split(pre_parent_df, args.kfolds, args.seed)
        for fid, (tri_p, vli_p) in enumerate(splits, 1):
            tr_ids = pre_parent_df.iloc[tri_p]["parent_id"].tolist()
            vl_ids = pre_parent_df.iloc[vli_p]["parent_id"].tolist()
            tri = np.array(sorted([r for pid in tr_ids for r in parent_to_rows[pid]]), dtype=int)
            vli = np.array(sorted([r for pid in vl_ids for r in parent_to_rows[pid]]), dtype=int)
            fold_row_idx.append((tri, vli))
            print(f"\n{'=' * 72}\n  [{variant.name}] Fold {fid}/{args.kfolds}  "
                  f"train={len(tri)} val={len(vli)}\n{'=' * 72}")
            fold_res.append(train_one_fold(
                variant, fid,
                [pre_sg[i] for i in tri], [pre_sg[i] for i in vli],
                [pre_rg[i] for i in tri], [pre_rg[i] for i in vli],
                pre_df.iloc[tri].reset_index(drop=True),
                args, device, cv_dir, le, nc))

        fold_composition_report(pre_df, fold_row_idx, le,
                                os.path.join(cv_dir, "fold_composition.csv"))
        write_cv_summary(variant, fold_res, cv_dir, args)
        if fold_res:
            print("\nEvaluating each CV fold model (and their ensemble) on the held-out test set ...")
            evaluate_cv_models_on_test(variant, fold_res, tst_sg, tst_rg,
                                       args, device, le, nc, cv_dir)
    else:
        print("Skipping cross-validation (--skip-cv 1).")

    if int(args.skip_full):
        print("\nStopping after cross-validation (--skip-full 1).")
        return

    # ── STAGE 2: final model on the COMPLETE development set ──────
    n_final = resolve_final_epochs(fold_res, args)
    print(f"\n{'=' * 78}\n  FINAL MODEL — complete development set "
          f"({len(pre_sg)} molecules), {n_final} epochs\n{'=' * 78}")
    bundle = train_full_model(variant, pre_sg, pre_rg, pre_df,
                              n_final, args, device, full_dir, le, nc)

    print("\nPredicting on the development (training) set ...")
    train_metrics = evaluate_split(variant, bundle, pre_sg, pre_rg, args, device, le,
                                   full_dir, "train", save_images=False)
    print("Predicting on the held-out test set ...")
    test_metrics = evaluate_split(variant, bundle, tst_sg, tst_rg, args, device, le,
                                  full_dir, "test", save_images=bool(args.save_images))

    summary = {"variant": asdict(variant),
               "final_epochs": n_final,
               "final_epochs_source": ("--final-epochs" if int(args.final_epochs) > 0
                                       else f"CV {args.final_epochs_rule} of per-fold best epoch"),
               "n_development_molecules": len(pre_sg),
               "n_test_molecules": len(tst_sg),
               "train": train_metrics,
               "test": test_metrics}
    with open(os.path.join(full_dir, "full_model_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 78}\n  {variant.name} — HELD-OUT TEST RESULTS "
          f"(model retrained on the complete development set)\n{'=' * 78}")
    for block in ("som_only", "reaction", "metabolite"):
        if block in test_metrics:
            print(f"  [{block}]")
            for k, v in test_metrics[block].items():
                print(f"    {k:26s}: {v:.4f}" if isinstance(v, float) else f"    {k:26s}: {v}")
    print(f"\nDone.\n  CV outputs        : {cv_dir}\n  Full-model outputs: {full_dir}")


def write_cv_summary(variant: VariantConfig, fold_res: List[Dict], cv_dir: str, args):
    """Per-fold table plus mean +/- SD for every headline metric (R2-1b)."""
    rows = []
    for r in fold_res:
        m = r["metrics"]
        row = {"fold": m["fold"], "best_epoch": m["best_epoch"],
               "selection_metric": m["selection_metric"],
               "best_selection_score": round(m["best_selection_score"], 4),
               "n_train_graphs": m["n_train_graphs"], "n_val_graphs": m["n_val_graphs"]}
        for k, v in (m.get("val_som") or {}).items():
            if isinstance(v, (int, float)):
                row[f"val_som_{k}"] = round(float(v), 4)
        for k, v in (m.get("val_rxn") or {}).items():
            if isinstance(v, (int, float)):
                row[f"val_rxn_{k}"] = round(float(v), 4)
        for k, v in (m.get("val_metabolite") or {}).items():
            if isinstance(v, (int, float)):
                row[f"val_met_{k}"] = round(float(v), 4)
        rows.append(row)

    dfm = pd.DataFrame(rows)
    dfm.to_csv(os.path.join(cv_dir, "cv_fold_metrics.csv"), index=False)

    num_cols = [c for c in dfm.columns
                if c not in ("fold", "selection_metric") and pd.api.types.is_numeric_dtype(dfm[c])]
    stats = pd.DataFrame({
        "metric": num_cols,
        "mean": [round(float(dfm[c].mean()), 4) for c in num_cols],
        "sd":   [round(float(dfm[c].std(ddof=1)), 4) if len(dfm) > 1 else 0.0 for c in num_cols],
        "min":  [round(float(dfm[c].min()), 4) for c in num_cols],
        "max":  [round(float(dfm[c].max()), 4) for c in num_cols],
    })
    stats.to_csv(os.path.join(cv_dir, "cv_mean_sd.csv"), index=False)

    lines = [f"===== {variant.name}: {args.kfolds}-fold scaffold CV on the development set =====",
             f"  {variant.description}",
             f"  selection metric : {variant.selection_metric}",
             "",
             "  NOTE: cross-validation is used for epoch-budget selection and stability",
             "  reporting only. The reported test-set result comes from a single model",
             "  retrained on the complete development set (see full_model_results/).",
             ""]
    lines.append(dfm.to_string(index=False))
    lines += ["", "----- mean +/- SD across folds -----"]
    for _, r in stats.iterrows():
        lines.append(f"  {r['metric']:34s} {r['mean']:.4f} +/- {r['sd']:.4f} "
                     f"[min {r['min']:.4f}, max {r['max']:.4f}]")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(cv_dir, "cv_summary.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(cv_dir, "cv_summary.json"), "w") as f:
        json.dump({"variant": asdict(variant),
                   "n_folds": len(fold_res),
                   "folds": [r["metrics"] for r in fold_res],
                   "mean_sd": stats.to_dict(orient="records")},
                  f, indent=2, default=str)
