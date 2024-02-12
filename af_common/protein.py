# Copyright 2023 California Institute of Technology
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Protein processing, adapted from the AlphaFold parser."""

import dataclasses
import io
from typing import Any, Mapping, Optional, List, Tuple
from af_common import residue_constants
from Bio.PDB import PDBParser
import numpy as np
import warnings

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass()
class Protein:
    """Protein structure representation."""

    # The first entry stores amino acid sequence in letter representation.
    # The second entry stores a 0-1 mask for observed standard residues.
    # Non-standard residues are mapped to <mask> to interact with protein language models.
    letter_sequences: List[Tuple[str, str, np.ndarray]]

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, atom_type_num, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Added
    # Integer for atom type.
    atomtypes: np.ndarray  # [num_res, element_type_num]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, atom_type_num]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, atom_type_num]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} chains "
                "because these cannot be written to PDB format."
            )


def from_pdb_string(
    pdb_str: str,
    chain_id: Optional[list] = None,
    bounding_box: Optional[np.array] = None,
    res_start: int = None,
    res_end: int = None,
    model_id: int = 0,
    allow_insertion_code=False,
    **kwargs,
) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.
      bounding_box: If provided, only chains with backbone intersecting with the box are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    model = models[model_id]
    if isinstance(chain_id, str):
        chain_id = [chain_id]

    seq = []
    atom_positions = []
    aatype = []
    atomtypes = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain_idx, chain in enumerate(model):
        if chain_id is not None and chain.get_id() not in chain_id:
            continue
        if bounding_box is not None:
            ca_pos = np.array(
                [res["CA"].get_coord() for res in chain if res.has_id("CA")]
            )
            ca_in_box = (ca_pos > bounding_box[0]) & (ca_pos < bounding_box[1])
            if not np.any(np.all(ca_in_box, axis=1), axis=0):
                continue
        for res_idx, res in enumerate(chain):
            if res_start is not None and res_idx < res_start:
                continue
            if res_end is not None and res_idx > res_end:
                continue
            # strict bounding
            if bounding_box is not None:
                if not res.has_id("CA"):
                    continue
                ca_pos = res["CA"].get_coord()
                ca_in_box = (ca_pos > bounding_box[0]) & (ca_pos < bounding_box[1])
                if not np.all(ca_in_box):
                    continue
            if res.id[2] != " ":
                if allow_insertion_code:
                    warnings.warn(
                        f"PDB contains an insertion code at chain {chain.id} and residue "
                        f"index {res.id[1]} and `allow_insertion_code` is set to True. "
                        "Please ensure the residue indices are consecutive before performing downstream analysis."
                    )
                else:
                    raise ValueError(
                        f"PDB contains an insertion code at chain {chain.id} and residue "
                        f"index {res.id[1]}. Such samples are not supported by default."
                    )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            if res_shortname == "X":
                # Unlike the alphafold parser, we skip all non-standard residues
                # Such modified residues are treated as covalent ligand with all-atom resolution
                continue
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            eletypes = np.zeros((residue_constants.atom_type_num,))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            # sidechain_atom_order = 3
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                if atom.occupancy and atom.occupancy < 0.5:
                    # Remove too-flexible atoms
                    continue
                # if atom.element not in residue_constants.element_id.keys():
                #     continue
                eletypes[
                    residue_constants.atom_order[atom.name]
                ] = residue_constants.element_id[atom.element]
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if mask[1] < 1:
                warnings.warn(f"Missing Ca: {res.id[1]}")
                # Skip if the alpha-carbon atom is not resolved
                continue
            seq.append(res_shortname)
            aatype.append(restype_idx)
            atomtypes.append(eletypes)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    # Evaluate the gapless protein sequence for each chain
    seqs = []
    last_chain_idx = -1
    last_chain_seq = None
    last_chain_mask = None
    for site_idx in range(len(seq)):
        if chain_ids[site_idx] != last_chain_idx:
            if last_chain_seq is not None:
                last_chain_seq = "".join(last_chain_seq)
                seqs.append(
                    (last_chain_idx, "".join(last_chain_seq), np.array(last_chain_mask))
                )
            last_chain_idx = chain_ids[site_idx]
            last_chain_seq = []
            last_chain_mask = []
            last_res_id = -999
        if residue_index[site_idx] <= last_res_id:
            raise ValueError(
                f"PDB residue index is not monotonous at chain {chain.id} and residue "
                f"index {res.id[1]}. The sample is discarded."
            )
        elif last_res_id == -999:
            gap_size = 0
        else:
            gap_size = residue_index[site_idx] - last_res_id - 1
        for _ in range(gap_size):
            last_chain_seq.append("<mask>")
            last_chain_mask.append(False)
        last_chain_seq.append(seq[site_idx])
        last_chain_mask.append(True)
    seqs.append((last_chain_idx, "".join(last_chain_seq), np.array(last_chain_mask)))

    for chain_seq in seqs:
        if np.mean(chain_seq[2]) < 0.75:
            raise ValueError(
                f"The PDB structure residue coverage for {chain.id}"
                f"is below 75%. The sample is discarded."
            )

    return Protein(
        letter_sequences=seqs,
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        atomtypes=np.array(atomtypes),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )


def extract_subchains_in_box(
    pdb_str: str,
    bounding_box: np.array,
) -> Protein:
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    model = models[0]

    metadata = []

    for chain in model:
        seq, mask = [], []
        ca_pos = np.array([res["CA"].get_coord() for res in chain if res.has_id("CA")])
        if ca_pos.size == 0:
            continue
        ca_in_box = (ca_pos > bounding_box[0]) & (ca_pos < bounding_box[1])
        if not np.any(np.all(ca_in_box, axis=1), axis=0):
            metadata.append(None)
            continue
        for res in chain:
            if res.id[2] != " ":
                print(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. Only the canonical residue is kept."
                )
                continue
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            seq.append(res_shortname)
            res_in_box = False
            for atom in res:
                atom_pos = atom.get_coord()
                atom_in_box = (atom_pos > bounding_box[0]) & (
                    atom_pos < bounding_box[1]
                )
                if np.all(atom_in_box):
                    res_in_box = True
                    break
            mask.append(res_in_box)
        seq = np.array(seq)
        res_idx = np.arange(len(seq))
        mask = np.array(mask)
        if not mask.any():
            metadata.append(None)
            continue
        begin_idx, end_idx = min(res_idx[mask]), max(res_idx[mask])
        begin_idx = max(begin_idx - 10, 0)
        end_idx = min(end_idx + 10, len(seq))
        metadata.append(
            (chain.get_id(), "".join(seq[begin_idx : end_idx + 1]), begin_idx)
        )

    return metadata


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


def to_pdb(prot: Protein, model=1) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
            )
        chain_ids[i] = PDB_CHAIN_IDS[i]

    # pdb_lines.append(f"MODEL     {model}")
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(
        _chain_end(
            atom_index,
            res_1to3(aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        )
    )
    pdb_lines.append("ENDMDL")
    # pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = False,
) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      remove_leading_feature_dimension: Whether to remove the leading dimension
        of the `features` values.

    Returns:
      A protein instance.
    """
    fold_output = result["structure_module"]

    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if "asym_id" in features:
        chain_index = _maybe_remove_leading_dim(features["asym_id"])
    else:
        chain_index = np.zeros_like(_maybe_remove_leading_dim(features["aatype"]))

    if b_factors is None:
        b_factors = np.zeros_like(fold_output["final_atom_mask"])

    return Protein(
        letter_sequences=None,
        aatype=_maybe_remove_leading_dim(features["aatype"]),
        atom_positions=fold_output["final_atom_positions"],
        atom_mask=fold_output["final_atom_mask"],
        residue_index=_maybe_remove_leading_dim(features["residue_index"]),
        chain_index=chain_index,
        b_factors=b_factors,
        atomtypes=None,
    )
