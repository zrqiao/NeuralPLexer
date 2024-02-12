from collections import defaultdict

import networkx
import networkx as nx
import numpy as np
from deprecated import deprecated
from rdkit import Chem
from rdkit.Chem import AllChem

# Periodic table metadata
# Runtime accessing from mendeleev is surprisingly slow, tabulate here
from neuralplexer.data.physical import PGROUP_IDS, PPERIOD_IDS


def mol_to_graph(mol):
    """Convert RDKit Mol to NetworkX graph
    Adapted from https://github.com/deepchem/deepchem

    Convert mol into a graph representation atoms are nodes, and bonds
    are vertices stored as graph

    Parameters
    ----------
    mol: rdkit Mol
      The molecule to convert into a graph.

    Returns
    -------
    graph: networkx.Graph
      Contains atoms indices as nodes, edges as bonds.
    """
    G = nx.DiGraph()
    num_atoms = mol.GetNumAtoms()
    G.add_nodes_from(range(num_atoms))
    if mol.GetNumBonds() == 0:
        # assert num_atoms == 1
        for i in range(mol.GetNumAtoms()):
            G.add_edge(i, i, bond_idx=i)
        return G
    for i in range(mol.GetNumBonds()):
        from_idx = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        to_idx = mol.GetBondWithIdx(i).GetEndAtomIdx()
        G.add_edge(from_idx, to_idx, bond_idx=2 * i)
        G.add_edge(to_idx, from_idx, bond_idx=2 * i + 1)
    return G


def is_potential_stereo_center(atom: Chem.rdchem.Atom):
    """
    Note: Instead of checking whether a real chiral tag can be assigned to the query atom,
    this function regards all neighbor atoms as unique groups.

    Based on rules from https://www.rdkit.org/docs/RDKit_Book.html#brief-description-of-the-findpotentialstereo-algorithm
    """
    assert atom.GetSymbol() != "H"
    if atom.GetDegree() >= 4:
        return True
    elif atom.GetTotalDegree() >= 4:
        return True
    elif atom.GetSymbol() in ["P", "As", "S", "Se"] and atom.GetDegree() >= 3:
        return True
    else:
        return False


def is_stereo_center(atom: Chem.rdchem.Atom):
    assert atom.GetSymbol() != "H"
    if atom.GetChiralTag() in [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]:
        return False
    else:
        return True


BONDORDER = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.AROMATIC: 1.5,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
}


def is_potential_stereo_bond(bond: Chem.rdchem.Bond):
    bond_order = BONDORDER.get(bond.GetBondType(), 1)
    if bond_order > 1.4 and bond_order < 3.0:
        return True
    else:
        return False


@deprecated()
def compute_atom_positional_encodings(
    query_atom_id,
    nx_graph: networkx.DiGraph,
    atom_idx_on_triangle,
):
    """
    Encoding the atom's position on a given bond-pair triangle.
    """
    atom_triangle_pe = [atom_idx_on_triangle[i] == query_atom_id for i in range(3)]
    atom_triangle_pe += [
        atom_idx_on_triangle[i] in nx_graph[query_atom_id].keys() for i in range(3)
    ]
    return np.array(atom_triangle_pe)


def compute_stereo_encodings(
    query_atom_id: int,
    bonded_atom_id: int,
    ref_bond_pos: int,
    mol: Chem.rdchem.Mol,
    nx_graph: networkx.DiGraph,
    atom_idx_on_triangle: tuple,
    ref_geom_xyz,
    cutoff=0.05,
):
    """
    Three types of stereochemistry annotations:
        1. In-clique index of the atom(s) the query atom is bonded to;
        2. On which side of the tangent plane (or unsure);
        3. If off-plane, above or below the plane (or unsure);
    All stereogenic bonds must be explicitly labelled within the input graph.

    :return: torch.Tensor with stereochemistry annotations
    """
    # Is a single atom or not
    stereo_enc = [atom_idx_on_triangle[0] == atom_idx_on_triangle[1]]
    # Is spinor type or not
    stereo_enc += [atom_idx_on_triangle[0] == atom_idx_on_triangle[2]]
    stereo_enc += [atom_idx_on_triangle[i] == query_atom_id for i in range(3)]
    # Whether bonded to other atoms in the triplet
    stereo_enc += [
        atom_idx_on_triangle[i] in nx_graph[query_atom_id].keys() for i in range(3)
    ]
    # stereo_enc += [
    #     mol.GetAtomWithIdx(bonded_atom_id).GetHybridization()
    #     == Chem.rdchem.HybridizationType.SP
    # ]
    stereo_enc += one_of_k_encoding(ref_bond_pos, [0, 1])
    if ref_geom_xyz is None:
        stereo_enc += [False] * 4
        return np.array(stereo_enc)
    in_bond_vec = (
        ref_geom_xyz[atom_idx_on_triangle[1]] - ref_geom_xyz[atom_idx_on_triangle[0]]
    )
    in_vec = in_bond_vec / np.linalg.norm(in_bond_vec)
    out_bond_vec = (
        ref_geom_xyz[atom_idx_on_triangle[2]] - ref_geom_xyz[atom_idx_on_triangle[1]]
    )
    out_vec = out_bond_vec / np.linalg.norm(out_bond_vec)
    z_vec = np.cross(in_vec, out_vec)

    nx_graph[query_atom_id][bonded_atom_id]["bond_idx"] // 2
    ref_bond_id = (
        nx_graph[atom_idx_on_triangle[ref_bond_pos]][
            atom_idx_on_triangle[ref_bond_pos + 1]
        ]["bond_idx"]
        // 2
    )

    # Resolving bond-centered stereochemistry
    # if mol.GetBondWithIdx(ref_bond_id).GetStereo() in [
    #     Chem.rdchem.BondStereo.STEREOCIS,
    #     Chem.rdchem.BondStereo.STEREOTRANS,
    #     Chem.rdchem.BondStereo.STEREOE,
    #     Chem.rdchem.BondStereo.STEREOZ,
    # ]:
    if is_potential_stereo_bond(mol.GetBondWithIdx(ref_bond_id)):
        query_bond_vec = ref_geom_xyz[query_atom_id] - ref_geom_xyz[bonded_atom_id]
        query_bond_vec = query_bond_vec / np.linalg.norm(query_bond_vec)
        if ref_bond_pos == 0:
            ref_bond_vec = in_vec
            if bonded_atom_id == atom_idx_on_triangle[0]:
                query_bond_vec = -query_bond_vec
        elif ref_bond_pos == 1:
            if bonded_atom_id == atom_idx_on_triangle[1]:
                query_bond_vec = -query_bond_vec
            ref_bond_vec = out_vec
        else:
            raise ValueError
        p_z_bond = np.dot(
            np.cross(ref_bond_vec, query_bond_vec),
            z_vec,
        )
        # print("bond p_z:", p_z_bond)
        if np.abs(p_z_bond) > cutoff:
            bond_stereo_enc = one_of_k_encoding(
                p_z_bond > 0,
                [True, False],
            )
        else:
            # Cannot resolve E/Z geometry
            bond_stereo_enc = [False, False]
    else:
        bond_stereo_enc = [False, False]
    stereo_enc += bond_stereo_enc

    # Resolving atom-centered stereochemistry
    if bonded_atom_id == atom_idx_on_triangle[1] and is_potential_stereo_center(
        mol.GetAtomWithIdx(bonded_atom_id)
    ):
        query_bonded_vec = ref_geom_xyz[query_atom_id] - ref_geom_xyz[bonded_atom_id]
        query_bonded_vec = query_bonded_vec / np.linalg.norm(query_bonded_vec)
        p_z_atom = np.dot(query_bonded_vec, z_vec)
        # print("atom p_z:", p_z_atom)
        if np.abs(p_z_atom) > cutoff:
            atom_stereo_enc = one_of_k_encoding(
                p_z_atom > 0,
                [True, False],
            )
        else:
            atom_stereo_enc = [False, False]
    else:
        atom_stereo_enc = [False, False]
    stereo_enc += atom_stereo_enc

    return np.array(stereo_enc)


def get_conformers_as_tensor(mol, n_conf, return_new_mol=False):
    conf_mol = Chem.AddHs(Chem.Mol(mol))
    try:
        AllChem.EmbedMultipleConfs(
            conf_mol, clearConfs=True, numConfs=n_conf, numThreads=0
        )
        assert len(conf_mol.GetConformers()) == n_conf
    except AssertionError:
        AllChem.EmbedMultipleConfs(
            conf_mol,
            clearConfs=True,
            numConfs=n_conf,
            numThreads=0,
            useRandomCoords=True,
        )
        assert len(conf_mol.GetConformers()) == n_conf
    AllChem.MMFFOptimizeMoleculeConfs(conf_mol, mmffVariant="MMFF94", numThreads=0)
    conf_mol = Chem.RemoveHs(conf_mol)
    xyzs = np.array([c.GetPositions() for c in conf_mol.GetConformers()])
    assert xyzs.shape[0] == n_conf
    if return_new_mol:
        return xyzs, conf_mol
    return xyzs


def one_of_k_encoding(x, allowable_set):
    """
    Maps inputs not in the allowable set to the last element.
    modified from https://github.com/XuhanLiu/NGFP
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_encoding(atom: Chem.rdchem.Atom):
    # Periodic table encoding
    encoding_list = (
        one_of_k_encoding(PGROUP_IDS[atom.GetSymbol()], list(range(1, 19)))
        + one_of_k_encoding(PPERIOD_IDS[atom.GetSymbol()], list(range(1, 6)))
        # + one_of_k_encoding(atom.GetDegree(), list(range(7)))
        # + one_of_k_encoding(
        #     atom.GetHybridization(),
        #     [
        #         Chem.rdchem.HybridizationType.SP,
        #         Chem.rdchem.HybridizationType.SP2,
        #         Chem.rdchem.HybridizationType.SP3,
        #         Chem.rdchem.HybridizationType.SP3D,
        #         Chem.rdchem.HybridizationType.SP3D2,
        #     ],
        # )
        # + [atom.GetIsAromatic()]
    )
    return np.array(encoding_list)


def get_bond_encoding(bond):
    bt = bond.GetBondType()
    return np.array(
        [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
        ]
    )


def compute_bond_pair_triangles(nx_graph):
    # Line graph edge trackers
    bab_dict, aaa_dict = dict(), dict()
    in_atoms, mid_atoms, out_atoms = [], [], []
    for atom_id in list(nx_graph):
        neighbor_nodes_array = np.array(nx_graph[atom_id])
        # 07/28: Allow for repeated nodes iff num_atoms < 3
        if len(nx_graph) == 1:
            # Single-atom species
            local_in = np.array([atom_id], dtype=int)
            local_out = np.array([atom_id], dtype=int)
        elif len(nx_graph) == 2:
            # Allow for i=k to handle diatomic molecules
            local_in = np.array(neighbor_nodes_array, dtype=int)
            local_out = np.array(neighbor_nodes_array, dtype=int)
        else:
            # Masking out i=k for others
            nna_rep = np.repeat(
                neighbor_nodes_array[:, np.newaxis], len(neighbor_nodes_array), axis=1
            )
            mask = ~np.eye(nna_rep.shape[0], dtype=bool)
            local_in = nna_rep[mask]
            local_out = nna_rep.transpose()[mask]
        local_mid = np.full_like(local_in, atom_id)
        in_atoms.append(local_in)
        mid_atoms.append(local_mid)
        out_atoms.append(local_out)
    in_atoms = np.concatenate(in_atoms)
    mid_atoms = np.concatenate(mid_atoms)
    out_atoms = np.concatenate(out_atoms)
    in_bonds = np.array(
        [nx_graph[in_atoms[i]][mid_atoms[i]]["bond_idx"] for i in range(len(in_atoms))]
    )
    out_bonds = np.array(
        [
            nx_graph[mid_atoms[i]][out_atoms[i]]["bond_idx"]
            for i in range(len(out_atoms))
        ]
    )
    for triangle_idx in range(len(in_atoms)):
        bab_dict[
            (in_bonds[triangle_idx], mid_atoms[triangle_idx], out_bonds[triangle_idx])
        ] = triangle_idx
        aaa_dict[
            (in_atoms[triangle_idx], mid_atoms[triangle_idx], out_atoms[triangle_idx])
        ] = triangle_idx
    return (
        np.stack([in_atoms, mid_atoms, out_atoms, in_bonds, out_bonds], axis=1),
        bab_dict,
        aaa_dict,
    )


def compute_atom_triangle_pairs(
    nx_graph, max_path_length, atom_idx_on_triangles, truncate=False, subsample=False
):
    at_dict, reachable_atoms = dict(), None
    atom_triangle_pair_list = []
    if truncate:
        reachable_atoms = defaultdict(set)
        shortest_path_lengths = dict(
            nx.all_pairs_shortest_path_length(nx_graph, cutoff=max_path_length)
        )
        for triangle_idx, atom_idx_on_triangle in enumerate(atom_idx_on_triangles):
            k_reachable_sets = [
                shortest_path_lengths[atom_idx_on_triangle[i]].keys() for i in range(3)
            ]
            k_reachable_nodes = (
                k_reachable_sets[0] | k_reachable_sets[1] | k_reachable_sets[2]
            )
            new_pairs = np.stack(
                [
                    np.array(list(k_reachable_nodes)),
                    np.full(len(k_reachable_nodes), triangle_idx),
                ],
                axis=1,
            )
            atom_triangle_pair_list.append(new_pairs)
        atom_triangle_pair_list = np.concatenate(atom_triangle_pair_list, axis=0)
    else:
        reachable_atoms = np.array(nx_graph.nodes, dtype=np.int_)
        new_pairs_node = np.broadcast_to(
            np.arange(len(nx_graph))[:, None],
            (
                len(nx_graph),
                len(atom_idx_on_triangles),
            ),
        )
        new_pairs_tri = np.broadcast_to(
            np.arange(len(atom_idx_on_triangles)),
            (
                len(nx_graph),
                len(atom_idx_on_triangles),
            ),
        )
        atom_triangle_pair_list = np.stack(
            [new_pairs_node.flatten(), new_pairs_tri.flatten()], axis=1
        )
    for prop_idx, at_ids in enumerate(atom_triangle_pair_list):
        at_dict[(at_ids[0], at_ids[1])] = prop_idx
        if truncate:
            reachable_atoms[at_ids[1]].add(at_ids[0])

    return atom_triangle_pair_list, at_dict, reachable_atoms


def compute_all_atom_positional_encodings(
    nx_graph, atom_idx_on_triangles, at_dict=None
):
    triangle_atoms_list, pos_encodings_list = [], []
    for triangle_idx, triangle_atoms in enumerate(atom_idx_on_triangles):
        if triangle_atoms[0] == triangle_atoms[2]:
            candidate_atoms = triangle_atoms[:2]
        else:
            candidate_atoms = triangle_atoms
        for query_atom in candidate_atoms:
            if at_dict is not None:
                # The third entry is the scattering index for u0ijk->Uijk
                triangle_atoms_list.append(
                    [triangle_idx, query_atom, at_dict[(query_atom, triangle_idx)]]
                )
            else:
                triangle_atoms_list.append([triangle_idx, query_atom])
            # triangle_atoms_list.append([triangle_idx, query_atom])
            pos_encodings_list.append(
                compute_atom_positional_encodings(
                    query_atom,
                    nx_graph,
                    triangle_atoms,
                )
            )
    return np.array(triangle_atoms_list), pos_encodings_list


def compute_all_stereo_chemistry_encodings(
    mol, nx_graph, atom_idx_on_triangles, aaa_dict, only_2d=False, ref_conf_xyz=None
):
    if not only_2d:
        if ref_conf_xyz is None:
            ref_conf_xyz = get_conformers_as_tensor(mol, 1)[0]
    else:
        ref_conf_xyz = None
    triangle_pairs_list, stereo_encodings_list = [], []
    for triangle_idx, triangle_atoms in enumerate(atom_idx_on_triangles):
        triangle_atoms = tuple(int(aidx) for aidx in triangle_atoms)
        for query_atom in nx_graph[triangle_atoms[0]].keys():
            if query_atom == triangle_atoms[1]:
                continue
            triangle_pairs_list.append(
                [
                    aaa_dict[triangle_atoms],
                    aaa_dict[(query_atom, triangle_atoms[0], triangle_atoms[1])],
                ]
            )
            stereo_encodings_list.append(
                compute_stereo_encodings(
                    query_atom,
                    triangle_atoms[0],
                    0,
                    mol,
                    nx_graph,
                    triangle_atoms,
                    ref_conf_xyz,
                )
            )
        for query_atom in nx_graph[triangle_atoms[1]].keys():
            if query_atom == triangle_atoms[0]:
                continue
            if query_atom == triangle_atoms[2]:
                continue
            # Outgoing bonds
            triangle_pairs_list.append(
                [
                    aaa_dict[triangle_atoms],
                    aaa_dict[(triangle_atoms[0], triangle_atoms[1], query_atom)],
                ]
            )
            stereo_encodings_list.append(
                compute_stereo_encodings(
                    query_atom,
                    triangle_atoms[1],
                    0,
                    mol,
                    nx_graph,
                    triangle_atoms,
                    ref_conf_xyz,
                )
            )
            # Incoming bonds
            triangle_pairs_list.append(
                [
                    aaa_dict[triangle_atoms],
                    aaa_dict[(query_atom, triangle_atoms[1], triangle_atoms[2])],
                ]
            )
            stereo_encodings_list.append(
                compute_stereo_encodings(
                    query_atom,
                    triangle_atoms[1],
                    1,
                    mol,
                    nx_graph,
                    triangle_atoms,
                    ref_conf_xyz,
                )
            )
        for query_atom in nx_graph[triangle_atoms[2]].keys():
            if query_atom == triangle_atoms[1]:
                continue
            triangle_pairs_list.append(
                [
                    aaa_dict[triangle_atoms],
                    aaa_dict[(triangle_atoms[1], triangle_atoms[2], query_atom)],
                ]
            )
            stereo_encodings_list.append(
                compute_stereo_encodings(
                    query_atom,
                    triangle_atoms[2],
                    1,
                    mol,
                    nx_graph,
                    triangle_atoms,
                    ref_conf_xyz,
                )
            )

    return np.array(triangle_pairs_list), stereo_encodings_list


def get_all_propagator_pairs(
    triangle_pairs_list, reachable_atoms, at_prop_idx_dict, truncate=False
):
    prop_pairs_list = []
    for triangle_pair_idx, triangle_pair in enumerate(triangle_pairs_list):
        triangle_ijk, triangle_jkl = int(triangle_pair[0]), int(triangle_pair[1])
        if truncate:
            reachable_us = reachable_atoms[triangle_ijk].intersection(
                reachable_atoms[triangle_jkl]
            )
        else:
            reachable_us = reachable_atoms
        for edge_u in reachable_us:
            # (incoming, outgoing, ijkl) layout
            prop_pairs_list.append(
                (
                    at_prop_idx_dict[(edge_u, triangle_ijk)],
                    at_prop_idx_dict[(edge_u, triangle_jkl)],
                    triangle_pair_idx,
                )
            )
    return np.array(prop_pairs_list)
