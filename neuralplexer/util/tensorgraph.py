from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class Relation:
    edge_type: str
    edge_rev_name: str
    edge_frd_name: str
    src_node_type: str
    dst_node_type: str
    num_edges: int


class MultiRelationGraphBatcher:
    """
    Collate sub-graphs of different node/edge types into a single instance.
    Returned multi-relation edge indices are stored in LongTensor of shape [2, N_edges].
    """

    def __init__(
        self,
        relation_forms: List[Relation],
        graph_metadata: Dict[str, int],
    ):
        self._relation_forms = relation_forms
        self._make_offset_dict(graph_metadata)

    def _make_offset_dict(self, graph_metadata):
        self._node_chunk_sizes = {}
        self._edge_chunk_sizes = {}
        self._offsets_lower = {}
        self._offsets_upper = {}
        all_node_types = set()
        for relation in self._relation_forms:
            assert (
                f"num_{relation.src_node_type}" in graph_metadata.keys()
            ), f"Missing metadata: num_{relation.src_node_type}"
            assert (
                f"num_{relation.dst_node_type}" in graph_metadata.keys()
            ), f"Missing metadata: num_{relation.src_node_type}"
            all_node_types.add(relation.src_node_type)
            all_node_types.add(relation.dst_node_type)
        offset = 0
        # Fix node type ordering
        self.all_node_types = list(all_node_types)
        for node_type in self.all_node_types:
            self._offsets_lower[node_type] = offset
            self._node_chunk_sizes[node_type] = graph_metadata[f"num_{node_type}"]
            new_offset = offset + self._node_chunk_sizes[node_type]
            self._offsets_upper[node_type] = new_offset
            offset = new_offset

    def collate_single_relation_graphs(self, indexer, node_attr_dict, edge_attr_dict):
        return {
            "node_attr": self.collate_node_attr(node_attr_dict),
            "edge_attr": self.collate_edge_attr(edge_attr_dict),
            "edge_index": self.collate_idx_list(indexer),
        }

    def collate_idx_list(
        self,
        indexer: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ret_eidxs_rev, ret_eidxs_frd = [], []
        for relation in self._relation_forms:
            assert relation.edge_rev_name in indexer.keys()
            assert relation.edge_frd_name in indexer.keys()
            assert indexer[relation.edge_rev_name].dim() == 1
            assert indexer[relation.edge_frd_name].dim() == 1
            assert torch.all(
                indexer[relation.edge_rev_name]
                < self._node_chunk_sizes[relation.src_node_type]
            ), f"Node index on edge exceeding boundary: {relation.edge_type}, {self._node_chunk_sizes[relation.src_node_type]}, {self._node_chunk_sizes[relation.dst_node_type]}, {max(indexer[relation.edge_rev_name])}, {max(indexer[relation.edge_frd_name])}"
            assert torch.all(
                indexer[relation.edge_frd_name]
                < self._node_chunk_sizes[relation.dst_node_type]
            ), f"Node index on edge exceeding boundary: {relation.edge_type}, {self._node_chunk_sizes[relation.src_node_type]}, {self._node_chunk_sizes[relation.dst_node_type]}, {max(indexer[relation.edge_rev_name])}, {max(indexer[relation.edge_frd_name])}"
            ret_eidxs_rev.append(
                indexer[relation.edge_rev_name]
                + self._offsets_lower[relation.src_node_type]
            )
            ret_eidxs_frd.append(
                indexer[relation.edge_frd_name]
                + self._offsets_lower[relation.dst_node_type]
            )
        ret_eidxs_rev = torch.cat(ret_eidxs_rev, dim=0)
        ret_eidxs_frd = torch.cat(ret_eidxs_frd, dim=0)
        return torch.stack([ret_eidxs_rev, ret_eidxs_frd], dim=0)

    def collate_node_attr(self, node_attr_dict: Dict[str, torch.Tensor]):
        for node_type in self.all_node_types:
            assert (
                node_attr_dict[node_type].shape[0] == self._node_chunk_sizes[node_type]
            ), f"Node count mismatch: {node_type}, {node_attr_dict[node_type].shape[0]}, {self._node_chunk_sizes[node_type]}"
        return torch.cat(
            [node_attr_dict[node_type] for node_type in self.all_node_types], dim=0
        )

    def collate_edge_attr(self, edge_attr_dict: Dict[str, torch.Tensor]):
        # for relation in self._relation_forms:
        #     print(relation.edge_type, edge_attr_dict[relation.edge_type].shape)
        return torch.cat(
            [edge_attr_dict[relation.edge_type] for relation in self._relation_forms],
            dim=0,
        )

    def zero_pad_edge_attr(
        self,
        edge_attr_dict: Dict[str, torch.Tensor],
        embedding_dim: int,
        device: torch.device,
    ):
        for relation in self._relation_forms:
            if edge_attr_dict[relation.edge_type] is None:
                edge_attr_dict[relation.edge_type] = torch.zeros(
                    (relation.num_edges, embedding_dim),
                    device=device,
                )
        return edge_attr_dict

    def offload_node_attr(self, cat_node_attr: torch.Tensor):
        node_chunk_sizes = [
            self._node_chunk_sizes[node_type] for node_type in self.all_node_types
        ]
        node_attr_split = torch.split(cat_node_attr, node_chunk_sizes)
        return {
            self.all_node_types[i]: node_attr_split[i]
            for i in range(len(self.all_node_types))
        }

    def offload_edge_attr(self, cat_edge_attr: torch.Tensor):
        edge_chunk_sizes = [relation.num_edges for relation in self._relation_forms]
        edge_attr_split = torch.split(cat_edge_attr, edge_chunk_sizes)
        return {
            self._relation_forms[i].edge_type: edge_attr_split[i]
            for i in range(len(self._relation_forms))
        }


def make_multi_relation_graph_batcher(
    list_of_relations: List[Tuple[str, str, str, str, str]],
    indexer,
    metadata,
):
    # Use one instantiation of the indexer to compute chunk sizes
    relation_forms = [
        Relation(
            edge_type=rl_tuple[0],
            edge_rev_name=rl_tuple[1],
            edge_frd_name=rl_tuple[2],
            src_node_type=rl_tuple[3],
            dst_node_type=rl_tuple[4],
            num_edges=indexer[rl_tuple[1]].shape[0],
        )
        for rl_tuple in list_of_relations
    ]
    return MultiRelationGraphBatcher(
        relation_forms,
        metadata,
    )
