import numpy as np
import pickle
import torch

from typing import List, Tuple, Iterable, Union
from embeddings import EmbeddingProvider

from utils import graph_utils


class KnowledgeGraph:

    def __init__(self, embedding_provider: EmbeddingProvider,
                 triples: Union[List[Tuple[str, str]], str],
                 entity_embedding_path: str = None, relation_embedding_path: str = None,
                 use_inverse: bool = False):

        self.embedding_provider = embedding_provider

        if isinstance(triples, str):
            with open(triples, 'rb') as triple_file:
                self.triples = pickle.load(triple_file)
        else:
            self.triples = triples

        self.entity_index = graph_utils.get_entity_index(triples)
        self.graph, graph_sources, self.graph_sources_full = (
            graph_utils.get_adjacency_list(triples, self.entity_index, use_inverse=use_inverse)
        )
        self.stop_action = len(self.entity_index) - 1

        if entity_embedding_path is not None:
            with open(entity_embedding_path, 'rb') as entity_file:
                self.entity_embeddings = pickle.load(entity_file)
        else:
            self.entity_embeddings = embedding_provider.entity_embeddings(list(self.entity_index.values()))
            # with open('entity_embeddings_causenet_precision_glove.pkl', 'wb') as file_:
            #    pickle.dump(self.entity_embeddings, file_)

        if relation_embedding_path is not None:
            with open(relation_embedding_path, 'rb') as relation_file:
                self.relation_embeddings = pickle.load(relation_file)
        else:
            self.relation_embeddings = embedding_provider.relation_embeddings(graph_sources)
            # with open('relation_embeddings_causenet_precision_glove.pkl', 'wb') as file_:
            #    pickle.dump(self.relation_embeddings, file_)

    def id_to_entity(self, entity_id: int) -> str:
        return self.entity_index[entity_id]

    def entity_to_id(self, entity: str) -> int:
        return self.entity_index.inverse[entity]

    def ids_to_triple(self, triple: Tuple[int, int]) -> Tuple[str, str]:
        return self.entity_index[triple[0]], self.entity_index[triple[1]]

    def relation(self, index1: int, index2: int) -> Tuple[str, str]:
        return self.graph_sources_full[(index1, index2)]

    def triples_to_ids(self, triple: Tuple[str, str]) -> Tuple[int, int]:
        return self.entity_index.inverse[triple[0]], self.entity_index.inverse[triple[1]]

    def neighbour_ids(self, entity_id: int) -> List[int]:
        return self.graph[entity_id]

    def neighbour_entities(self, entity_id: int) -> Iterable[int]:
        return map(self.id_to_entity, self.graph[entity_id])

    def get_embeddings(self, entity_ids: List[int]) -> np.array:
        return self.entity_embeddings[entity_ids]

    def get_embeddings_relation(self, current_node: int, nbs_ids: List[int]) -> np.array:
        relation_embs = [self.relation_embeddings[(current_node, nbs_id)] for nbs_id in nbs_ids]
        return relation_embs

    def get_embeddings_relation_object(self, current_node: int, nbs_ids: List[int]) -> np.array:
        nbs_embs = self.entity_embeddings[nbs_ids]
        relation_embs = torch.stack([self.relation_embeddings[(current_node, nbs_id)] for nbs_id in nbs_ids])
        embs = torch.cat([relation_embs, nbs_embs], axis=1)
        # embs = (nbs_embs + np.array(relation_embs)) / 2
        return embs

    @property
    def num_entity_dimensions(self):
        return self.embedding_provider.num_entity_dimensions

    @property
    def num_relation_dimensions(self):
        return self.embedding_provider.num_relation_dimensions

    @property
    def num_question_dimensions(self):
        return self.embedding_provider.num_question_dimensions
