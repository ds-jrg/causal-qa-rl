import json
import random
import torch

from prepare_datasets import Question
from collections import defaultdict
from typing import List, Union

from knowledge_graph import KnowledgeGraph


class EnvironmentTorch():

    def __init__(self, graph: KnowledgeGraph, questions: List[Question], valid_mode: bool = False,
                 max_path_len: int = 2, max_actions: int = 5000,
                 use_question: bool = False, use_relation_action: bool = True,
                 debug_mode: bool = False):
        self.max_path_len = max_path_len
        self.max_actions = max_actions
        self.use_question = use_question
        self.use_relation_action = use_relation_action

        self.graph = graph
        self.questions = questions
        self.valid_mode = valid_mode

        self.current_question_id = -1
        self.current_node = None
        self.current_context = {}
        self.current_path = []

        self.path_storage = defaultdict(list)
        self.current_batch_path_storage = []
        self._reset_batch_storage = False

        self.debug_mode = debug_mode
        if self.debug_mode:
            self.debug_context = {}
            self.debug_graph.add_edges_from(self.graph.triples)

    def reset(self):
        self.path_storage[self.current_question_id].append(self.current_path)

        if self.current_question_id == -1 or self._reset_batch_storage:
            self.current_batch_path_storage = []
            self._reset_batch_storage = False
        else:
            current_path_entities = [self.graph.id_to_entity(entity) for entity in self.current_path[1:]]
            self.current_batch_path_storage.append((self.questions[self.current_question_id].question,
                                                    current_path_entities))

        self.current_question_id = (self.current_question_id + 1) % len(self.questions)
        if not self.valid_mode and self.current_question_id == 0:
            random.shuffle(self.questions)

        self.current_node = self.graph.entity_to_id(self.questions[self.current_question_id].cause)

        self.current_context = {}
        self.debug_context = {}
        self.current_path = [self.current_node]
        return self._get_observation()

    # Only needed for beam search to quickly change/get the state for a node
    def get_state(self, current_node: Union[str, int], question_id: int):
        temp_node = self.current_node
        temp_question_id = self.current_question_id

        if isinstance(current_node, str):
            self.current_node = self.graph.entity_to_id(current_node)
        else:
            self.current_node = current_node
        self.current_question_id = question_id
        obs = self._get_observation()

        self.current_node = temp_node
        self.current_question_id = temp_question_id
        return obs

    def step(self, action_id):
        next_node = self.graph.neighbour_ids(self.current_node)[action_id]
        if next_node != self.graph.stop_action:
            self.current_node = next_node

        self.current_path.append(self.current_node)
        next_node_entity = self.graph.id_to_entity(self.current_node)

        question = self.questions[self.current_question_id]
        if len(self.current_path) == self.max_path_len + 1:
            if next_node_entity == question.effect and question.binary_answer \
                    or next_node_entity == question.effect and question.binary_answer is None:
                return self._get_observation(), 1.0, True
            # for negative binary questions
            elif self.valid_mode and question.binary_answer is not None \
                    and not question.binary_answer and next_node_entity != question.effect:
                return self._get_observation(), 1.0, True
            else:
                return self._get_observation(), 0.0, True
        else:
            return self._get_observation(), 0.0, False

    def _get_observation(self):
        current_node_emb = self.graph.entity_embeddings[self.current_node]
        # Concatenate question + current node embeddings
        if self.use_question:
            current_question_emb = self.questions[self.current_question_id].embedding
            current_node_emb = torch.cat([current_question_emb, current_node_emb], axis=0)

        neighbours = self.graph.neighbour_ids(self.current_node)
        if len(neighbours) > self.max_actions:
            neighbours = neighbours[:self.max_actions]

        # Concatenate relation + target node embeddings
        if self.use_relation_action:
            neighbours_embeddings = self.graph.get_embeddings_relation_object(self.current_node, neighbours)
            # dim = self.graph.num_entity_dimensions + self.graph.num_relation_dimensions
        else:
            neighbours_embeddings = self.graph.get_embeddings(neighbours)
            # dim = self.graph.num_entity_dimensions
        # neighbours_embeddings = torch.cat([torch.from_numpy(neighbours_embeddings).float(),
        #                                   torch.zeros((self.max_actions-neighbours_embeddings.shape[0] , dim),
        #                                               dtype=torch.float32)], axis=0)

        return current_node_emb.unsqueeze(0), neighbours_embeddings.unsqueeze(0)

    def save_paths(self, count: int = 5, path: str = "paths.json"):
        path_output = {}
        for question_id, paths in self.path_storage.items():
            # dummy question
            if question_id == -1:
                continue
            paths = [[self.graph.id_to_entity(id_) for id_ in path] for path in paths[-count:]]
            path_output[self.questions[question_id].question] = paths

        with open(path, 'w+') as file_:
            json.dump(path_output, file_, indent=4)
