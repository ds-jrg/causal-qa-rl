import random
import torch
import json
import torch.nn.functional as F

from dataclasses import dataclass
from copy import copy
from collections import deque
from typing import List, Any, Optional
from knowledge_graph import KnowledgeGraph
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from agents import LSTMActorCriticAgent
from own import agents


##############
# Beam Search
##############

@dataclass
class Candidate:
    question: str
    path: List[str]
    prob: float
    state: Optional[Any]

    def __str__(self):
        return f'Question: {self.question} | Path: {self.path} | Probability: {self.prob}'

    def __repr__(self):
        return f'Question: {self.question} | Path: {self.path} | Probability: {self.prob}'


def _run_agent(agent, state_ob, state_action, state):
    with torch.no_grad():
        if isinstance(agent, LSTMActorCriticAgent) or isinstance(agent, agents.LSTMActorCriticAgent):
            action_pred, _, agent_state = agent(state_ob, state_action, state)
        else:
            action_pred, agent_state = agent(state_ob, state_action, state)
    action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
    action_prob = torch.log(F.softmax(action_pred, dim=-1).squeeze(0).clamp(min=1e-20)).cpu().numpy()
    return action_prob, agent_state


def random_walk(question, environment, width: int = 5, path_len: int = 2):
    kg = environment.graph
    candidates = []
    for _ in range(width):
        path = [question.cause]
        for _ in range(path_len):
            neighbours = list(kg.neighbour_ids(kg.entity_to_id(path[-1])))
            sample = random.sample(neighbours, 1)[0]
            if sample == kg.stop_action:
                path.append(path[-1])
            else:
                path.append(kg.id_to_entity(sample))
        candidates.append(Candidate(question.question, path, 0.0, None))
        # print(path)

    return candidates, 0


def beam_search(question, agent, environment, device, width: int = 5, path_len: int = 2):
    question_id = question.id_
    question_str = question.question
    paths = [Candidate(question_str, path=[question.cause], prob=0.0, state=agent.get_initial_state(device))]
    nodes = set()
    nodes.add(question.cause)
    found = False

    for _ in range(path_len):
        candidates = []
        for p in paths:
            current = p.path[-1]
            state_ob, state_action = environment.get_state(current, question_id)
            action_prob, agent_state = _run_agent(agent, state_ob.to(device), state_action.to(device), p.state)

            current_end_node_id = environment.graph.entity_to_id(current)
            for idx, prob in enumerate(action_prob):
                entity_id = environment.graph.neighbour_ids(current_end_node_id)[idx]
                if entity_id == environment.graph.stop_action:
                    entity_id = current_end_node_id
                entity_label = environment.graph.id_to_entity(entity_id)
                candidates.append(Candidate(question_str, path=copy(p.path) + [entity_label],
                                  prob=p.prob + float(prob), state=agent_state))

        candidates = sorted(candidates, key=lambda x: x.prob, reverse=True)
        paths = candidates[:width]
        for p in paths:
            if not found:
                nodes.update(p.path)
            if question.effect in nodes:
                found = True

    # Dont need the state anymore
    for p in paths:
        p.state = None
    return paths, len(nodes)


########
# BFS
########


@dataclass
class BFSNode:
    id_: int
    prev_node: 'BFSNode'
    path_len: int


def bfs(kg: KnowledgeGraph, entity1, entity2, max_path_len):
    found = set()
    found.add(entity1)
    q = deque()
    start = BFSNode(entity1, None, 0)
    q.append(start)
    count = 0

    while len(q):
        current_node = q.popleft()
        count += 1
        if kg.id_to_entity(current_node.id_) == kg.id_to_entity(entity2):
            path = backtrack_path(current_node)
            if len(path) - 1 < max_path_len:
                path += [path[-1]] * (max_path_len - (len(path) - 1))
            return path, count
        if current_node.path_len >= max_path_len:
            continue

        for neighbor in kg.neighbour_ids(current_node.id_)[:5000]:
            if neighbor not in found:
                q.append(BFSNode(neighbor, current_node, current_node.path_len + 1))
                found.add(neighbor)
    return None, count


def backtrack_path(node: BFSNode):
    path = []
    while node is not None:
        path.append(node.id_)
        node = node.prev_node
    return list(reversed(path))


#########################
# Save Paths
#########################


def paths_to_context(questions, questions_candidates, graph, num_paths: int = 5):
    contexts = []
    stay = 'stop stop action'
    for question, candidates in zip(questions, questions_candidates):
        question_contexts = {}
        question_contexts['id_'] = question.id_
        candidates = candidates[:num_paths]
        if all(p == stay for c in candidates for p in c.path):
            question_contexts['context_simple'] = ''
            question_contexts['context_sentence'] = ''
        else:
            context_simple = ''
            context_sentence = ''
            for candidate in candidates[:10]:
                for p1, p2 in zip(candidate.path, candidate.path[1::]):
                    if p1 != stay and p2 != stay and p1 != p2:
                        index1 = graph.entity_to_id(p1)
                        index2 = graph.entity_to_id(p2)
                        context_simple += p1 + ' can cause ' + p2 + '. '
                        context_sentence += graph.relation(index1, index2) + ' '
            question_contexts['context_simple'] = context_simple.strip()
            question_contexts['context_sentence'] = context_sentence.strip()
        contexts.append(question_contexts)

    with open('msmarco_context.jsonl', 'w+') as f:
        for context in contexts:
            f.write(json.dumps(context) + '\n')


def compute_metrics(labels, predictions):
    assert len(labels) == len(predictions)
    metrics = {}
    metrics['accuracy'] = round(accuracy_score(labels, predictions), 3)
    metrics['f1_score'] = round(f1_score(labels, predictions, average='binary'), 3)
    metrics['recall'] = round(recall_score(labels, predictions, average='binary'), 3)
    metrics['precision'] = round(precision_score(labels, predictions, average='binary'), 3)
    return metrics


def run_beam_search(agent, eval_env, device, width: int = 50):
    question_candidates = []
    true_labels = []
    predictions = []
    eval_env.current_question_id = -1
    count = 0
    for question in eval_env.questions:
        true_labels.append(int(question.binary_answer))
        candidates, c = beam_search(question, agent, eval_env, device,
                                    width=width,
                                    path_len=eval_env.max_path_len)
        count += c
        question_candidates.append(candidates)
        pred = int(any(p == question.effect for candidate in candidates for p in candidate.path))
        predictions.append(pred)
    return true_labels, predictions, question_candidates, count


def run_greedy_decoding(agent, eval_env, device):
    true_labels = []
    predictions = []
    eval_env.current_question_id = -1
    nodes = set()
    for question in eval_env.questions:
        true_labels.append(int(question.binary_answer))
        state_ob, state_action = eval_env.reset()
        agent_state = agent.get_initial_state(device)
        while True:
            with torch.no_grad():
                if isinstance(agent, LSTMActorCriticAgent) or isinstance(agent, agents.LSTMActorCriticAgent):
                    action_pred, _, agent_state = agent(state_ob.to(device),
                                                        state_action.to(device),
                                                        agent_state)
                else:
                    action_pred, agent_state = agent(state_ob.to(device),
                                                     state_action.to(device),
                                                     agent_state)
                action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
                action_prob = F.softmax(action_pred, dim=-1).cpu()
            action = torch.argmax(action_prob, dim=-1)
            (state_ob, state_action), _, done = eval_env.step(action.item())
            if done:
                break
        current_path_entities = [eval_env.graph.id_to_entity(entity)
                                 for entity in eval_env.current_path[1:]]
        nodes.update(current_path_entities)
        predictions.append(int(question.effect in current_path_entities))
    return true_labels, predictions, len(nodes)
