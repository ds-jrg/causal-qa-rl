import random
import torch
import numpy as np
import json
import torch.nn.functional as F

from dataclasses import dataclass
from collections import deque
from typing import List, Any, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


##############
# Beam Search
##############

@dataclass
class Candidate:
    question: str
    path: List[str]
    prob: float
    state: Optional[Any]
    prev_id: Optional[Any]

    def __str__(self):
        return f'Question: {self.question} | Path: {self.path} | Probability: {self.prob}'

    def __repr__(self):
        return f'Question: {self.question} | Path: {self.path} | Probability: {self.prob}'


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
        candidates.append(Candidate(question.question, path, 0.0, None, None))

    return candidates, 0


def _run_agent(agent, state_ob, state_action, state):
    with torch.no_grad():
        action_pred, _, agent_state = agent(state_ob, state_action, state)
    action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
    action_prob = torch.log(F.softmax(action_pred, dim=-1).clamp(min=np.finfo(np.float32).eps)).cpu().numpy()
    return action_prob, agent_state


def beam_search(question, agent, environment, device, width: int = 5, path_len: int = 2):
    question_id = question.id_
    question_str = question.question
    paths = [Candidate(question_str, path=[question.cause], prob=0.0, state=agent.get_initial_state(device), prev_id=0)]
    nodes = set()
    nodes.add(question.cause)
    found = False

    for _ in range(path_len):
        candidates = []
        for p in paths:
            current = p.path[-1]
            state_ob, state_action = environment.get_state(current, question_id)
            action_prob, agent_state = _run_agent(agent, state_ob.view(1, 1, -1).to(device),
                                                  state_action.view(1, 1, *state_action.shape).to(device), p.state)

            current_end_node_id = environment.graph.entity_to_id(current)
            action_prob = action_prob[:len(environment.graph.neighbour_ids(current_end_node_id))]
            for idx, prob in enumerate(action_prob):
                entity_id = environment.graph.neighbour_ids(current_end_node_id)[idx]
                if entity_id == environment.graph.stop_action:
                    entity_id = current_end_node_id
                entity_label = environment.graph.id_to_entity(entity_id)
                candidates.append(Candidate(question_str, path=p.path + [entity_label],
                                  prob=p.prob + float(prob), state=agent_state, prev_id=0))

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


def beam_search2(question, agent, environment, device, width: int = 5, path_len: int = 2):
    question_id = question.id_
    question_str = question.question
    paths = [Candidate(question_str, path=[question.cause], prob=0.0, state=None, prev_id=0)]
    nodes = set()
    nodes.add(question.cause)
    found = False

    observations = torch.zeros((width, 1, 600), device=device)
    actions_tensors = torch.zeros((width, 1, environment.max_actions, 600), device=device)
    states_h = torch.zeros((1, width, 1024), device=device)
    states_c = torch.zeros((1, width, 1024), device=device)
    (h, c) = agent.get_initial_state(device)

    for _ in range(path_len):
        current_nodes = []
        for idx, p in enumerate(paths):
            current = p.path[-1]
            current_nodes.append(current)
            state_ob, state_action = environment.get_state(current, question_id)
            observations[idx].copy_(state_ob)
            actions_tensors[idx].copy_(state_action)
            states_h[0][idx].copy_(h[0][p.prev_id])
            states_c[0][idx].copy_(c[0][p.prev_id])

        action_prob, (h, c) = _run_agent(agent, observations, actions_tensors, (states_h, states_c))

        candidates = []
        for path_idx, p in enumerate(paths):
            current = current_nodes[path_idx]
            current_end_node_id = environment.graph.entity_to_id(current)

            probs = action_prob[path_idx]
            probs = probs[:len(environment.graph.neighbour_ids(current_end_node_id))]
            for idx, prob in enumerate(probs):
                entity_id = environment.graph.neighbour_ids(current_end_node_id)[idx]
                if entity_id == environment.graph.stop_action:
                    entity_id = current_end_node_id
                entity_label = environment.graph.id_to_entity(entity_id)
                candidates.append(Candidate(question_str, path=p.path + [entity_label], prob=p.prob + float(prob),
                                            state=None, prev_id=path_idx))

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


def bfs(kg, entity1, entity2, max_path_len, max_actions=5000):
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

        for neighbor in kg.neighbour_ids(current_node.id_)[:max_actions]:
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

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    metrics['tp'] = round(tp, 3)
    metrics['fn'] = round(fn, 3)
    metrics['fp'] = round(fp, 3)
    metrics['tn'] = round(tn, 3)

    return metrics


def run_beam_search(agent, eval_env, device, width: int = 50):
    true_labels = []
    predictions = []
    question_candidates = []
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
    question_candidates = []
    eval_env.current_question_id = -1
    nodes = set()
    for question in eval_env.questions:
        true_labels.append(int(question.binary_answer))

        path_prob = 0
        state_ob, state_action = eval_env.reset()
        agent_state = agent.get_initial_state(device)
        done = False
        while not done:
            action_prob, agent_state = _run_agent(agent, state_ob.view(1, 1, -1).to(device),
                                                  state_action.view(1, 1, *state_action.shape).to(device), agent_state)
            action = np.argmax(action_prob, axis=-1)
            path_prob += np.max(action_prob, axis=-1)
            (state_ob, state_action), _, done = eval_env.step(action.item())

        current_path_entities = [eval_env.graph.id_to_entity(entity)
                                 for entity in eval_env.current_path]
        question_candidates.append([Candidate(question.question, current_path_entities, path_prob, None, 0)])
        nodes.update(current_path_entities)
        predictions.append(int(question.effect in current_path_entities))

    return true_labels, predictions, question_candidates, len(nodes)


def grad_norm_check(agent):
    total_norm = 0
    parameters = [p for p in agent.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
