#!/usr/bin/env python

import torch
import argparse

from agents import LSTMActorCriticAgent, LSTMReinforceAgent
from environment import EnvironmentTorch
from utils.dataset_utils import get_questions_msmarco, get_questions_sem_eval
from embeddings import GloveEmbeddingProvider
from utils.agent_utils import run_beam_search
from utils.agent_utils import run_greedy_decoding
from utils.agent_utils import paths_to_context
from utils.agent_utils import compute_metrics
from utils import graph_utils
from knowledge_graph import KnowledgeGraph


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='models/models/msmarco_1.pt')
    parser.add_argument("--dataset", type=str, default='msmarco')
    parser.add_argument("--knowledge_graph", type=str, default='data/causenet-precision.jsonl.bz2')
    parser.add_argument("--embeddings", type=str, default='data/glove.6B.zip')
    parser.add_argument("--test_file", type=str, default='datasets/msmarco_test.json')
    parser.add_argument("--path_len_eval", type=int, default=2)

    parser.add_argument("--hidden_dim_lstm", type=int, default=1024, help='Projection dimension of the LSTM.')
    parser.add_argument("--hidden_dim_mlp", type=int, default=2048,
                        help='Hidden dimension of the two feedforward heads')

    parser.add_argument('--use_question', action='store_true',
                        help='Whether to concatenate the question embedding with the current entity embedding or not.')
    parser.add_argument('--no-use_question', dest='use_question', action='store_false')
    parser.set_defaults(use_question=True)

    parser.add_argument('--use_relation_action', action='store_true',
                        help='Whether to concatenate the sentence embedding with the next entity/action embedding.')
    parser.add_argument('--no-use_relation_action', dest='use_relation_action', action='store_false')
    parser.set_defaults(use_relation_action=True)

    parser.add_argument('--use_critic', action='store_true')
    parser.add_argument('--no-use_critic', dest='use_critic', action='store_false')
    parser.set_defaults(use_critic=True)

    parser.add_argument("--beam_width", type=int, default=50)
    parser.add_argument('--beam_search', action='store_true')
    parser.add_argument('--no-beam_search', dest='beam_search', action='store_false')
    parser.set_defaults(beam_search=True)
    parser.add_argument('--use_inverse', action='store_true')
    parser.add_argument('--no-use_inverse', dest='use_inverse', action='store_false')
    parser.set_defaults(use_inverse=True)
    args = parser.parse_args()
    return args


def main(args):
    triples = graph_utils.get_causenet_triples(args.knowledge_graph, include_source=True)
    provider = GloveEmbeddingProvider(args.embeddings)

    kg = KnowledgeGraph(embedding_provider=provider, triples=triples, use_inverse=args.use_inverse)

    in_dim = kg.num_question_dimensions + kg.num_entity_dimensions if args.use_question else kg.num_entity_dimensions
    if args.use_relation_action:
        out_dim = kg.num_relation_dimensions + kg.num_entity_dimensions
    else:
        out_dim = kg.num_entity_dimensions

    if args.use_critic:
        agent = LSTMActorCriticAgent(input_dim=in_dim,
                                     output_dim=out_dim,
                                     hidden_dim_mlp=args.hidden_dim_mlp,
                                     hidden_dim_lstm=args.hidden_dim_lstm)
    else:
        agent = LSTMReinforceAgent(input_dim=in_dim,
                                   output_dim=out_dim,
                                   hidden_dim_mlp=args.hidden_dim_mlp,
                                   hidden_dim_lstm=args.hidden_dim_lstm)

    agent.load_state_dict(torch.load(args.model, map_location=DEVICE))
    agent.to(DEVICE)

    if args.dataset == 'msmarco':
        valid = get_questions_msmarco(kg, args.test_file, True)
    elif args.dataset == 'semeval':
        valid = get_questions_sem_eval(kg, args.test_file, True)
    else:
        raise ValueError('Use msmarco or semeval as dataset.')

    eval_env = EnvironmentTorch(kg, valid,
                                max_path_len=args.path_len_eval,
                                use_question=args.use_question,
                                use_relation_action=args.use_relation_action,
                                valid_mode=True, debug_mode=False)

    agent.eval()
    if args.beam_search:
        true_labels, predictions, question_candidates, count = run_beam_search(agent, eval_env, DEVICE, args.beam_width)
        paths_to_context(valid, question_candidates, eval_env.graph)
        metrics = compute_metrics(true_labels, predictions)
    else:
        true_labels, predictions, _, count = run_greedy_decoding(agent, eval_env, DEVICE)
        metrics = compute_metrics(true_labels, predictions)

    metrics['nodes'] = count / len(valid)
    print(metrics)


if __name__ == '__main__':
    main(parse_args())
