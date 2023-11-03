#!/usr/bin/env python

import torch
import argparse

from environment import EnvironmentTorch
from prepare_datasets import get_questions_msmarco, get_questions_sem_eval
from embeddings import GloveEmbeddingProvider
from utils.utils_agent import run_beam_search
from utils.utils_agent import run_greedy_decoding
from utils.utils_agent import paths_to_context
from utils.utils_agent import compute_metrics
from utils import graph_utils
from knowledge_graph import KnowledgeGraph


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='models/msmarco_2.pt')
    parser.add_argument("--dataset", type=str, default='msmarco')
    parser.add_argument("--knowledge_graph", type=str, default='data/causenet-precision.jsonl.bz2')
    parser.add_argument("--embeddings", type=str, default='data/glove.6B.zip')
    parser.add_argument("--test_file", type=str, default='datasets/msmarco_test.json')
    parser.add_argument("--path_len_eval", type=int, default=2)
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
    agent = torch.load(args.model, map_location=DEVICE)
    triples = graph_utils.get_causenet_triples(args.knowledge_graph, include_source=True)
    provider = GloveEmbeddingProvider(args.embeddings)

    kg = KnowledgeGraph(embedding_provider=provider,
                        triples=triples,
                        entity_embedding_path=None,
                        relation_embedding_path=None,
                        use_inverse=args.use_inverse,
                        )

    if args.dataset == 'msmarco':
        valid = get_questions_msmarco(kg, args.test_file, True)
    elif args.dataset == 'semeval':
        valid = get_questions_sem_eval(kg, args.test_file, True)
    else:
        raise ValueError('Use msmarco or semeval as dataset.')

    eval_env = EnvironmentTorch(kg, valid,
                                max_path_len=args.path_len_eval,
                                use_question=True,
                                use_relation_action=True,
                                valid_mode=True, debug_mode=False)

    agent.eval()
    if args.beam_search:
        true_labels, predictions, question_candidates, count = run_beam_search(agent, eval_env, DEVICE, args.beam_width)
        paths_to_context(valid, question_candidates, eval_env.graph)
        metrics = compute_metrics(true_labels, predictions)
    else:
        true_labels, predictions, count = run_greedy_decoding(agent, eval_env, DEVICE)
        metrics = compute_metrics(true_labels, predictions)

    # with open("model.txt", "w") as f:
    #    f.write(str(predictions))

    metrics['nodes'] = count / len(valid)
    print(metrics)


if __name__ == '__main__':
    main(parse_args())
