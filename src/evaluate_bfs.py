#!/usr/bin/env python

import argparse
from utils.dataset_utils import get_questions_msmarco, get_questions_sem_eval
from utils import graph_utils
from embeddings import GloveEmbeddingProvider
from knowledge_graph import KnowledgeGraph
from utils.agent_utils import bfs, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='msmarco')
    parser.add_argument("--knowledge_graph", type=str, default='data/causenet-precision.jsonl.bz2')
    parser.add_argument("--embeddings", type=str, default='data/glove.6B.zip')
    parser.add_argument("--test_file", type=str, default='datasets/msmarco_test.json')
    parser.add_argument("--path_len", type=int, default=2)
    args = parser.parse_args()
    return args


def main(args):
    triples = graph_utils.get_causenet_triples(args.knowledge_graph, include_source=True)
    provider = GloveEmbeddingProvider(args.embeddings)
    kg = KnowledgeGraph(embedding_provider=provider, triples=triples, use_inverse=True)

    if args.dataset == 'msmarco':
        valid = get_questions_msmarco(kg, args.test_file, True)
    elif args.dataset == 'semeval':
        valid = get_questions_sem_eval(kg, args.test_file, True)
    else:
        raise ValueError('Use msmarco or semeval as dataset.')

    labels = []
    preds = []
    c = 0
    for q in valid:
        labels.append(int(q.binary_answer))
        if q.effect is None:
            preds.append(0)
            continue
        path, count = bfs(kg, kg.entity_to_id(q.cause), kg.entity_to_id(q.effect), args.path_len)
        c += count
        if path is not None:
            preds.append(1)
        else:
            preds.append(0)

    metrics = compute_metrics(labels, preds)
    metrics['nodes'] = c / len(valid)
    print(metrics)


if __name__ == '__main__':
    main(parse_args())
