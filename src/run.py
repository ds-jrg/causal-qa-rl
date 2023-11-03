#!/usr/bin/env python

import torch
import wandb
import argparse
import numpy as np
import random

from environment import EnvironmentTorch
from agents import LSTMActorCriticAgent, LSTMReinforceAgent
from prepare_datasets import get_questions_csv, get_questions_msmarco, get_questions_sem_eval
from trainer import A2CTrainer, REINFORCETrainer
from utils import graph_utils
from knowledge_graph import KnowledgeGraph
from embeddings import DebugEmbeddingProvider, GloveEmbeddingProvider


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='msmarco', help='Dataset to use: msmarco or semeval')
    parser.add_argument("--knowledge_graph", type=str, default='data/causenet-precision.jsonl.bz2')
    parser.add_argument("--embeddings", type=str, default='data/glove.6B.zip')
    parser.add_argument("--train_file", type=str, default='datasets/msmarco_test.json')
    parser.add_argument("--test_file", type=str, default='datasets/msmarco_test.json')
    parser.add_argument('--debug', action='store_true',
                        help='Uses causenet-sample and a toy dataset for faster loading for debugging.')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=False)

    parser.add_argument('--use_inverse', action='store_true', help='Whether to add inverse edges to the graph.')
    parser.add_argument('--no-use_inverse', dest='use_inverse', action='store_false')
    parser.set_defaults(use_inverse=True)

    parser.add_argument('--use_actor_critic', action='store_true',
                        help='Whether to use Actor-Critic. If False then REINFORCE is used.')
    parser.add_argument('--no-use_actor_critic', dest='use_actor_critic', action='store_false')
    parser.set_defaults(use_actor_critic=True)

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

    parser.add_argument('--use_full_path', action='store_true',
                        help='Whether to check the whole path for the target entity or the last node.')
    parser.add_argument('--no-use_full_path', dest='use_full_path', action='store_false')
    parser.set_defaults(use_full_path=True)

    parser.add_argument("--steps", type=int, default=1000, help='Number of train steps.')

    parser.add_argument('--supervised', action='store_true', help='Whether to apply supervised learning at the start.')
    parser.add_argument('--no-supervised', dest='supervised', action='store_false')
    parser.set_defaults(supervised=False)

    parser.add_argument('--normalize_returns', action='store_true', help='Whether to normalize the returns/advantages.')
    parser.add_argument('--no-normalize_returns', dest='normalize_returns', action='store_false')
    parser.set_defaults(normalize_returns=False)

    parser.add_argument('--beam_search', action='store_true',
                        help='Whether to use beam search, if False then greedy decoding is used.')
    parser.add_argument('--no-beam_search', dest='beam_search', action='store_false')
    parser.set_defaults(beam_search=True)

    parser.add_argument("--eval_interval", type=int, default=10, help='Interval in train steps for evaluation/logging.')
    parser.add_argument("--batch_size", type=int, default=128, help='Batch size for policy gradient training.')
    parser.add_argument("--path_len_train", type=int, default=2, help='Path rollout length for training.')
    parser.add_argument("--path_len_eval", type=int, default=2, help='Path rollout length for evaluation.')
    parser.add_argument("--beam_width", type=int, default=50, help='Beam width during inference time.')
    parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate for AdamW.')
    parser.add_argument("--entropy_weight", type=float, default=0.01, help='Weight of the entropy regularization.')
    parser.add_argument("--discount", type=float, default=0.99, help='Discount factor.')
    parser.add_argument("--supervised_ratio", type=float, default=0.5,
                        help='Ratio of train questions used for supervised learning.')
    parser.add_argument("--supervised_steps", type=int, default=100, help='Supervised train steps.')
    parser.add_argument("--supervised_batch_size", type=int, default=32, help='Batch size for supervised learning.')
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help='Maximum norm for gradient clipping.')
    parser.add_argument("--lambda_gae", type=float, default=0.95, help='GAE lambda.')
    parser.add_argument("--seed", type=int, default=42, help='Seed.')

    parser.add_argument('--use_gae', action='store_true', help='Whether to use GAE or not.')
    parser.add_argument('--no-use_gae', dest='use_gae', action='store_false')
    parser.set_defaults(use_gae=True)

    parser.add_argument('--use_advantage', action='store_true')
    parser.add_argument('--no-use_advantage', dest='use_advantage', action='store_false')
    parser.set_defaults(use_advantage=True)

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--no-use_wandb', dest='use_wandb', action='store_false')
    parser.set_defaults(use_wandb=False)
    parser.add_argument("--wandb_project", type=str, default='')
    parser.add_argument("--wandb_entity", type=str, default='')
    parser.add_argument("--name", type=str, default='msmarco', help='Name for the run on wandb.')

    return parser.parse_args()


def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    wandb_conf_dict = {
        'dataset': args.dataset,
        'steps': args.steps,
        'use_question': args.use_question,
        'use_relation_action': args.use_relation_action,
        'use_full_path': args.use_full_path,
        'use_inverse': args.use_inverse,
        'actor_critic': args.use_actor_critic,
        'hidden_dim_mlp': args.hidden_dim_mlp,
        'hidden_dim_lstm': args.hidden_dim_lstm,
        'eval_interval': args.eval_interval,
        'supervised': args.supervised,
        'batch_size': args.batch_size,
        'normalize_returns': args.normalize_returns,
        'path_length_train': args.path_len_train,
        'path_length_eval': args.path_len_eval,
        'beam_width': args.beam_width,
        'lr': args.lr,
        'beam_search': args.beam_search,
        'discount': args.discount,
        'entropy_weight': args.entropy_weight,
        'supervised_ratio': args.supervised_ratio,
        'supervised_steps': args.supervised_steps,
        'supervised_batch_size': args.supervised_batch_size,
        'max_grad_norm': args.max_grad_norm,
        'lambda_gae': args.lambda_gae,
    }

    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=wandb_conf_dict,
            name=args.name,
        )

    if args.debug:
        triples = graph_utils.get_causenet_triples(args.knowledge_graph, include_source=False)
        kg = KnowledgeGraph(embedding_provider=DebugEmbeddingProvider(),
                            triples=triples,
                            entity_embedding_path=None,
                            relation_embedding_path=None,
                            use_inverse=args.use_inverse)
        train, valid = get_questions_csv(kg, args.train_file)
    else:
        triples = graph_utils.get_causenet_triples(args.knowledge_graph, include_source=True)
        provider = GloveEmbeddingProvider(args.embeddings)
        kg = KnowledgeGraph(embedding_provider=provider,
                            triples=triples,
                            entity_embedding_path=None,
                            relation_embedding_path=None,
                            use_inverse=args.use_inverse)
        if args.dataset == 'msmarco':
            train = get_questions_msmarco(kg, args.train_file, False)
            valid = get_questions_msmarco(kg, args.test_file, True)
        elif args.dataset == 'semeval':
            train = get_questions_sem_eval(kg, args.train_file, False)
            valid = get_questions_sem_eval(kg, args.test_file, True)

    print('Train Questions: ', len(train))
    print('Valid Questions: ', len(valid))

    train_env = EnvironmentTorch(kg, train,
                                 max_path_len=args.path_len_train,
                                 use_question=args.use_question,
                                 use_relation_action=args.use_relation_action,
                                 valid_mode=False, debug_mode=False)
    eval_env = EnvironmentTorch(kg, valid,
                                max_path_len=args.path_len_eval,
                                use_question=args.use_question,
                                use_relation_action=args.use_relation_action,
                                valid_mode=True, debug_mode=False)

    in_dim = kg.num_question_dimensions + kg.num_entity_dimensions if args.use_question else kg.num_entity_dimensions
    if args.use_relation_action:
        out_dim = kg.num_relation_dimensions + kg.num_entity_dimensions
    else:
        out_dim = kg.num_entity_dimensions

    if args.use_actor_critic:
        agent = LSTMActorCriticAgent(input_dim=in_dim,
                                     output_dim=out_dim,
                                     hidden_dim_mlp=args.hidden_dim_mlp,
                                     hidden_dim_lstm=args.hidden_dim_lstm)
        agent.to(DEVICE)
        agent.weight_init()
        trainer = A2CTrainer(train_env, eval_env, agent, DEVICE,
                             use_wandb=args.use_wandb,
                             lr=args.lr,
                             batch_size=args.batch_size,
                             eval_interval=args.eval_interval,
                             eval_episodes=len(valid),
                             supervised=args.supervised,
                             use_full_path=args.use_full_path,
                             beam_search=args.beam_search,
                             beam_width=args.beam_width,
                             supervised_ratio=args.supervised_ratio,
                             supervised_steps=args.supervised_steps,
                             supervised_batch_size=args.supervised_batch_size,
                             beta_entropy=args.entropy_weight,
                             max_grad_norm=args.max_grad_norm,
                             discount=args.discount,
                             normalize_returns=args.normalize_returns,
                             lambda_gae=args.lambda_gae,
                             use_advantage=args.use_advantage,
                             use_gae=args.use_gae)
    else:
        agent = LSTMReinforceAgent(input_dim=in_dim,
                                   output_dim=out_dim,
                                   hidden_dim_mlp=args.hidden_dim_mlp,
                                   hidden_dim_lstm=args.hidden_dim_lstm)
        agent.to(DEVICE)
        agent.weight_init()
        trainer = REINFORCETrainer(train_env, eval_env, agent, DEVICE,
                                   use_wandb=args.use_wandb,
                                   lr=args.lr,
                                   batch_size=args.batch_size,
                                   eval_interval=args.eval_interval,
                                   eval_episodes=len(valid),
                                   supervised=args.supervised,
                                   use_full_path=args.use_full_path,
                                   supervised_ratio=args.supervised_ratio,
                                   supervised_steps=args.supervised_steps,
                                   supervised_batch_size=args.supervised_batch_size,
                                   beam_search=args.beam_search,
                                   beam_width=args.beam_width,
                                   max_grad_norm=args.max_grad_norm,
                                   beta_entropy=args.entropy_weight,
                                   discount=args.discount,
                                   normalize_returns=args.normalize_returns)

    trainer.train(steps=args.steps)
    eval_env.save_paths()

    torch.save(agent, f'data/models/{args.name}.pt')
    if args.use_wandb:
        artifact = wandb.Artifact(args.name, type='model')
        artifact.add_file(f'data/models/{args.name}.pt')
        run.log_artifact(artifact)


if __name__ == '__main__':
    main(parse_args())
