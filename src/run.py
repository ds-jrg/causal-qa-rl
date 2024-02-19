#!/usr/bin/env python

import torch
import wandb

from agents import LSTMActorCriticAgent, LSTMReinforceAgent
from trainer import A2CTrainer, Trainer, PPOTrainer
from utils import graph_utils
from utils import config
from knowledge_graph import KnowledgeGraph
from embeddings import DebugEmbeddingProvider, GloveEmbeddingProvider, BERTEmbeddingProvider


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def main():
    args, wandb_conf_dict, trainer_arguments = config.setup()

    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=wandb_conf_dict,
            name=args.name,
        )

    if args.debug:
        triples = graph_utils.get_causenet_triples('datasets/causenet-sample.json', include_source=False)
        provider = DebugEmbeddingProvider()
    else:
        triples = graph_utils.get_causenet_triples(args.knowledge_graph, include_source=True)

        if args.embeddings == 'glove':
            provider = GloveEmbeddingProvider(args.glove_embeddings)
        elif args.embeddings == 'lm':
            provider = BERTEmbeddingProvider(args.lm_entity_embedding_path,
                                             args.lm_relation_embedding_path,
                                             args.lm_train_questions_embedding_path,
                                             args.lm_test_questions_embedding_path)
        else:
            raise ValueError('Use one of glove, lm as embeddings')

    kg = KnowledgeGraph(embedding_provider=provider, triples=triples, use_inverse=args.use_inverse)

    train_env = config.build_train_environment(args, kg)
    eval_envs = config.build_test_environments(args, kg)

    in_dim = kg.num_question_dimensions + kg.num_entity_dimensions if args.use_question else kg.num_entity_dimensions
    if args.use_relation_action:
        out_dim = kg.num_relation_dimensions + kg.num_entity_dimensions
    else:
        out_dim = kg.num_entity_dimensions

    if args.algorithm == 'a2c' or args.algorithm == 'ppo':
        agent = LSTMActorCriticAgent(input_dim=in_dim,
                                     output_dim=out_dim,
                                     hidden_dim_mlp=args.hidden_dim_mlp,
                                     hidden_dim_lstm=args.hidden_dim_lstm)
    elif args.algorithm == 'reinforce':
        agent = LSTMReinforceAgent(input_dim=in_dim,
                                   output_dim=out_dim,
                                   hidden_dim_mlp=args.hidden_dim_mlp,
                                   hidden_dim_lstm=args.hidden_dim_lstm)
    else:
        raise ValueError('Use one of a2c, ppo, reinforce as algorithm')

    agent.to(DEVICE)
    agent.weight_init()

    if args.algorithm == 'a2c':
        trainer = A2CTrainer(train_env, eval_envs, agent, DEVICE, **trainer_arguments)
    elif args.algorithm == 'ppo':
        trainer = PPOTrainer(train_env, eval_envs, agent, DEVICE, **trainer_arguments)
    elif args.algorithm == 'reinforce':
        trainer = Trainer(train_env, eval_envs, agent, DEVICE, **trainer_arguments)

    trainer.train(steps=args.steps)
    for env in eval_envs:
        env.save_paths(path=f'data/paths/{env.dataset_name}.json')

    torch.save(agent, f'data/models/{args.name}.pt')
    if args.use_wandb:
        artifact = wandb.Artifact(args.name, type='model')
        artifact.add_file(f'data/models/{args.name}.pt')
        run.log_artifact(artifact)


if __name__ == '__main__':
    main()
