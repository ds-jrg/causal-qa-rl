#!/usr/bin/env python

import torch
import wandb
import optuna

from agents import LSTMActorCriticAgent, LSTMReinforceAgent
from utils.dataset_utils import get_questions_csv, get_questions_msmarco, get_questions_sem_eval
from trainer import A2CTrainer, Trainer, PPOTrainer
from utils import graph_utils, config
from knowledge_graph import KnowledgeGraph
from embeddings import BERTEmbeddingProvider, DebugEmbeddingProvider, GloveEmbeddingProvider
from optuna.integration.wandb import WeightsAndBiasesCallback

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

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


def run_model(trial):
    if args.debug:
        train, valid = get_questions_csv(kg, args.train_file)
    elif args.dataset == 'msmarco':
        train = get_questions_msmarco(kg, args.train_file, False)
        valid = get_questions_msmarco(kg, args.test_file, True)
    elif args.dataset == 'semeval':
        train = get_questions_sem_eval(kg, args.train_file, False)
        valid = get_questions_sem_eval(kg, args.test_file, True)

    print('Train Questions: ', len(train))
    print('Valid Questions: ', len(valid))

    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    trainer_arguments["lr"] = lr
    lambda_gae = trial.suggest_float("lambda_gae", 0.9, 1.0)
    trainer_arguments["lambda_gae"] = lambda_gae
    gamma = trial.suggest_float("gamma", 0.9, 1.0)
    trainer_arguments["discount"] = gamma
    beta_entropy = trial.suggest_float("beta_entropy", 0.005, 0.1)
    trainer_arguments["beta_entropy"] = beta_entropy

    # supervised_batch_size = trial.suggest_int("supervised_batch_size", 16, 128, 16)
    # trainer_arguments["supervised_batch_size"] = supervised_batch_size
    # supervised_steps = trial.suggest_int("supervised_steps", 100, 500, 100)
    # trainer_arguments["supervised_steps"] = supervised_steps
    # supervised_ratio = trial.suggest_float("supervised_ratio", 0.4, 1.0, 0.1)
    # trainer_arguments["supervised_ratio"] = supervised_ratio

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
        return ValueError('Use one of a2c, ppo, reinforce as algorithm')

    agent.to(DEVICE)
    agent.weight_init()

    if args.algorithm == 'a2c':
        trainer = A2CTrainer(train_env, eval_envs, agent, DEVICE, **trainer_arguments)
    elif args.algorithm == 'ppo':
        trainer = PPOTrainer(train_env, eval_envs, agent, DEVICE, **trainer_arguments)
    elif args.algorithm == 'reinforce':
        trainer = Trainer(train_env, eval_envs, agent, DEVICE, **trainer_arguments)

    eval_return = trainer.train(steps=args.steps)
    return eval_return


if __name__ == '__main__':
    wandb_kwargs = {"project": args.wandb_project, "name": args.name}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)

    study = optuna.create_study(direction='maximize')
    study.optimize(run_model, n_trials=args.trials, callbacks=[wandbc])
    print(study.best_params)
