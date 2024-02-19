import argparse
import random
import torch
import numpy as np
from utils.dataset_utils import get_questions_csv, get_questions_msmarco, get_questions_sem_eval
from environment import EnvironmentTorch


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, default='msmarco', help='Name for the model/run.')
    parser.add_argument('--train_datasets', nargs='+',
                        default=['datasets/msmarco_train_valid.json', 'datasets/sem_train_valid.csv'])
    parser.add_argument('--test_datasets', nargs='+',
                        default=['datasets/msmarco_test.json', 'datasets/sem_test.csv'])

    parser.add_argument("--knowledge_graph", type=str, default='data/causenet-precision.jsonl.bz2')
    parser.add_argument("--embeddings", type=str, default='glove', help='Which embeddings to use glove or lm.')
    parser.add_argument("--glove_embeddings", type=str, default='data/glove.6B.zip')
    parser.add_argument("--lm_entity_embedding_path", type=str, default='data/causenet_entites_e5_large_v2_1024.pt')
    parser.add_argument("--lm_relation_embedding_path", type=str, default='data/causenet_sources_e5_large_v2_1024.pt')
    parser.add_argument("--lm_train_questions_embedding_path", type=str,
                        default='data/msmarco_train_valid_e5_large_v2_1024.pt')
    parser.add_argument("--lm_test_questions_embedding_path", type=str, default='data/msmarco_test_e5_large_v2_1024.pt')

    parser.add_argument('--debug', action='store_true',
                        help='Uses causenet-sample and a toy dataset for faster loading for debugging.')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=False)

    parser.add_argument('--use_inverse', action='store_true', help='Whether to add inverse edges to the graph.')
    parser.add_argument('--no-use_inverse', dest='use_inverse', action='store_false')
    parser.set_defaults(use_inverse=True)

    parser.add_argument("--algorithm", type=str, default='a2c', help='RL algorithm to use: reinforce, a2c, ppo.')

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

    parser.add_argument("--eval_interval", type=int, default=20, help='Interval in train steps for evaluation/logging.')
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

    parser.add_argument("--ppo_epochs", type=int, default=2, help='Epochs of the inner PPO loop.')
    parser.add_argument("--ppo_batch_size", type=int, default=32, help='Batch Size of the inner PPO loop.')
    parser.add_argument("--ppo_clip", type=int, default=0.2, help='Clip threshold for the actor and critic.')

    parser.add_argument('--clip_critic', action='store_true', help='Whether to use clip the critic in PPO.')
    parser.add_argument('--no-clip_critic', dest='clip_critic', action='store_false')
    parser.set_defaults(clip_critic=False)

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--no-use_wandb', dest='use_wandb', action='store_false')
    parser.set_defaults(use_wandb=False)
    parser.add_argument("--wandb_project", type=str, default='test')
    parser.add_argument("--wandb_entity", type=str, default='test')

    return parser.parse_args()


def get_wandb_config_dict(args):
    wandb_conf_dict = {
        'datasets': args.train_datasets,
        'embeddings': args.embeddings,
        'lm_entity_embeddings': args.lm_entity_embedding_path,
        'lm_relation_embeddings': args.lm_relation_embedding_path,
        'lm_train_questions_embeddings': args.lm_train_questions_embedding_path,
        'lm_test_questions_embeddings': args.lm_test_questions_embedding_path,
        'steps': args.steps,
        'use_question': args.use_question,
        'use_relation_action': args.use_relation_action,
        'use_full_path': args.use_full_path,
        'use_inverse': args.use_inverse,
        'algorithm': args.algorithm,
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
        'use_gae': args.use_gae,
        'ppo_epochs': args.ppo_epochs,
        'ppo_batch_size': args.ppo_batch_size,
        'ppo_clip': args.ppo_clip,
        'clip_critic': args.clip_critic,
    }
    return wandb_conf_dict


def build_trainer_arguments(args):
    trainer_arguments = {
        'use_wandb': args.use_wandb,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'eval_interval': args.eval_interval,
        'supervised': args.supervised,
        'use_full_path': args.use_full_path,
        'beam_search': args.beam_search,
        'beam_width': args.beam_width,
        'supervised_ratio': args.supervised_ratio,
        'supervised_steps': args.supervised_steps,
        'supervised_batch_size': args.supervised_batch_size,
        'beta_entropy': args.entropy_weight,
        'max_grad_norm': args.max_grad_norm,
        'discount': args.discount,
        'normalize_returns': args.normalize_returns,
    }

    if args.algorithm == 'a2c' or args.algorithm == 'ppo':
        a2c_arguments = {'lambda_gae': args.lambda_gae, 'use_gae': args.use_gae}
        trainer_arguments = {**trainer_arguments, **a2c_arguments}

    if args.algorithm == 'ppo':
        ppo_arguments = {'ppo_epochs': args.ppo_epochs, 'ppo_batch_size': args.ppo_batch_size,
                         'ppo_clip': args.ppo_clip, 'clip_critic': args.clip_critic}
        trainer_arguments = {**trainer_arguments, **ppo_arguments}

    return trainer_arguments


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup():
    args = parse_args()
    set_seed(args.seed)
    wandb_config_dict = get_wandb_config_dict(args)
    trainer_arguments = build_trainer_arguments(args)
    return args, wandb_config_dict, trainer_arguments


def build_train_environment(args, kg):
    if args.debug:
        train = get_questions_csv(kg, 'datasets/questions_sample.csv')
    else:
        train = []
        for dataset in args.train_datasets:
            if 'msmarco' in dataset:
                train += get_questions_msmarco(kg, dataset, False)
            elif 'sem' in dataset:
                train += get_questions_sem_eval(kg, dataset, False)

    print('Train Questions: ', len(train))
    train_env = EnvironmentTorch(kg, train,
                                 max_path_len=args.path_len_train,
                                 use_question=args.use_question,
                                 use_relation_action=args.use_relation_action,
                                 valid_mode=False, debug_mode=False)
    return train_env


def build_test_environments(args, kg):
    if args.debug:
        test_dataset = get_questions_csv(kg, 'datasets/questions_sample.csv')
        print('Test Questions: ', len(test_dataset))
        test_env = EnvironmentTorch(kg, test_dataset,
                                    max_path_len=args.path_len_train,
                                    use_question=args.use_question,
                                    use_relation_action=args.use_relation_action,
                                    valid_mode=True, debug_mode=False, dataset_name='debug1')
        test_env2 = EnvironmentTorch(kg, test_dataset,
                                     max_path_len=args.path_len_train,
                                     use_question=args.use_question,
                                     use_relation_action=args.use_relation_action,
                                     valid_mode=True, debug_mode=False, dataset_name='debug2')
        return [test_env, test_env2]

    environments = []
    for dataset in args.test_datasets:
        if 'msmarco' in dataset:
            test_dataset = get_questions_msmarco(kg, dataset, True)
            name = 'msmarco'
        elif 'sem' in dataset:
            test_dataset = get_questions_sem_eval(kg, dataset, True)
            name = 'semeval'

        print('Test Questions: ', len(test_dataset))
        test_env = EnvironmentTorch(kg, test_dataset,
                                    max_path_len=args.path_len_train,
                                    use_question=args.use_question,
                                    use_relation_action=args.use_relation_action,
                                    valid_mode=True, debug_mode=False, dataset_name=name)

        environments.append(test_env)
    return environments
