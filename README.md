# Answering Causal Questions With Reinforcement Learning

## Installation

Clone the repository:
```
git clone https://github.com/CausalRLQA/CausalRLQA
```

Optionally create a python venv or conda environment. Requires `Python >= 3.9`.

With `Anaconda3`:
```
conda create --name causalqa python=3.9
conda activate causalqa
```

With `python`:
```
python -m venv causalqa
source causalqa/bin/activate
```

Run `setup.sh` to download CauseNet-Precision, GloVe embeddings, the pre-trained models, and to install the required dependencies:
```
./setup.sh
```
(You might have to update the torch settings depending on your system)

## Reproduce Results:

To reproduce the evaluation results for `MS MARCO` and `SemEval` run:
```
./reproduce_msmarco_evaluation.sh
./reproduce_semeval_evaluation.sh
```

To reproduce the results of the ablation study for `MS MARCO` and `SemEval` run:
```
./reproduce_msmarco_ablation.sh
./reproduce_semeval_ablation.sh
```

## Training:

The `run.py` script can be used to train the agent:

For example, with the configurations we used for `MS MARCO`:
```
src/run.py --name "msmarco_evaluation" \
		 --dataset "msmarco" \
		 --steps 2000 \
		 --supervised \
		 --supervised_steps 300 \
		 --supervised_ratio 0.8 \
		 --supervised_batch_size 64
```

To enable logging with `wandb` set the `--use_wandb` flag and the `--wandb_project` and `--wandb_entity` accordingly.

More usage information:

```
usage: run.py [-h] [--dataset DATASET] [--knowledge_graph KNOWLEDGE_GRAPH] [--embeddings EMBEDDINGS]
              [--train_file TRAIN_FILE] [--test_file TEST_FILE] [--debug] [--no-debug] [--use_inverse]
              [--no-use_inverse] [--use_actor_critic] [--no-use_actor_critic]
              [--hidden_dim_lstm HIDDEN_DIM_LSTM] [--hidden_dim_mlp HIDDEN_DIM_MLP] [--use_question]
              [--no-use_question] [--use_relation_action] [--no-use_relation_action] [--use_full_path]
              [--no-use_full_path] [--steps STEPS] [--supervised] [--no-supervised] [--normalize_returns]
              [--no-normalize_returns] [--beam_search] [--no-beam_search] [--eval_interval EVAL_INTERVAL]
              [--batch_size BATCH_SIZE] [--path_len_train PATH_LEN_TRAIN] [--path_len_eval PATH_LEN_EVAL]
              [--beam_width BEAM_WIDTH] [--lr LR] [--entropy_weight ENTROPY_WEIGHT] [--discount DISCOUNT]
              [--supervised_ratio SUPERVISED_RATIO] [--supervised_steps SUPERVISED_STEPS]
              [--supervised_batch_size SUPERVISED_BATCH_SIZE] [--max_grad_norm MAX_GRAD_NORM]
              [--lambda_gae LAMBDA_GAE] [--seed SEED] [--use_gae] [--no-use_gae] [--use_advantage]
              [--no-use_advantage] [--use_wandb] [--no-use_wandb] [--wandb_project WANDB_PROJECT]
              [--wandb_entity WANDB_ENTITY] [--name NAME]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset to use: msmarco or semeval (default: msmarco)
  --knowledge_graph KNOWLEDGE_GRAPH
  --embeddings EMBEDDINGS
  --train_file TRAIN_FILE
  --test_file TEST_FILE
  --debug               Uses causenet-sample and a toy dataset for faster loading for debugging. (default:
                        False)
  --no-debug
  --use_inverse         Whether to add inverse edges to the graph. (default: True)
  --no-use_inverse
  --use_actor_critic    Whether to use Actor-Critic. If False then REINFORCE is used. (default: True)
  --no-use_actor_critic
  --hidden_dim_lstm HIDDEN_DIM_LSTM
                        Projection dimension of the LSTM. (default: 1024)
  --hidden_dim_mlp HIDDEN_DIM_MLP
                        Hidden dimension of the two feedforward heads (default: 2048)
  --use_question        Whether to concatenate the question embedding with the current entity embedding or
                        not. (default: True)
  --no-use_question
  --use_relation_action
                        Whether to concatenate the sentence embedding with the next entity/action embedding.
                        (default: True)
  --no-use_relation_action
  --use_full_path       Whether to check the whole path for the target entity or the last node. (default:
                        True)
  --no-use_full_path
  --steps STEPS         Number of train steps. (default: 1000)
  --supervised          Whether to apply supervised learning at the start. (default: False)
  --no-supervised
  --normalize_returns   Whether to normalize the returns/advantages. (default: False)
  --no-normalize_returns
  --beam_search         Whether to use beam search, if False then greedy decoding is used. (default: True)
  --no-beam_search
  --eval_interval EVAL_INTERVAL
                        Interval in train steps for evaluation/logging. (default: 10)
  --batch_size BATCH_SIZE
                        Batch size for policy gradient training. (default: 128)
  --path_len_train PATH_LEN_TRAIN
                        Path rollout length for training. (default: 2)
  --path_len_eval PATH_LEN_EVAL
                        Path rollout length for evaluation. (default: 2)
  --beam_width BEAM_WIDTH
                        Beam width during inference time. (default: 50)
  --lr LR               Learning rate for AdamW. (default: 0.0001)
  --entropy_weight ENTROPY_WEIGHT
                        Weight of the entropy regularization. (default: 0.01)
  --discount DISCOUNT   Discount factor. (default: 0.99)
  --supervised_ratio SUPERVISED_RATIO
                        Ratio of train questions used for supervised learning. (default: 0.5)
  --supervised_steps SUPERVISED_STEPS
                        Supervised train steps. (default: 100)
  --supervised_batch_size SUPERVISED_BATCH_SIZE
                        Batch size for supervised learning. (default: 32)
  --max_grad_norm MAX_GRAD_NORM
                        Maximum norm for gradient clipping. (default: 0.5)
  --lambda_gae LAMBDA_GAE
                        GAE lambda. (default: 0.95)
  --seed SEED           Seed. (default: 42)
  --use_gae             Whether to use GAE or not. (default: True)
  --no-use_gae
  --use_advantage
  --no-use_advantage
  --use_wandb
  --no-use_wandb
  --wandb_project WANDB_PROJECT
  --wandb_entity WANDB_ENTITY
  --name NAME           Name for the run on wandb. (default: msmarco)
```
