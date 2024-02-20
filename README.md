# Answering Causal Questions With Reinforcement Learning

Paper: https://arxiv.org/abs/2311.02760

## Installation

Clone the repository:
```
git clone https://github.com/ds-jrg/causal-qa-rl.git
```

Optionally create a python venv or conda environment. Requires `Python >= 3.10`.

With `Anaconda3`:
```
conda create --name causalqa python=3.10
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

## Reproduce Results

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

## Training

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

## Inference

An inference example can be found in this [evaluation script](src/evaluate_agent.py).

## Embeddings

We provide a script [here](src/compute_embeddings.py), to compute embeddings from different transformer based models for MS MARCO, SemEval and CauseNet.

Per default we use GloVe embeddings, because they provided similar performance.
