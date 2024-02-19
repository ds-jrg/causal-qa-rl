#!/usr/bin/env bash

echo "Agent 2-Hop" > temp.tsv
src/evaluate_agent.py --dataset semeval --test_file datasets/sem_test.csv \
					  --model "models/semeval_2.pt" >> temp.tsv
echo "- Beam Search" >> temp.tsv
src/evaluate_agent.py --dataset semeval --test_file datasets/sem_test.csv \
					  --model "models/semeval_2_no_beam_search.pt" --no-beam_search >> temp.tsv
echo "- Supervised Learning" >> temp.tsv
src/evaluate_agent.py --dataset semeval --test_file datasets/sem_test.csv \
                      --model "models/semeval_2_no_supervised.pt" >> temp.tsv
echo "- Actor-Critic" >> temp.tsv
src/evaluate_agent.py --dataset semeval --test_file datasets/sem_test.csv \
  					  --model "models/semeval_2_no_actor_critic.pt" --no-use_critic >> temp.tsv
echo "- Inverse Edges" >> temp.tsv
src/evaluate_agent.py --dataset semeval --test_file datasets/sem_test.csv \
					  --model "models/semeval_2_no_inverse_edges.pt" --no-use_inverse >> temp.tsv

./build_md.py temp.tsv "semeval_ablation.md"
rm temp.tsv
