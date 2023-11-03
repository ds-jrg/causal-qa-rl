#!/usr/bin/env bash

# BFS
echo "BFS 1-Hop" >> temp.tsv
src/bfs.py --dataset semeval --test_file datasets/sem_test.csv --path_len 1 >> temp.tsv
echo "BFS 2-Hop" >> temp.tsv
src/bfs.py --dataset semeval --test_file datasets/sem_test.csv --path_len 2 >> temp.tsv
echo "BFS 3-Hop" >> temp.tsv
src/bfs.py --dataset semeval --test_file datasets/sem_test.csv --path_len 3 >> temp.tsv
echo "BFS 4-Hop" >> temp.tsv
src/bfs.py --dataset semeval --test_file datasets/sem_test.csv --path_len 4 >> temp.tsv

# UnifiedQA
echo "UnifiedQA-v2" >> temp.tsv
src/unifiedqa/evaluate_unifiedqa.py --dataset semeval --test_file datasets/sem_test.csv \
                                    --context_file "src/predictions/semeval3_context.jsonl" \
                                    >> temp.tsv
echo "UnifiedQA-v2-T" >> temp.tsv
src/unifiedqa/evaluate_unifiedqa.py --dataset semeval --use_context --test_file datasets/sem_test.csv \
                                    --context_file "src/predictions/semeval_simple_context.jsonl" \
                                    >> temp.tsv
echo "UnifiedQA-v2-P" >> temp.tsv
src/unifiedqa/evaluate_unifiedqa.py --dataset semeval --use_context --test_file datasets/sem_test.csv \
                                    --context_file "src/predictions/semeval3_context.jsonl" \
                                    >> temp.tsv

# Agent
echo "Agent 1-Hop" >> temp.tsv
src/evaluate.py --dataset semeval --test_file datasets/sem_test.csv --model "models/semeval_1.pt" --path_len_eval 1 >> temp.tsv
echo "Agent 2-Hop" >> temp.tsv
src/evaluate.py --dataset semeval --test_file datasets/sem_test.csv --model "models/semeval_2.pt" --path_len_eval 2 >> temp.tsv
echo "Agent 3-Hop" >> temp.tsv
src/evaluate.py --dataset semeval --test_file datasets/sem_test.csv --model "models/semeval_3.pt" --path_len_eval 3 >> temp.tsv
echo "Agent 4-Hop" >> temp.tsv
src/evaluate.py --dataset semeval --test_file datasets/sem_test.csv --model "models/semeval_4.pt" --path_len_eval 4 >> temp.tsv

./build_md.py temp.tsv "semeval_evaluation.md"
rm temp.tsv
