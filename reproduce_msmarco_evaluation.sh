#!/usr/bin/env bash

# BFS
echo "BFS 1-Hop" >> temp.tsv
src/evaluate_bfs.py --dataset msmarco --path_len 1 >> temp.tsv
echo "BFS 2-Hop" >> temp.tsv
src/evaluate_bfs.py --dataset msmarco --path_len 2 >> temp.tsv
echo "BFS 3-Hop" >> temp.tsv
src/evaluate_bfs.py --dataset msmarco --path_len 3 >> temp.tsv
echo "BFS 4-Hop" >> temp.tsv
src/evaluate_bfs.py --dataset msmarco --path_len 4 >> temp.tsv

# UnifiedQA
echo "UnifiedQA-v2" >> temp.tsv
src/evaluate_unifiedqa.py --dataset msmarco \
                          --context_file "src/predictions/msmarco3_context.jsonl" \
                          >> temp.tsv
echo "UnifiedQA-v2-T" >> temp.tsv
src/evaluate_unifiedqa.py --dataset msmarco --use_context \
                          --context_file "src/predictions/msmarco_simple_context.jsonl" \
                          >> temp.tsv
echo "UnifiedQA-v2-P" >> temp.tsv
src/evaluate_unifiedqa.py --dataset msmarco --use_context \
                          --context_file "src/predictions/msmarco3_context.jsonl" \
                          >> temp.tsv

# Agent
echo "Agent 1-Hop" >> temp.tsv
src/evaluate_agent.py --dataset msmarco --model "models/msmarco_1.pt" --path_len_eval 1 >> temp.tsv
echo "Agent 2-Hop" >> temp.tsv
src/evaluate_agent.py --dataset msmarco --model "models/msmarco_2.pt" --path_len_eval 2 >> temp.tsv
echo "Agent 3-Hop" >> temp.tsv
src/evaluate_agent.py --dataset msmarco --model "models/msmarco_3.pt" --path_len_eval 3 >> temp.tsv
echo "Agent 4-Hop" >> temp.tsv
src/evaluate_agent.py --dataset msmarco --model "models/msmarco_4.pt" --path_len_eval 4 >> temp.tsv


./build_md.py temp.tsv "msmarco_evaluation.md"
rm temp.tsv
