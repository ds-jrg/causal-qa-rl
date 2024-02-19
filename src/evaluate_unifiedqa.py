#!/usr/bin/env python

import argparse
import json
import torch
import pandas as pd

from tqdm import tqdm
from functools import partial

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed
from utils.agent_utils import compute_metrics


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default='allenai/unifiedqa-v2-t5-base-1363200',
                        help='Checkpoint to evaluate.')
    parser.add_argument("--tokenizer", type=str, default='allenai/unifiedqa-v2-t5-base-1363200')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=8,
                        help='Number of processes for dataset loading.')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default='msmarco')
    parser.add_argument("--test_file", type=str, default='datasets/msmarco_test.json')
    parser.add_argument('--use_context', action='store_true')
    parser.add_argument('--no-use_context', dest='use_context', action='store_false')
    parser.set_defaults(use_context=False)
    parser.add_argument("--context_file", type=str, default='src/predictions/msmarco3_context.jsonl')
    return parser.parse_args()


def preprocess(string):
    return string.lower().strip().replace('\n', '').replace('\t', '').replace('   ', ' ').replace('  ', ' ')


# concatenate question+context with \\n as a separator
def build_input_msmarco(batch, use_context: bool = False):
    input_ = [(f'{preprocess(question)}? \\n {preprocess(context)}' if use_context else f'{preprocess(question)}?')
              for question, context in zip(batch['question'], batch['context_sentence'])]
    batch['input'] = input_
    batch['answer'] = [x[0].lower() for x in batch['answer:Extracted']]
    return batch


def build_input_semeval(batch, use_context: bool = False):
    input_ = [(f'can {cause} cause {effect}? \\n {preprocess(context)}'
               if use_context else f'can {cause} cause {effect}?')
              for cause, effect, context in zip(batch['cause'], batch['effect'], batch['context_sentence'])]
    batch['input'] = input_
    batch['answer'] = [('yes' if x == 'causal' else 'no') for x in batch['causal']]
    return batch


def run_model(batch, model, tokenizer):
    encoded_inputs = tokenizer(batch, padding='longest', return_tensors="pt").to(DEVICE)
    res = model.generate(**encoded_inputs, max_length=500)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


def read_msmarco(args):
    with open(args.test_file) as f:
        lines = [json.loads(line.strip()) for line in f]

    with open(args.context_file) as f:
        lines_context = [json.loads(line.strip()) for line in f]

    data = Dataset.from_pandas(pd.DataFrame(data=lines[0]))
    data = data.filter(lambda x: x['answer:Extracted'][0] == 'Yes' or x['answer:Extracted'][0] == 'No',
                       batched=False, load_from_cache_file=False, num_proc=args.num_procs)
    data = pd.DataFrame(data=data.to_dict())
    data = pd.concat([data, pd.DataFrame(data=lines_context)], axis=1)
    data = Dataset.from_pandas(data)
    build_input = partial(build_input_msmarco, use_context=args.use_context)
    data = data.map(build_input, batched=True, load_from_cache_file=False, num_proc=args.num_procs)

    true_labels = [int(d['answer:Extracted'][0].lower() == 'yes') for d in data]
    return data, true_labels


def read_semeval(args):
    with open(args.context_file) as f:
        lines_context = [json.loads(line.strip()) for line in f]

    data = pd.read_csv(args.test_file)
    data = pd.concat([data, pd.DataFrame(data=lines_context)], axis=1)
    data = Dataset.from_pandas(data)
    build_input = partial(build_input_semeval, use_context=args.use_context)
    data = data.map(build_input, batched=True, load_from_cache_file=False, num_proc=args.num_procs)

    true_labels = [int(d['answer'] == 'yes') for d in data]
    return data, true_labels


def evaluate_unifiedqa(args):
    set_seed(args.seed)
    if args.dataset == 'msmarco':
        data, true_labels = read_msmarco(args)
    elif args.dataset == 'semeval':
        data, true_labels = read_semeval(args)
    else:
        raise ValueError('Use msmarco or semeval as dataset.')

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(DEVICE)

    loader = DataLoader(data, shuffle=False, num_workers=0, batch_size=args.batch_size)
    predictions = []
    for batch in tqdm(loader):
        batch_predictions = run_model(batch['input'], model, tokenizer)
        predictions.extend(batch_predictions)

    answers = data['answer']
    answers = [answer.split('\t') for answer in answers]
    pred_numbers = [int(pred.lower() == 'yes') for pred in predictions]

    metrics = compute_metrics(true_labels, pred_numbers)
    print(metrics)


if __name__ == '__main__':
    evaluate_unifiedqa(parse_args())
