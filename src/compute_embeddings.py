#!/usr/bin/env python

import json
import torch
import csv
from functools import partial
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from utils import graph_utils
from tqdm import tqdm


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def run_sentence_transformer(input_texts, model_name='intfloat/e5-large-v2', batch_size=128,
                             normalize_embeddings=True, trust_remote_code=False):
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    print(model.max_seq_length)
    embeddings = model.encode(input_texts, normalize_embeddings=normalize_embeddings, show_progress_bar=True, batch_size=batch_size)
    print(embeddings.shape)
    return embeddings


def run_e5_mistral_instruct(input_texts, batch_size=64):

    def last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
    model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct').to(DEVICE)

    max_length = 1024

    res = []
    for current in tqdm(range(0, len(input_texts), batch_size)):
        batch_dict = tokenizer(input_texts[current:current+batch_size], max_length=max_length - 1,
                               return_attention_mask=False, padding=False, truncation=True).to(DEVICE)
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')

        with torch.no_grad():
            model_output = model(**batch_dict)
        embeddings = last_token_pool(model_output.last_hidden_state, batch_dict['attention_mask'])
        res.append(embeddings)

    res = torch.cat(res, 0)
    res = F.normalize(res, p=2, dim=1)
    print(res.shape)
    return res


def run_transformers(input_texts, batch_size=64, model='sentence-transformers/all-MiniLM-L6-v2',
                     tokenizer='sentence-transformers/all-MiniLM-L6-v2'):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    model = AutoModel.from_pretrained(model).to(DEVICE)

    res = []
    for current in tqdm(range(0, len(input_texts), batch_size)):
        encoded_input = tokenizer(input_texts[current:current+batch_size], padding=True,
                                  truncation=True, return_tensors='pt').to(DEVICE)

        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).to('cpu')
        res.append(sentence_embeddings)

    res = torch.cat(res, 0)
    res = F.normalize(res, p=2, dim=1)
    print(res.shape)
    return res


def transform_e5(questions):
    return ["query: " + question for question in questions]


def transform_e5_mistral_instruct(questions):
    instruction = "Given a cause and effect, determine whether there is a causal relationship between them."
    return [f'Instruct: {instruction}\nQuery: {question}' for question in questions]


def transform_nomic(questions):
    return ["classification: " + question for question in questions]


def embed_msmarco(model, transform):
    PATH_QA_TRAIN = "datasets/msmarco_train_valid.json"
    PATH_QA_VALID = "datasets/msmarco_test.json"
    train = json.load(open(PATH_QA_TRAIN))
    valid = json.load(open(PATH_QA_VALID))

    train = [question['question'] + '?' for question in train]
    valid = [question['question'] + '?' for question in valid]

    embeddings_train = model(transform(train))
    embeddings_valid = model(transform(valid))
    emb_train = dict(zip(train, embeddings_train))
    emb_valid = dict(zip(valid, embeddings_valid))

    return emb_train, emb_valid, embeddings_train.shape[-1]


def load_semeval(path):
    questions = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            cause = row[0].replace('_', ' ')
            effect = row[1].replace('_', ' ')
            questions.append(f'can {cause} cause {effect}?')
    return questions


def embed_semeval(model, transform):
    train = load_semeval("datasets/sem_train_valid.csv")
    valid = load_semeval("datasets/sem_test.csv")

    embeddings_train = model(transform(train))
    embeddings_valid = model(transform(valid))
    emb_train = dict(zip(train, embeddings_train))
    emb_valid = dict(zip(valid, embeddings_valid))

    return emb_train, emb_valid, embeddings_train.shape[-1]


def embed_msmarco_semeval(model, transform, model_name):
    mtrain, mtest, mdim = embed_msmarco(model, transform)
    strain, stest, sdim = embed_semeval(model, transform)
    train = {**mtrain, **strain}
    test = {**mtest, **stest}
    torch.save(train, f'train_valid_{model_name}_{mdim}.pt')
    torch.save(test, f'test_{model_name}_{sdim}.pt')


def embed_causenet(model, transform, model_name):
    triples = graph_utils.get_causenet_triples('data/causenet-precision.jsonl.bz2', include_source=True)
    entity_index = graph_utils.get_entity_index(triples)

    ids, entities = [], []
    for a, b in entity_index.items():
        ids.append(a)
        entities.append(b)

    embeddings_entities = model(transform(entities))
    _, _, sources_full = graph_utils.get_adjacency_list(triples, entity_index, use_inverse=True)

    ids_sources, sources = [], []
    for a, b in sources_full.items():
        ids_sources.append(a)
        sources.append(b)
    embeddings_sources = model(transform(sources))
    result_sources = dict(zip(ids_sources, embeddings_sources))

    torch.save(embeddings_entities, f'causenet_entites_{model_name}_{embeddings_entities.shape[-1]}.pt')
    torch.save(result_sources, f'causenet_sources_{model_name}_{embeddings_sources.shape[-1]}.pt')


if __name__ == '__main__':
    # E5 https://huggingface.co/intfloat/e5-large-v2
    embed_msmarco_semeval(run_sentence_transformer, transform_e5, "e5_large_v2")
    embed_causenet(run_sentence_transformer, transform_e5, "e5_large_v2")

    # BGE https://huggingface.co/BAAI/bge-large-en-v1.5
    # embed_msmarco_semeval(partial(run_sentence_transformer, model_name="BAAI/bge-large-en-v1.5"),
    #                      lambda x: x, "bge_large_v1.5")
    # embed_causenet(partial(run_sentence_transformer, model_name="BAAI/bge-large-en-v1.5"),
    #               lambda x: x, "bge_large_v1.5")

    # E5 Mistral Instruct https://huggingface.co/intfloat/e5-mistral-7b-instruct
    # embed_msmarco_semeval(run_e5_mistral_instruct, transform_e5_mistral_instruct, "e5_mistral_instruct")
    # embed_causenet(run_e5_mistral_instruct, transform_e5_mistral_instruct, "e5_mistral_instruct")

    # nomic-embed-text-v1 https://huggingface.co/nomic-ai/nomic-embed-text-v1
    run = partial(run_sentence_transformer, model_name="nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    # embed_msmarco_semeval(run, lambda x: x, "nomic-embed-text-v1")
    # embed_causenet(run, lambda x: x, "nomic-embed-text-v1")

    # Mini-LM https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    # embed_msmarco_semeval(run_transformers, lambda x: x, "mini-lm")
    # embed_causenet(run_transformers, lambda x: x, "mini-lm")
