from dataclasses import dataclass
from itertools import chain
import json
import csv
from typing import Optional
import nltk
import unicodedata
import numpy as np

from nltk import word_tokenize
from nltk.corpus import wordnet


nltk.download('wordnet')
nltk.download('omw-1.4')


@dataclass
class Question:
    question: str
    cause: str
    effect: str
    embedding: np.array
    binary_answer: Optional[bool]
    id_: int

    def __str__(self):
        return (f'(Question: {self.question} | Cause: {self.cause} | Effect: {self.effect}'
                f' | Embedding: {self.embedding.shape} | Binary_Answer: {self.binary_answer})')

    def __repr__(self):
        return (f'(Question: {self.question} | Cause: {self.cause} | Effect: {self.effect}'
                f' | Embedding: {self.embedding.shape} | Binary_Answer: {self.binary_answer})')


def get_synonyms(word):
    synonyms = wordnet.synsets(word)
    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
    lemmas.add(word)
    return lemmas


def normalize_causal_concept(string):
    # as in conceptNet
    # https://en.wikipedia.org/wiki/Unicode_equivalence#Normalization
    tokens = word_tokenize(string)
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(unicodedata.normalize('NFKC', token.lower()))
    tokens = [unicodedata.normalize('NFKC', token.lower())
              for token in tokens]
    return ' '.join(tokens)


def get_concept_from_question(question, index):
    start = question['query'][index][0]
    end = question['query'][index][1] + 1
    concept = [t[0] for t in question['question:POS'][start:end]]
    concept = ' '.join(concept)
    return normalize_causal_concept(concept)


def get_entity(entity, kg):
    return entity if entity in kg.entity_index.inverse else None


def get_triple(cause, effect, kg):
    if cause in kg.entity_index.inverse and effect in kg.entity_index.inverse:
        if kg.entity_to_id(effect) in kg.neighbour_ids(kg.entity_to_id(cause)):
            return True
    return False


def build_questions(dataset, kg, valid: bool = False):
    questions = []
    embedding_provider = kg.embedding_provider
    idx = 0
    contexts = []
    for question in dataset:
        # remove questions without answer
        if (question['answer:Extracted'][0] == 'Yes' or (question['answer:Extracted'][0] == 'No' and valid)):
            question_contexts = {}
            question_contexts['id_'] = idx

            cause = get_concept_from_question(question, 0)
            effect = get_concept_from_question(question, 1)
            cause_entity = get_entity(cause, kg)
            effect_entity = get_entity(effect, kg)
            if get_triple(cause, effect, kg):
                question_contexts['context_only'] = f'{cause_entity} causes {effect_entity}.'
            else:
                question_contexts['context_only'] = ''
            if valid and cause_entity is None:
                cause_entity = kg.id_to_entity(kg.stop_action)
            if (cause_entity is not None and effect_entity is not None) or (valid and cause_entity is not None):
                questions.append(Question(question['question'] + '?',
                                 cause_entity, effect_entity,
                                 embedding_provider.question_embeddings(question['question'] + '?'),
                                 question['answer:Extracted'][0] == 'Yes', idx))
                idx += 1
            contexts.append(question_contexts)

    # with open('msmarco_simple_context.jsonl', 'w+') as f:
    #    for context in contexts:
    #        f.write(json.dumps(context) + '\n')

    return questions


def get_questions_msmarco(kg, path, valid=False):
    path = json.load(open(path))
    questions = build_questions(path, kg, valid=valid)
    return questions


def get_questions_csv(kg, path):
    train = []
    valid = []
    embedding_provider = kg.embedding_provider
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for idx, row in enumerate(reader):
            cause = row[0].replace('_', ' ')
            effect = row[1].replace('_', ' ')
            train.append(Question(cause, cause, effect, embedding_provider.question_embeddings(cause), True, idx))
            valid.append(Question(cause, cause, effect, embedding_provider.question_embeddings(cause), True, idx))
    return train, valid


def get_questions_sem_eval(kg, path, valid=False):
    questions = []
    embedding_provider = kg.embedding_provider
    contexts = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        idx = 0
        for row in reader:
            question_contexts = {}
            question_contexts['id_'] = idx
            cause = row[0].replace('_', ' ')
            effect = row[1].replace('_', ' ')
            causal = row[2].replace('_', ' ')
            cause_entity = get_entity(cause, kg)
            effect_entity = get_entity(effect, kg)
            if get_triple(cause, effect, kg):
                question_contexts['context_only'] = f'{cause_entity} causes {effect_entity}.'
            else:
                question_contexts['context_only'] = ''
            if valid and cause_entity is None:
                cause_entity = kg.id_to_entity(kg.stop_action)
            if (cause_entity is not None and effect_entity is not None) or (valid and cause_entity is not None):
                questions.append(Question(f'can {cause} cause {effect}?',
                                 cause_entity, effect_entity,
                                 embedding_provider.question_embeddings(f'can {cause} cause {effect}?'),
                                 causal == 'causal', idx))
                idx += 1
            contexts.append(question_contexts)
    # with open('semeval_simple_context.jsonl', 'w+') as f:
    #    for context in contexts:
    #        f.write(json.dumps(context) + '\n')
    return questions
