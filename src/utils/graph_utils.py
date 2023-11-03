import bz2
import json
import gzip
import nltk
import unicodedata

from bidict import bidict
from collections import defaultdict
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple, Dict


nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = set(stopwords.words('english'))


#########################
# Causenet
#########################

def get_causenet_triples(path: str, include_source: bool = False) -> List[Tuple[str, str]]:
    if 'sample' in path:
        with open(path, 'r') as file_:
            graph = json.load(file_)
    else:
        graph = bz2.open(path, mode='rt')

    triples = []
    for relation in graph:
        if isinstance(relation, dict):
            parsed_relation = relation
        else:
            parsed_relation = json.loads(relation)
        cause = parsed_relation['causal_relation']['cause']['concept'].replace('_', ' ')
        effect = parsed_relation['causal_relation']['effect']['concept'].replace('_', ' ')
        if cause != effect:
            if include_source:
                source = _get_source(parsed_relation, cause, effect)
                triples.append((cause, effect, remove_stop_words(source), source))
            else:
                triples.append((cause, effect, None, None))
    # print('Number triples in graph: ', len(triples))
    return triples


def remove_stop_words(context: str) -> List[str]:
    tokens = word_tokenize(context)
    return [t for t in tokens if t not in STOP_WORDS]


def _get_source(relation, cause, effect):
    for source in relation['sources']:
        if source['type'] in {'clueweb12_sentence', 'wikipedia_sentence'}:
            return source['payload']['sentence']

    if source['type'] == 'wikipedia_list':
        source = _build_source(source['payload']['list_toc_section_heading'], cause, effect)
    else:
        source = _build_source(source['payload']['infobox_argument'], cause, effect)
    return source


def _build_source(connection, cause, effect):
    if connection == 'causes':
        source = cause + ' causes ' + effect
    elif connection == 'Cause':
        source = cause + ' can cause ' + effect
    elif connection == 'cause':
        source = cause + ' can cause ' + effect
    elif connection == 'risks':
        source = cause + ' risks ' + effect
    elif connection == 'symptoms':
        source = effect + ' is a symptom of ' + cause
    elif connection == 'Symptoms':
        source = effect + ' is a symptom of ' + cause
    elif connection == 'Signs and symptoms':
        source = effect + ' is a sign or symptom of ' + cause
    elif connection == 'Causes':
        source = cause + ' causes ' + effect
    elif connection == 'Risk factor':
        source = cause + ' is a risk factor for ' + effect
    else:
        raise ValueError(f'No source with {source} connection')
    return source


#########################
# CSKG
#########################

cskg_relations = {
    '/r/dbpedia/genre': 'genre',
    '/r/UsedFor': 'used for',
    '/r/Antonym': 'antonym',
    '/r/dbpedia/knownFor': 'known for',
    '/r/DefinedAs': 'defined as',
    'at:xWant': 'want',
    '/r/MotivatedByGoal': 'motivated by goal',
    '/r/HasProperty': 'has property',
    '/r/PartOf': 'part of',
    '/r/HasFirstSubevent': 'has first subevent',
    '/r/Desires': 'desires',
    '/r/HasPrerequisite': 'has prerequisite',
    '/r/AtLocation': 'at location',
    '/r/FormOf': 'form of',
    'at:xIntent': 'xintent',
    '/r/DistinctFrom': 'distinct from',
    '/r/NotHasProperty': 'not has property',
    '/r/dbpedia/capital': 'capital',
    'at:oReact': 'oreact',
    '/r/EtymologicallyRelatedTo': 'etymologically related to',
    '/r/HasSubevent': 'has subevent',
    '/r/dbpedia/leader': 'leader',
    '/r/dbpedia/field': 'field',
    'at:xEffect': 'xeffect',
    '/r/MannerOf': 'manner of',
    '/r/RelatedTo': 'related to',
    '/r/SimilarTo': 'similar to',
    '/r/ReceivesAction': 'receives action',
    '/r/dbpedia/genus': 'genus',
    '/r/dbpedia/product': 'product',
    '/r/NotCapableOf': 'not capable of',
    '/r/HasLastSubevent': 'has last subevent',
    '/r/CapableOf': 'capable of',
    'at:xNeed': 'need',
    '/r/DerivedFrom': 'derived from',
    'at:oEffect': 'oeffect',
    '/r/MadeOf': 'made of',
    'at:xAttr': 'attr',
    'at:xReact': 'xreact',
    '/r/Causes': 'causes',
    '/r/CausesDesire': 'causes desire',
    '/r/Entails': 'entails',
    'fn:HasLexicalUnit': 'has lexical unit',
    '/r/dbpedia/occupation': 'occupation',
    '/r/HasA': 'has a',
    '/r/CreatedBy': 'created by',
    '/r/dbpedia/influencedBy': 'influenced by',
    '/r/HasContext': 'has context',
    '/r/Synonym': 'synonym',
    '/r/IsA': 'is a',
    '/r/SymbolOf': 'symbol of',
    '/r/InstanceOf': 'instance of',
    '/r/EtymologicallyDerivedFrom': 'etymologically derived from',
    'mw:MayHaveProperty': 'may have property',
    'at:oWant': 'want',
    '/r/LocatedNear': 'located near',
    '/r/NotDesires': 'not desires',
    '/r/dbpedia/language': 'language'
}


def get_cskg_triples(path: str) -> List[Tuple[str, str]]:
    with gzip.open(path, 'r') as cskg_file:
        triples = [line.decode('utf-8').strip().split('\t') for line in cskg_file]
    lemmatizer = WordNetLemmatizer()
    entities = set(chain.from_iterable([set(t[4].split('|')) | set(t[5].split('|')) for t in triples[1:]]))
    entities = [' '.join({lemmatizer.lemmatize(unicodedata.normalize('NFKC', x.lower())) for x in ent.split(' ')})
                for ent in entities]
    for idx, e in enumerate(entities):
        if idx > 20:
            break
        print(e)
    # triples = [(triple[4].split('|')[0], triple[5].split('|')[0], cskg_relations[triple[2]])
    # for triple in triples[1:]]
    return triples, entities


#########################
# Graph Representation
#########################

def get_entity_index(triples: List[Tuple[str, str]]) -> bidict[int, str]:
    entities = set(chain.from_iterable([(t[0], t[1]) for t in triples]))
    # print(len(entities))
    entities = sorted(entities)
    entity_index = bidict(zip(range(len(entities)), entities))
    entity_index[len(entity_index)] = 'stop stop action'
    return entity_index


def get_adjacency_list(triples: List[Tuple[str, str]],
                       entity_index: bidict[int, str], use_inverse: bool = False) \
        -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], List[str]]]:
    graph = defaultdict(list)
    graph_sources = {}
    graph_sources_full = {}
    for triple in triples:
        subject_id = entity_index.inverse[triple[0]]
        object_id = entity_index.inverse[triple[1]]
        graph[subject_id].append(object_id)
        graph_sources[(subject_id, object_id)] = triple[2]
        graph_sources_full[(subject_id, object_id)] = triple[3]
        if use_inverse:
            graph[object_id].append(subject_id)
            graph_sources[(object_id, subject_id)] = triple[2]
            graph_sources_full[(object_id, subject_id)] = triple[3]

    for key in entity_index:
        graph[key].insert(0, len(entity_index)-1)
        if isinstance(triples[0][2], str):
            graph_sources[(key, len(entity_index)-1)] = 'stop'
        else:
            graph_sources[(key, len(entity_index)-1)] = ['stop']

    return graph, graph_sources, graph_sources_full
