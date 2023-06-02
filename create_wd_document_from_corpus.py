import json
import os.path

import spacy
from spacy.tokenizer import Tokenizer
import neuralcoref
from conll_reader import read_CoNLL
from mentionsfromjson import loadMentionsFromJson
from shared.CONSTANTS import CONFIG, EECDCR_CONFIG_DICT
from tqdm import tqdm

from shared.classes import Corpus


nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = Tokenizer(nlp.vocab, prefix_search=None,
                          suffix_search=None,
                          infix_finditer=None,
                          token_match=None)
neuralcoref.add_to_pipe(nlp)


def dismember_document_to_token(document):
    print('entered dismember document token')
    document_as_tokens = []
    for sentence in document.sentences.values():
        for token in sentence.get_tokens():
            document_as_tokens.append(token)

    full_text = " ".join([token.token for token in document_as_tokens])

    return document_as_tokens, full_text


def create_wd_document_from_corpus(corpus: Corpus):
    print("/n Creating a document with the correfering mentions at same document")
    cand_list = []
    coref_chain = 50000
    for topic in tqdm(corpus.topics.values(), desc='Resolving WDCR'):
        for doc_id, document in topic.docs.items():
            print('doc id',doc_id)
            print('document',document)
            tokens, full_text = dismember_document_to_token(document)
            #print('\nAfter returning from dismember document function \ntokens',tokens)
            print('\nfull text',full_text)
            doc = nlp(full_text)
            print('\n After returning from nlp function \n')
            print('doc \n', doc)
            doc_clusters = doc._.coref_clusters
            print('\n\ndoc clusters\n', doc_clusters)

            for cluster in doc_clusters:
                for mention in cluster.mentions:
                    for sentence in document.sentences.values():
                        raw_sentence = sentence.get_raw_sentence()
                        if mention.sent.text.startswith(raw_sentence):
                            sent_id = sentence.sent_id
                    token_numbers = [int(token.token_id) for token in tokens[mention.start:mention.end]
                                     if token.token_id.isdecimal()]
                    token_str = mention.string
                    is_continuous = token_numbers == list(range(token_numbers[0], token_numbers[-1] + 1))
                    cand_list.append({
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "tokens_numbers": token_numbers,
                        "tokens_str": token_str,
                        "coref_chain": f"{coref_chain}",
                        "is_continuous": is_continuous,
                        "is_singleton": False
                    })
                coref_chain += 1
        print('\n\nfinished one iteration of create_wd_document_from_corpus')
    if not os.path.exists(EECDCR_CONFIG_DICT["wd_entity_coref_file_path"].format(CONFIG['dataset_name'])):
        os.makedirs(EECDCR_CONFIG_DICT["wd_entity_coref_file_path"].format(CONFIG['dataset_name']))

    with open(EECDCR_CONFIG_DICT["wd_entity_coref_file"].format(CONFIG['dataset_name']), "w") as f:
        json.dump(cand_list, f, indent=1)


def create_complete_wd_document():
    """
        This function creates a document with all within document coreference resolutions
        for a dataset defined in CONFIG
    """
    corpus = read_CoNLL()
    corpus = loadMentionsFromJson(corpus)
    create_wd_document_from_corpus(corpus)
    print("Finished creating wd document \n")
