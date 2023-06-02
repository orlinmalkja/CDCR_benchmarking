import os.path

from shared.classes import Corpus, EntityMention, EventMention, Token, Sentence, Document, Topic
from shared.CONSTANTS import mentions_path, CONFIG, EECDCR_CONFIG_DICT
import json
import spacy
from tqdm import tqdm
from features.build_features import match_allen_srl_structures, load_elmo_embeddings
from EECDCR.all_models.train_model import train_model
from conll_reader import read_CoNLL
from create_wd_document_from_corpus import create_complete_wd_document
from features.build_features import match_allen_srl_structures, load_elmo_embeddings
from features.create_elmo_embeddings import ElmoEmbedding
from make_gold_files import create_gold_files_for_corpus
#from mentionsfromjson import loadMentionsFromJson
from run_eecdcr import test_model, run_conll_scorer
from shared.CONSTANTS import CONFIG, EECDCR_TRAIN_CONFIG_DICT, EECDCR_CONFIG_DICT
from srl_things import get_srl_data, find_args_by_dependency_parsing, find_left_and_right_mentions
import logging
import random
import numpy as np


## convert data (i.e train, dev) in str format -> Corpus object
## Reason: In order to add the mentions to the respective sentence, which in turn is part of the Corpus

def conll_to_corpus(data):

    data = data.split("\n")

    prev_sentence_id = ""
    sentence = Sentence(-1)

    prev_doc_id = ""
    doc_id = ""
    document = Document("Bla, Bla, Bla, Mr. Freeman")

    prev_topic_id = ""
    topic_name = ""
    topic = Topic("Nanananananana, BATMAN!")

    corpus = Corpus()

    for bs in data:
        # jump over empty entries
        if not bs:
            continue

        # if the first entry is a # we know its the beginning or the end line
        if bs.startswith("#begin"):
            # if its the first entry just ignore it
            continue
        if bs.startswith("#end"):
            # if its the last entry just ignore it
            continue

        split_line = bs.split("\t")

        if len(split_line) == 5:
            token_txt = split_line[3]
        elif len(split_line) == 3:
              token_txt = "\n"
        else:
            continue
        
        conll_line = bs
        # Reading the topic, document id, sentence and token
        topic_subtopic = conll_line.split("\t")[0]
        sentence_id = int(conll_line.split("\t")[1])
        token_id = int(conll_line.split("\t")[2])
        doc_id = conll_line.split("\t")[0].split("/")[-1]
        token_txt = conll_line.split("\t")[3]

        #topic_document = split_line[0]
        #topic_name, sub_topic, doc_id = topic_document.split("/")
        #topic_name = topic_name + "/" + sub_topic

        topic_name = "/".join(topic_subtopic.split("/",2)[:2])


        if token_txt != "":
            token = Token(token_id, token_txt)
        else:
            # for some reason some tokens are empty. For analysis reasons, these are printed.
            print(f"Skipped the token {bs}, since it was empty.")


        if sentence_id != prev_sentence_id:
            if prev_sentence_id != "":
                document.add_sentence(prev_sentence_id, sentence)
            sentence = Sentence(sentence_id)
            sentence.add_token(token)


        elif sentence_id == prev_sentence_id:

            if doc_id == prev_doc_id:
                sentence.add_token(token)
            else:
                document.add_sentence(prev_sentence_id, sentence)
                sentence = Sentence(sentence_id)
                prev_sentence_id = sentence_id
                sentence.add_token(token)

        prev_sentence_id = sentence_id

        ## --------

        # if we start a new document (name of the document changes)
        if doc_id != prev_doc_id:
            if prev_doc_id != "":
                topic.add_doc(prev_doc_id, document)

            document = Document(doc_id)
            # document.add_sentence(sentence_id, sentence)  # (Orlin)
            prev_doc_id = doc_id

        # if a new topic starts (name of the topic changes)
        if topic_name != prev_topic_id:
            if prev_topic_id != "":
                # if dataset_name == "MEANTime":
                #     # prev_topic_id = f"{number_of_seen_topics}MEANTIMEcross"
                #     prev_topic_id = meantimeNameConverter[topic_name]
                #     number_of_seen_topics += 1
                corpus.add_topic(prev_topic_id, topic)
            topic = Topic(topic_name)
            prev_topic_id = topic_name



        # after we run through all the data we just save the last topic.
    if prev_topic_id not in corpus.topics:
        # if dataset_name == "MEANTime":
        #     topic_name = meantimeNameConverter[topic_name]  # f"{number_of_seen_topics}MEANTIMEcross"
        # document.add_sentence(prev_sentence_id, sentence)

        topic.add_doc(doc_id, document)
        if prev_sentence_id not in document.sentences:
            document.add_sentence(prev_sentence_id, sentence)

        # if prev_doc_id not in topic.docs:
        #  topic.add_doc(prev_doc_id,document)

        corpus.add_topic(topic_name, topic)

    return corpus


def _create_corpus(data_conll, entity_mentions, event_mentions):

    '''
    Creates a corpus
    :param data_conll: conll file 
    :param entity_mentions: JSON file of the entity mentions in the given conll file
    :param event_mentions: JSON file of the event mentions in the given conll file
    :return: JSON object of the corpus
    '''

    # data in conll format
    corpus = conll_to_corpus(data_conll)

    # ## Add mentions to the sentences of the corpus ## 
    corpus = load_mentions_from_json(corpus, entity_mentions, event_mentions)  

    if CONFIG['use_srl']:
        srl_data = get_srl_data(corpus)
        match_allen_srl_structures(corpus, srl_data, True)
    if CONFIG['use_dep']:
        find_args_by_dependency_parsing(corpus, is_gold=True)
    if CONFIG['wiggle']:
        find_left_and_right_mentions(corpus, is_gold=True)

    elmo_embedder = ElmoEmbedding(CONFIG['elmo_options_file'], CONFIG['elmo_weight_file'])
    corpus = load_elmo_embeddings(corpus, elmo_embedder, set_pred_mentions=True)

    return corpus



## Add mentions to the sentences of the corpus ##

def load_mentions_from_json(corpus, entity_mentions, event_mentions):

    '''
    Adds the entity and event mentions to the corresponding sentences of the corpus
    :param corpus: object of Type corpus - with hierarchy of Corpus/Topic/Doc/Sentence/Token .. 
    :param entity_mentions: JSON file of the entity mentions to be added in the corpus
    :param event_mentions: JSON file of the event mentions -||-
    :return: JSON object of the corpus
    '''

    nlp = spacy.load("en_core_web_sm")

    for entity_mention in tqdm(entity_mentions, desc="Handling all entity mentions"):
        corpus = _createMention(corpus, entity_mention, nlp, 'entity')


    for event_mention in tqdm(event_mentions, desc="Handling all event mentions"):
        corpus = _createMention(corpus, event_mention, nlp, 'event')


    return corpus


def _createMention(corpus, entry, nlp, eventOrEntity):

    sentence_found = False
    
    if type(entry) !=dict:
        entry = json.loads(entry)

    doc_id = entry["conll_doc_key"].split("/")[-1]
        
    
    sent_id = int(f"{entry['sent_id']}")
    tokens_number = entry['tokens_number']
    tokens = []
    mention_str = entry['tokens_str']
    is_continuous = entry['is_continuous']
    is_singleton = entry['is_singleton']
    mention_type = entry['mention_type']
    coref_chain = entry['coref_chain']
    topic_id = entry["topic_id"]

    topic_name = str(topic_id) + "/" + str(entry["subtopic"])

    unique_doc_id = entry["conll_doc_key"]

    for topic_name in corpus.topics:                                               #    Orlin: I don't think we need a for loop here ?
        try:
           
            sentence = corpus.topics[topic_name].docs[doc_id].sentences[sent_id]

            sentence_found = True
            print("Is sentence found?  "+str(sentence_found)) 
            for token_number in tokens_number:
                tokens.append(sentence.tokens[token_number])
            #break
        except:
            print("There is no sentence in our corpus where this mention belong to")
            continue  
    
    """
        The sentence_found is necessary, since the mention_.json also holds the mentions of topics we do not evaluate
        while training / testing. In order to avoid this, we will just ignore it in this case.
        And for some data sets - for example MEANTime - there is no direct connection between the topic id and the
        document id. So we are unable to get the topic (unless with a giant list of all documents of all topics) and 
        therefore can't say if we found a mention.
    """

    if sentence_found:
        head_text = ""
        head_lemma = ""
        token_strings = " ".join([token.get_token() for token in tokens])
        doc = nlp(token_strings)
        for token in doc:

            if token.dep_ == 'ROOT':
                head_text = token.text
                head_lemma = token.lemma_
                break

        if eventOrEntity == 'entity':
            mention = EntityMention(unique_doc_id, sent_id, tokens_number, tokens, mention_str,
                                    head_text, head_lemma, is_singleton, is_continuous, coref_chain, mention_type)
            #sentence.add_gold_mention(mention, False)

            
        elif eventOrEntity == 'event':
            mention = EventMention(unique_doc_id, sent_id, tokens_number, tokens, mention_str,
                                   head_text, head_lemma, is_singleton, is_continuous, coref_chain)
            #print(mention.__str__()+" was added")   
        
        print(mention.__str__()+" was added")                   
        sentence.add_gold_mention(mention, eventOrEntity)
        corpus.topics[topic_name].docs[doc_id].sentences[sent_id] = sentence  # Update the changed sentence in the topic
        corpus.topics[topic_name].add_gold_mention(mention, eventOrEntity)   # add the mention in the topic
        
        return corpus  


    return None