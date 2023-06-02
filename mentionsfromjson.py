import os.path

from shared.classes import Corpus, EntityMention, EventMention
from shared.CONSTANTS import mentions_path, CONFIG, EECDCR_CONFIG_DICT
import json
import spacy
from tqdm import tqdm


# def _save_coref_mentions(mentions):
#     coref_chain = 49999
#     mention_dict = {}
#     for mention in mentions:
#         key = f"{mention.doc_id}_{mention.gold_tag}"
#         if key not in mention_dict:
#             mention_dict[key] = [mention]
#         else:
#             mention_dict[key].append(mention)
#
#     cand_list = []
#     for key in mention_dict:
#         if len(mention_dict[key]) == 1:
#             continue
#
#         coref_chain += 1
#
#         for cand in mention_dict[key]:
#             cand_list.append({
#                 "doc_id": cand.doc_id,
#                 "sent_id": cand.sent_id,
#                 "tokens_numbers": cand.tokens_numbers,
#                 "tokens_str": cand.mention_str,
#                 "coref_chain": f"{coref_chain}",
#                 "is_continuous": cand.is_continuous,
#                 "is_singleton": False
#             })
#
#     """
#         In the case of training we want to append the dev clusters to the train cluster.
#         But in the case of testing we always want to only have the momentarily necessary document clusters.
#     """
#     if not CONFIG['test']:
#         if os.path.exists(EECDCR_CONFIG_DICT["wd_entity_coref_file"].format(CONFIG['dataset_name'])):
#             with open(EECDCR_CONFIG_DICT["wd_entity_coref_file"], "r") as f:
#                 data = json.load(f)
#             for entry in data:
#                 cand_list.append(entry)
#
#     with open(EECDCR_CONFIG_DICT["wd_entity_coref_file"].format(CONFIG['dataset_name']), "w") as f:
#         json.dump(cand_list, f, indent=1)


def _attachMentionToCorpus(mention, corpus: Corpus, is_event):
    doc_id = mention.doc_id
    sent_id = f"{mention.sent_id}"

    if CONFIG['dataset_name'] == "MEANTime":
        doc_id = doc_id.split("_")[0]

    for topic in corpus.topics:
        try:
            sentence = corpus.topics[topic].docs[doc_id].sentences[sent_id]

            for token_number in mention.tokens_numbers:
                mention.add_token(sentence.tokens[token_number])

            sentence.add_gold_mention(mention, is_event)
            break
        except:
            continue

    return corpus


def _createMention(corpus, entry, nlp, eventOrEntity):
    sentence_found = False
    doc_id = entry['doc_id']
    sent_id = f"{entry['sent_id']}"
    tokens_number = entry['tokens_number']
    tokens = []
    mention_str = entry['tokens_str']
    is_continuous = entry['is_continuous']
    is_singleton = entry['is_singleton']
    mention_type = entry['mention_type']
    coref_chain = entry['coref_chain']
    topic_id = entry["topic_id"]


    if CONFIG['dataset_name'] == "ECB+":
        topic_nr = topic_id.split("e")[0]
        topic_id = topic_nr+"/" + topic_id  # the topic id would be like 1/1ecb now
        doc_id = doc_id.split("_")[1]  # get the second part of doc_id (10ecb)
        doc_id = doc_id.split("e")[0]  # get the actual doc number, which is 10
        unique_doc_id = topic_id+"/"+doc_id   # topic/subtopic/doc ids

    for topic in corpus.topics:                                               #    Orlin: I don't think we need a for loop here ?
        try:
            sentence = corpus.topics[topic_id].docs[doc_id].sentences[sent_id]
            sentence_found = True
            for token_number in tokens_number:
                tokens.append(sentence.tokens[token_number])
            #break
        except:
                #print("There is no sentence in our corpus where this mention belong to")
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
            sentence.add_gold_mention(mention, False)
            return mention
        elif eventOrEntity == 'event':
            mention = EventMention(unique_doc_id, sent_id, tokens_number, tokens, mention_str,
                                   head_text, head_lemma, is_singleton, is_continuous, coref_chain)
            sentence.add_gold_mention(mention, True)
            return mention
    return None


def loadMentionsFromJson(corpus: Corpus = None):
    dataset_name = CONFIG['dataset_name']
    nlp = spacy.load("en_core_web_sm")
    entity_path = mentions_path[dataset_name] + "/entity_mentions_.json"
    event_path = mentions_path[dataset_name] + "/event_mentions_.json"
    all_mentions = []

    with open(entity_path, "r", encoding="utf8") as f:
        entity_mentions = json.load(f)

    with open(event_path, "r", encoding="utf8") as f:
        event_mentions = json.load(f)

    """
        If you don't want to use singletons every mention which is a singleton ('is_singleton': True) is removed
    """
    if not CONFIG['use_singletons']:
        entity_mentions = [x for x in entity_mentions if not x['is_singleton']]
        event_mentions = [x for x in event_mentions if not x['is_singleton']]

    for entity_mention in tqdm(entity_mentions, desc="Handling all entity mentions"):
        mention = _createMention(corpus, entity_mention, nlp, 'entity')

        if mention is not None:
            all_mentions.append(mention)
        # corpus = _attachMentionToCorpus(mention, corpus, False)

    for event_mention in tqdm(event_mentions, desc="Handling all event mentions"):
        mention = _createMention(corpus, event_mention, nlp, 'event')
        if mention is not None:
            all_mentions.append(mention)
        # corpus = _attachMentionToCorpus(mention, corpus, True)

    return corpus
