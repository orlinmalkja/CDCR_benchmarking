import os
from collections import defaultdict
import spacy

from allennlp.predictors import SemanticRoleLabelerPredictor

from typing import Dict, List

from features.allen_srl_reader import SRLSentence, SRLVerb, SRLArg
from features.extraction_utils import match_event, match_entity, getAllSubs, getAllObjs, match_subj_with_event, \
    find_nominalizations_args
from shared.classes import Corpus
from shared.CONSTANTS import CONFIG
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')
matched_args = 0
matched_args_same_ix = 0

matched_events = 0
matched_events_same_ix = 0

from pipeline import WORKING_DIR

#os.chdir(WORKING_DIR)


def get_srl_data(corpus: Corpus) -> Dict[str, Dict[str, SRLSentence]]:
    """
    Extracts labels from semantic role labeling (SRL).

    Args:
        corpus: A EECDCE document collection object.

    Returns:
        A dictionary with EECDCR SRL sentence structures.

    """
    
    if not os.path.exists(CONFIG['bert_file']):
        raise Exception("Bert Model was not found.")

    predictor = SemanticRoleLabelerPredictor.from_path(CONFIG['bert_file'])

    srl_data = defaultdict(lambda: defaultdict(SRLSentence))
    for topic in tqdm(list(corpus.topics.values()), desc='Get SRL data'):
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.sentences.items():
                srl_sent = SRLSentence(doc_id, sent_id)
                srl = predictor.predict_tokenized([t.token for t in sent.tokens])

                for verb in srl["verbs"]:
                    srl_verb_obj = SRLVerb()
                    srl_verb_obj.verb = SRLArg(verb["verb"], [srl["words"].index(verb["verb"])])

                    for tag_id, tag in enumerate(verb["tags"]):
                        for tag_type in ["ARG0", "ARG1", "TMP", "LOC", "NEG"]:
                            check_tag(tag, tag_id, srl_verb_obj, tag_type, srl["words"])
                    srl_sent.add_srl_vrb(srl_verb_obj)

                srl_data[doc_id][sent_id] = srl_sent
    return srl_data


def check_tag(tag: str, tag_id: int, srl_verb_obj: SRLVerb, attr: str, words: List[str]):
    """
    Checks tags from SRL and initialize SRL objects from EECDCR.

    Args:
        tag: A SRL tag.
        tag_id: A SRL tag id.
        srl_verb_obj: A SRL verb object from EECDCR.
        attr: An attribute for which we need to check in tags.
        words: A list of words from SRL tagger.

    """
    tag_attr_dict = {"ARG0": "arg0",
                     "ARG1": "arg1",
                     "TMP": "arg_tmp",
                     "LOC": "arg_loc",
                     "NEG": "arg_neg"}
    if attr in tag:
        if tag[0] == "B":
            setattr(srl_verb_obj, tag_attr_dict[attr], SRLArg(words[tag_id], [tag_id]))
        else:
            srl_arg = getattr(srl_verb_obj, tag_attr_dict[attr])
            if srl_arg is None:
                srl_arg = SRLArg("", [])
            srl_arg.text += " " + words[tag_id]
            srl_arg.ecb_tok_ids.append(tag_id)
            setattr(srl_verb_obj, tag_attr_dict[attr], srl_arg)


def find_args_by_dependency_parsing(dataset, is_gold):
    '''
    Runs dependency parser on the split's sentences and augments the predicate-argument structures
    :param dataset: an object represents the split (Corpus object)
    :param is_gold: whether to match arguments and predicates with gold or predicted mentions
    '''
    global matched_args, matched_args_same_ix, matched_events,matched_events_same_ix
    matched_args = 0
    matched_args_same_ix = 0
    matched_events = 0
    matched_events_same_ix = 0
    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                sent_str = sent.get_raw_sentence()
                parsed_sent = nlp(sent_str)
                findSVOs(parsed_sent=parsed_sent, sent=sent, is_gold=is_gold)

    print('matched events : {} '.format(matched_events))
    print('matched args : {} '.format(matched_args))


def findSVOs(parsed_sent, sent, is_gold):
    global matched_events, matched_events_same_ix
    global matched_args, matched_args_same_ix
    verbs = [tok for tok in parsed_sent if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    for v in verbs:
        subs, pass_subs = getAllSubs(v)
        v, objs = getAllObjs(v)
        if len(subs) > 0 or len(objs) > 0 or len(pass_subs) > 0:
            for sub in subs:
                match_subj_with_event(verb_text=v.orth_,
                                      verb_index=v.i, subj_text=sub.orth_,
                                      subj_index=sub.i, sent=sent, is_gold=is_gold)

            for obj in objs:
                match_obj_with_event(verb_text=v.orth_,
                                     verb_index=v.i, obj_text=obj.orth_,
                                     obj_index=obj.i, sent=sent, is_gold=is_gold)
            for obj in pass_subs:
                match_obj_with_event(verb_text=v.orth_,
                                     verb_index=v.i, obj_text=obj.orth_,
                                     obj_index=obj.i, sent=sent, is_gold=is_gold)

    find_nominalizations_args(parsed_sent, sent, is_gold) # Handling nominalizations


def match_obj_with_event(verb_text, verb_index, obj_text, obj_index, sent, is_gold):
    '''
    Given a verb and an object extracted by the dependency parser , this function tries to match
    the verb with an event mention and the object with an entity mention
    :param verb_text: the verb's text
    :param verb_index: the verb index in the parsed sentence
    :param obj_text: the object's text
    :param obj_index: the object index in the parsed sentence
    :param sent: an object represents the sentence (Sentence object)
    :param is_gold: whether to match with gold or predicted mentions
    '''
    event = match_event(verb_text, verb_index, sent, is_gold)
    if event is not None and event.arg1 is None:
        entity = match_entity(obj_text, obj_index, sent, is_gold)
        if entity is not None:
            if event.arg0 is not None and event.arg0 == (entity.mention_str, entity.mention_id):
                return
            if event.amloc is not None and event.amloc == (entity.mention_str, entity.mention_id):
                return
            if event.amtmp is not None and event.amtmp == (entity.mention_str, entity.mention_id):
                return
            event.arg1 = (entity.mention_str, entity.mention_id)
            entity.add_predicate((event.mention_str, event.mention_id), 'A1')


def find_left_and_right_mentions(dataset, is_gold):
    '''
    Finds for each event in the split's its closest left and right entity mentions
    and augments the predicate-argument structures
    :param dataset: an object represents the split (Corpus object)
    :param is_gold: whether to use gold or predicted mentions
    '''
    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                add_left_and_right_mentions(sent, is_gold)


def add_left_and_right_mentions(sent, is_gold):
    '''
    The function finds the closest left and right entity mentions of each event mention
     and sets them as Arg0 and Arg1, respectively.
    :param sent: Sentence object
    :param is_gold: whether to match with gold or predicted mentions
    '''
    sent_events = sent.gold_event_mentions if is_gold else sent.pred_event_mentions
    for event in sent_events:
        if event.arg0 is None:
            left_ent = sent.find_nearest_entity_mention(event, is_left=True, is_gold=is_gold)
            if left_ent is not None:
                double_arg = False
                if event.arg1 is not None and event.arg1 == (left_ent.mention_str, left_ent.mention_id):
                    double_arg = True
                if event.amloc is not None and event.amloc == (left_ent.mention_str, left_ent.mention_id):
                    double_arg = True
                if event.amtmp is not None and event.amtmp == (left_ent.mention_str, left_ent.mention_id):
                    double_arg = True

                if not double_arg:
                    event.arg0 = (left_ent.mention_str, left_ent.mention_id)
                    left_ent.add_predicate((event.mention_str, event.mention_id), 'A0')

        if event.arg1 is None:
            right_ent = sent.find_nearest_entity_mention(event, is_left=False, is_gold=is_gold)
            if right_ent is not None:
                double_arg = False
                if event.arg0 is not None and event.arg0 == (right_ent.mention_str, right_ent.mention_id):
                    double_arg = True
                if event.amloc is not None and event.amloc == (right_ent.mention_str, right_ent.mention_id):
                    double_arg = True
                if event.amtmp is not None and event.amtmp == (right_ent.mention_str, right_ent.mention_id):
                    double_arg = True
                if not double_arg:
                    event.arg1 = (right_ent.mention_str, right_ent.mention_id)
                    right_ent.add_predicate((event.mention_str, event.mention_id), 'A1')
