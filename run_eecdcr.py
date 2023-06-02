import json
import os.path
import subprocess
from datetime import datetime

import torch

from EECDCR.all_models.model_utils import test_models
from shared.CONSTANTS import EECDCR_CONFIG_DICT, CONFIG


def load_entity_wd_clusters(config_dict):
    '''
    Loads from a file the within-document (WD) entity coreference clusters predicted by an external WD entity coreference
    model/tool and ordered those clusters in a dictionary according to their documents.
    :param config_dict: a configuration dictionary that contains a path to a file stores the
    within-document (WD) entity coreference clusters predicted by an external WD entity coreference
    system.
    :return: a dictionary contains a mapping of a documents to their predicted entity clusters
    '''
    doc_to_entity_mentions = {}

    with open(config_dict["wd_entity_coref_file"].format(CONFIG['dataset_name']), 'r') as js_file:
        js_mentions = json.load(js_file)

    # load all entity mentions in the json
    for js_mention in js_mentions:
        doc_id = js_mention["doc_id"].replace('.xml', '')
        if doc_id not in doc_to_entity_mentions:
            doc_to_entity_mentions[doc_id] = {}
        sent_id = js_mention["sent_id"]
        if sent_id not in doc_to_entity_mentions[doc_id]:
            doc_to_entity_mentions[doc_id][sent_id] = []
        tokens_numbers = js_mention["tokens_numbers"]
        mention_str = js_mention["tokens_str"]

        try:
            coref_chain = js_mention["coref_chain"]
        except:
            continue

        doc_to_entity_mentions[doc_id][sent_id].append((doc_id, sent_id, tokens_numbers,
                                                        mention_str, coref_chain))
    return doc_to_entity_mentions


def _load_check_point(fname):
    '''
    Loads Pytorch model from a file
    :param fname: model's filename
    :return:Pytorch model
    '''
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    return torch.load(fname, map_location=torch.device(device))


def test_model(corpus, model_train_dir,output_dir, document_clustering):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    device = 'cuda:0' if torch.cuda.is_available() else "cpu"


    #cd_event_model = _load_check_point(EECDCR_CONFIG_DICT["cd_event_model_path"])
    #cd_entity_model = _load_check_point(EECDCR_CONFIG_DICT["cd_entity_model_path"])

    cd_event_model = _load_check_point(model_train_dir+"/cd_event_best_model")
    cd_entity_model = _load_check_point(model_train_dir+"/cd_entity_best_model")
  

    cd_event_model.to(device)
    cd_entity_model.to(device)

    doc_to_entity_mentions = load_entity_wd_clusters(EECDCR_CONFIG_DICT)
    event_b3_f1, entity_b3_f1, all_entity_clusters, all_event_clusters = test_models(document_clustering, corpus, cd_event_model, cd_entity_model, device,
                                            EECDCR_CONFIG_DICT, write_clusters=True, out_dir=output_dir,
                                            doc_to_entity_mentions=doc_to_entity_mentions, analyze_scores=False, )
    
    return event_b3_f1, entity_b3_f1, all_entity_clusters, all_event_clusters


def run_conll_scorer(out_dir):
    if EECDCR_CONFIG_DICT["test_use_gold_mentions"]:
        event_response_filename = os.path.join(out_dir, 'CD_test_event_mention_based.response_conll')
        entity_response_filename = os.path.join(out_dir, 'CD_test_entity_mention_based.response_conll')
    else:
        event_response_filename = os.path.join(out_dir, 'CD_test_event_span_based.response_conll')
        entity_response_filename = os.path.join(out_dir, 'CD_test_entity_span_based.response_conll')

    event_conll_file = os.path.join(out_dir, 'event_scorer_cd_out.txt')
    entity_conll_file = os.path.join(out_dir, 'entity_scorer_cd_out.txt')

    event_scorer_command = (
        'perl ./scorer/scorer.pl all {} {} none > {} \n'.format(EECDCR_CONFIG_DICT["event_gold_file"].format(CONFIG['dataset_name']),
                                                                event_response_filename,
                                                                event_conll_file))

    entity_scorer_command = (
        'perl ./scorer/scorer.pl all {} {} none > {} \n'.format(EECDCR_CONFIG_DICT["entity_gold_file"].format(CONFIG['dataset_name']),
                                                                entity_response_filename,
                                                                entity_conll_file))

    processes = []
    print('Run scorer command for cross-document event coreference')
    processes.append(subprocess.Popen(event_scorer_command, shell=True))

    print('Run scorer command for cross-document entity coreference')
    processes.append(subprocess.Popen(entity_scorer_command, shell=True))

    while processes:
        status = processes[0].poll()
        if status is not None:
            processes.pop(0)

    print('Running scorers has been done.')
    print('Save results...')

    scores_file = open(os.path.join(out_dir, 'conll_f1_scores.txt'), 'w')

    event_f1 = read_conll_f1(event_conll_file)
    entity_f1 = read_conll_f1(entity_conll_file)
    scores_file.write('Event CoNLL F1: {}\n'.format(event_f1))
    scores_file.write('Entity CoNLL F1: {}\n'.format(entity_f1))

    scores_file.close()

    return


def read_conll_f1(filename):
    """
    This function reads the results of the CoNLL scorer , extracts the F1 measures of the MUS,
    B-cubed and the CEAF-e and calculates CoNLL F1 score.
    :param filename: a file stores the scorer's results.
    :return: the CoNLL F1
    """
    f1_list = []
    print("filename in read_conll_f1", filename)
    with open(filename, "r") as ins:
        for line in ins:
            new_line = line.strip()
            if new_line.find('F1:') != -1:
                f1_list.append(float(new_line.split(': ')[-1][:-1]))

    muc_f1 = f1_list[1]
    bcued_f1 = f1_list[3]
    ceafe_f1 = f1_list[7]

    return (muc_f1 + bcued_f1 + ceafe_f1) / float(3)
