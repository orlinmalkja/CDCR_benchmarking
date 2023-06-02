import ast, ctypes
import requests

from flask import Flask, Response, Request, request, jsonify
import time
import uvicorn
import nest_asyncio
nest_asyncio.apply()
from fastapi import FastAPI

import pickle
from datetime import datetime
import os
from shared.classes import Token, Sentence, Document, Topic, Corpus
import torch

from EECDCR.all_models.train_model import train_model
from conll_reader import read_CoNLL
from create_wd_document_from_corpus import create_complete_wd_document
from features.build_features import match_allen_srl_structures, load_elmo_embeddings
from features.create_elmo_embeddings import ElmoEmbedding
from make_gold_files import create_gold_files_for_corpus
from run_eecdcr import test_model, run_conll_scorer
from shared.CONSTANTS import CONFIG, EECDCR_TRAIN_CONFIG_DICT, EECDCR_CONFIG_DICT
from srl_things import get_srl_data, find_args_by_dependency_parsing, find_left_and_right_mentions
import logging
import random
import numpy as np

from predict_model import _test

import spacy
import json
from tqdm import tqdm

from functions_train_API import conll_to_corpus, load_mentions_from_json, _create_corpus
import re



random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

import requests
from flask import Flask, Response, Request, request, jsonify



def shut_up_logger():
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
    logging.getLogger('urllib3.connectionpool').disabled = True
    logging.getLogger('matplotlib').disabled = True
    logging.getLogger('pytorch_transformers').disabled = True
    logging.getLogger('pytorch_pretrained_bert.modeling').disabled = True
    logging.getLogger('allennlp.common.registrable').disabled = True
    logging.getLogger('allennlp.common.from_params').disabled = True
    logging.getLogger('allennlp.data.vocabulary').disabled = True
    logging.getLogger('h5py._conv').disabled = True

     

app = Flask(__name__)
#app = FastAPI()

sample_data = [1,2,4,5]

@app.route('/health', methods=['GET'])
def check_service():
    global sample_data

    return jsonify({'data' : sample_data})

def _train(train_conll, train_entity_mentions, train_event_mentions, dev_conll, dev_entity_mentions, dev_event_mentions, model_name):

    
    print("Server: Started the training")

    # We convert the conll data (train and dev) into Corpus object and add the respective mentions to them
    # We transform the train and dev data into corpus objects which are ready to be an input to train function
   
    train_corpus = _create_corpus(train_conll, train_entity_mentions, train_event_mentions)

    dev_corpus = _create_corpus(dev_conll, dev_entity_mentions, dev_event_mentions)

    _train_out_dir = f"resources/eecdcr_models/self_trained/{model_name}/"
    if not os.path.exists(_train_out_dir):
        os.makedirs(_train_out_dir)

    document_clustering = False

    ## Start the training 

    train_start = datetime.now()
    print(f"Start model training after {str(datetime.now() - train_start)}")
    train_model(document_clustering, train_corpus, dev_corpus, _train_out_dir, EECDCR_TRAIN_CONFIG_DICT)
    print(f"Model training finished, after: {str(datetime.now() - train_start)}")

    
    metrics_file = f"{_train_out_dir}/summary.txt"
    ## Load the metrics on the dev set
    f = open(metrics_file, "r")
    metrics_dev_set = f.read()

    response = {"train_out_dir": _train_out_dir, "metrics_dev":metrics_dev_set}

    return response


## APIs ## 

@app.route("/train", methods=['POST'])
def train():

    # getting the passed data
    data = request.get_json()['data']
    train_conll, train_entity_mentions, train_event_mentions, dev_conll, dev_entity_mentions, dev_event_mentions, model_name = data[0], data[1], data[2], data[3], data[4], data[5], data[6]
    
    response = _train(train_conll, train_entity_mentions, train_event_mentions, dev_conll, dev_entity_mentions, dev_event_mentions, model_name)

    return response

def assign_predicted_coref_chain(test_entity_mentions, test_event_mentions, all_entity_clusters, all_event_clusters):


    """ Assign the predicted coref chain of the mentions in predicted clusters i.e all_entity_clusters, all_event_clusters to the corresponding mentions provided as test data i.e test_entity_mentions  and test_event_mentions
        We want to get the mention_id of  the passed mention via API call, so that we can find it in the predicted cluster. 
        We extract the start_offset, end_offset etc, in order to create the mention_id, which we use to find that mention in the predicted cluster
        After having found the mention in the predicted_cluster, then we assign the predicted coref chain to it

    Parameters:
    test_entity_mentions (list of JSON objects):   List of test entity mentions which are pushed from the client
    test_event_mentions  (list of JSON objects):   List of test event mentions  -||-
    all_entity_clusters  (Cluster object):         List of predicted clusters which contain entity mentions having predicted coref chain
    all_event_clusters   (Cluster object):         List of predicted clusters which have event mentions having predicted coref chain

    Returns:
    list of JSON objects: test_entity_mentions  
    list of JSON objects: test_event_mentions
     
    """

    mentions = []
  
    for mention in test_entity_mentions:
        start_offset, end_offset = mention["tokens_number"][0], mention["tokens_number"][-1]
        sent_id, doc_id = mention["sent_id"], mention["conll_doc_key"]
        mention_id = '_'.join([doc_id, str(sent_id), str(start_offset), str(end_offset)])  

        pred_coref_chain = ""

        for entity_cluster in all_entity_clusters:
            all_entity_mentions = entity_cluster.mentions ## a dictionary of mentions: (mention_id, Mention object)
            if mention_id in all_entity_mentions:
                pred_coref_chain = all_entity_mentions[mention_id].cd_coref_chain
                mention["coref_chain_predicted"] = str(pred_coref_chain)
                print("Assigned the predicted cd coref chain to the mention")
                break
            
    for mention in test_event_mentions:
        start_offset, end_offset = mention["tokens_number"][0], mention["tokens_number"][-1]
        sent_id, doc_id = mention["sent_id"], mention["conll_doc_key"]
        mention_id = '_'.join([doc_id, str(sent_id), str(start_offset), str(end_offset)])

        pred_coref_chain = ""

        for event_cluster in all_event_clusters:
            all_event_mentions = event_cluster.mentions ## a dictionary of mentions: (mention_id, Mention object)
            if mention_id in all_event_mentions:
                pred_coref_chain = all_event_mentions[mention_id].cd_coref_chain
                mention["coref_chain_predicted"] = str(pred_coref_chain)
                print("Assigned the predicted cd coref chain to the mention")
                break
            
        
    return test_entity_mentions, test_event_mentions



@app.route("/test", methods=['POST'])
def test():

    # getting the passed data  ##
    data = request.get_json()['data']
    test_conll, test_entity_mentions, test_event_mentions, train_model_dir, document_clustering = data[0], data[1], data[2], data[3], data[4]

    ## ---------------------------------- ##
    test_corpus =  _create_corpus(test_conll, test_entity_mentions, test_event_mentions)
    
    ## Getting the predicted entity and event clusters ##
    all_entity_clusters, all_event_clusters = _test_API(test_corpus, train_model_dir, document_clustering)

    ## assign predicted coref chain to the mentions (entity and event)  which were passed via API ##
    test_entity_mentions, test_event_mentions = assign_predicted_coref_chain(test_entity_mentions, test_event_mentions,all_entity_clusters, all_event_clusters)

    ## --------------------------------- ##

    return {"Entity mentions":test_entity_mentions,"Event mentions": test_event_mentions}



def _test_API(test_corpus, train_model_dir, document_clustering):
    print('Entered the test')
    out_dir = f"data/output/{datetime.today().strftime('%Y-%m-%d')}/{CONFIG['dataset_name']}/{datetime.now().strftime('%H-%M')}/"
    print('Creating test corpus..')
    print("test corpus\n",test_corpus)

    if not os.path.exists(f"pickle_data"):
        os.mkdir(f"pickle_data")

    # ToDO: fix this with the dataset name
    if not os.path.exists(f"pickle_data/{CONFIG['dataset_name']}"):
        os.mkdir(f"pickle_data/{CONFIG['dataset_name']}")

    with open(f"pickle_data/{CONFIG['dataset_name']}/test_corpus.pickle", "wb") as f:
        pickle.dump(test_corpus, f)

    print('Creating gold files..')
    create_gold_files_for_corpus(test_corpus)

    print('Testing the model..')
    _, _ , all_entity_clusters, all_event_clusters = test_model(test_corpus, train_model_dir, out_dir, document_clustering)

    run_conll_scorer(out_dir)

    return all_entity_clusters, all_event_clusters


    

@app.route("/create_train_dev_data", methods=['GET'])
def create_train_dev_data():
    global _train_corpus, _dev_corpus

    _train_corpus = _create_corpus("train")
    _dev_corpus = _create_corpus("dev")

    with open(f"cache/train_corpus.p", "wb") as f:
         pickle.dump(_train_corpus, f)
    
    with open(f"cache/dev_corpus.p", "wb") as f:
         pickle.dump(_dev_corpus, f)

    return "Created the train and dev corpus successfully"



if __name__ == '__main__':

    #uvicorn.run(app,host="127.0.0.1", port=5000)
    app.run(debug = True,use_reloader=False)
 

    
