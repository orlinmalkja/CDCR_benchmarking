import os
from setup import *
from logger import LOGGER
import pandas as pd
import re
from tqdm import tqdm
import torch
import math
import random
import requests
import json

from shared.CONSTANTS import CONFIG, EECDCR_TRAIN_CONFIG_DICT, EECDCR_CONFIG_DICT


pd.set_option('display.max_columns', None)

WORKING_DIR = os.path.join(os.getcwd())
DOWNLOAD_DIR = os.path.join(WORKING_DIR, DOWNLOAD_DIR_NAME)


from functions_pipeline import *
from functions import *





def train_model(experiment_config, train_conll, train_event_mentions, train_entity_mentions , dev_conll, dev_event_mentions, dev_entity_mentions, model_name):
    pass
    # TODO: Implement this function using the API (should return a trained model)
    # Save the trained model to experiment_config["model_name_train"]

    ##### Train API #####
    payload_train = {"data":[train_conll, train_entity_mentions, train_event_mentions, dev_conll, dev_entity_mentions, dev_event_mentions, model_name ]}
    response = requests.post("http://localhost:5000/train",json=payload_train)

    train_out_dir = json.loads(response.text)["train_out_dir"]
    #experiment_config["model_name_train"] = response["train_out_dir"]
    print("Response \n",response.text)
    print("Finished the train API")
    return train_out_dir

def test_model(experiment_config, test_conll, test_entity_mentions, test_event_mentions, train_model_dir, document_clustering):
    
    # TODO: Implement this function using the API
    # Test the model file experiment_config["model_name_test"]
    # if none, test the model that has just been trained before

    # Orlin: For testing purposes, the test mentions are same as dev mentions :)

    ##### Test API #####
    #train_model_dir = experiment_config["model_name_train"]
    payload_test = {"data":[test_conll, test_entity_mentions, test_event_mentions, train_model_dir, document_clustering]}

    response = requests.post("http://localhost:5000/test",json=payload_test)

    print("Response \n",response.text)
    print("Finished testing")


if __name__ == '__main__':
    LOGGER.info("Welcome to the CDCR-Benchmarking Pipeline. \n-----------------")
    
    with open(DATASET_CONFIG_FILE, "r") as f:
        config = json.loads(f.read())

    is_custom_dataset = False   # True if one provides conll file & mentions file

    while True:
        try:
            LOGGER.info(
                "Please choose how to proceed: \n0) process a single dataset \n1) use a experiment config file (recommended)\n")
            a = input()
            assert a == "0" or a == "1"
            break
        except (ValueError, AssertionError) as e:
            LOGGER.warning("Oops! That input was not correct (1 or 2). Please retry by typing a number.")
    # ----> DATASET SELECTION
    if int(a) == 0:
        LOGGER.info(
            "Please choose a dataset to process. \nYou may \n-> use your own dataset by providing conll- and mentions-files \nor \n-> use one of our standard datasets we provide.")
        while True:
            try:
                LOGGER.info(
                    "Please choose how to proceed: \n0) provide conll-file & mentions-file \n1) choose a dataset from our list\n")
                a = input()
                assert a == "0" or a == "1"
                break
            except (ValueError, AssertionError) as e:
                LOGGER.warning("Oops! That input was not correct (1 or 2). Please retry by typing a number.")
        if int(a) == 0:

            is_custom_dataset = True

            # Import the train, dev and test conll, entity mentions, event mentions
            train_conll, train_entity_mentions, \
                 train_event_mentions, dev_conll, dev_entity_mentions, dev_event_mentions,\
                  test_conll, test_entity_mentions , test_event_mentions    = import_dataset()
            LOGGER.info("Imported the train, dev and test conll, entity mentions, event mentions")
        elif int(a) == 1:
            dataset, dataset_config_index = choose_dataset(config)

            LOGGER.info("Setup done.")

            # ---> Reading Data
            conll_df = read_conll(dataset["name"])
            mentions_df_list = []
            for i, mentions_url in enumerate(dataset["mentions_url"]):
                filename = mentions_url.split("/")[-1]
                mentions_df = read_mentions(dataset["name"], filename)
                mentions_df_list.append(mentions_df)

            # ----> Splitting Data
            dataset, \
            train_conll_df, test_conll_df, dev_conll_df, \
            train_mentions_df_list, test_mentions_df_list, dev_mentions_df_list \
                = split_menu(dataset, conll_df, mentions_df_list, dataset_config_index, config)

            train_event_mentions_df_w_singletons, train_entity_mentions_df_w_singletons = train_mentions_df_list[0], train_mentions_df_list[1]
            dev_event_mentions_df_w_singletons , dev_entity_mentions_df_w_singletons = dev_mentions_df_list[0], dev_mentions_df_list[1]
            test_event_mentions_df_w_singletons, test_entity_mentions_df_w_singletons = test_mentions_df_list[0], test_mentions_df_list[1]

            # lets include train, dev, test conll dataframes and the corresponding event and entity mentions into a dict called all_datasets_dict
            all_datasets_dict = {
                 "train_conll_df":train_conll_df,
                 "train_event_mentions_df": train_mentions_df_list[0],
                 "train_entity_mentions_df": train_mentions_df_list[1],
                 "dev_conll_df": dev_conll_df,
                 "dev_event_mentions_df": dev_mentions_df_list[0],
                 "dev_entity_mentions_df": dev_mentions_df_list[1],
                 "test_conll_df": test_conll_df,
                 "test_event_mentions_df": test_mentions_df_list[0],
                 "test_entity_mentions_df": test_mentions_df_list[1]
                 }
            
              
            # Append only dataset to total data
            total_train_conll_df = train_conll_df
            total_test_conll_df = test_conll_df
            total_dev_conll_df = dev_conll_df
            total_train_mentions_df = pd.DataFrame()
            total_test_mentions_df = pd.DataFrame()
            total_dev_mentions_df = pd.DataFrame()

            for train_mentions_df in train_mentions_df_list:
                total_train_mentions_df = pd.concat([total_train_mentions_df, train_mentions_df])
            
            for test_mentions_df in test_mentions_df_list:
                total_test_mentions_df = pd.concat([total_test_mentions_df, test_mentions_df])
            
            for dev_mentions_df in dev_mentions_df_list:
                total_dev_mentions_df = pd.concat([total_dev_mentions_df, dev_mentions_df])

            

            LOGGER.debug(f"Proceeding with this dataset config: \n {json.dumps(dataset, indent=4)}")

            experiment_config = {
                "train_datasets": [
                    {
                        "name": dataset["name"],
                        "given_split": True
                    }
                ],
                "test_datasets": [
                    {
                        "name": dataset["name"],
                        "given_split": True
                    }
                ],
                "dev_datasets": [
                    {
                        "name": dataset["name"],
                        "given_split": True
                    }
                ],
                "singletons": {
                    "train": None,
                    "test": None,
                    "dev": None
                },
                "model_name_train": "experiment_"+dataset["name"]+"_"+f'{random.randrange(1, 10**3):03}',
                "model_name_test": None,
                "evaluation_granularity": {     # TODO: Add menu for user to choose granularity
                    "dataset": False,
                    "topic": True,
                    "subtopic": False
                },
                "evaluation_metrics": {     # TODO: Add menu for user to choose metric
                    "old_conll": True,
                    "conll_lea": False
                },
                "metric_aggregation": {     # TODO: Add menu for user to choose metric aggregation
                    "separate_mention_types": True,
                    "combined_mention_types": True
                }
            }

            # User chooses singleton inclusion
            while True:
                try:
                    LOGGER.info(
                        "Do you want to remove singletons in the train split? (y/n)\n")
                    a = input().lower()
                    assert a == "y" or a == "n"
                    break
                except (ValueError, AssertionError) as e:
                    LOGGER.warning("Oops! That input was not correct (y/n). Please retry.")
            experiment_config["singletons"]["train"] = (a == "y")
            while True:
                try:
                    LOGGER.info(
                        "Do you want to remove singletons in the test split? (y/n)\n")
                    a = input().lower()
                    assert a == "y" or a == "n"
                    break
                except (ValueError, AssertionError) as e:
                    LOGGER.warning("Oops! That input was not correct (y/n). Please retry.")
            experiment_config["singletons"]["test"] = (a == "y")
            while True:
                try:
                    LOGGER.info(
                        "Do you want to remove singletons in the dev split? (y/n)\n")
                    a = input().lower()
                    assert a == "y" or a == "n"
                    break
                except (ValueError, AssertionError) as e:
                    LOGGER.warning("Oops! That input was not correct (y/n). Please retry.")
            experiment_config["singletons"]["dev"] = (a == "y")

            # export the config created by the users wishes
            with open("experiment_config_updated.json", "w") as f:
                json.dump(experiment_config, f)

    # ----> EXPERIMENT CONFIG
    elif int(a) == 1:
        with open(EXPERIMENT_CONFIG_FILE, "r") as f:
            LOGGER.info(f"Reading {EXPERIMENT_CONFIG_FILE}...")
            experiment_config = json.loads(f.read())
            LOGGER.info(f"Using the following datasets for this experiment:")
            LOGGER.info(f"Train: {[dataset['name'] for dataset in experiment_config['train_datasets']]}")
            LOGGER.info(f"Test: {[dataset['name'] for dataset in experiment_config['test_datasets']]}")
            LOGGER.info(f"Dev: {[dataset['name'] for dataset in experiment_config['dev_datasets']]}")

            total_train_conll_df = pd.DataFrame()
            total_test_conll_df = pd.DataFrame()
            total_dev_conll_df = pd.DataFrame()
            total_train_mentions_df = pd.DataFrame()
            total_test_mentions_df = pd.DataFrame()
            total_dev_mentions_df = pd.DataFrame()

            
            LOGGER.info("Processing train split...")
            for exp_dataset in experiment_config['train_datasets']:
                LOGGER.info(f"-> {exp_dataset['name']}")

                # ---> Reading Data
                dataset = [ds for ds in config["datasets"] if ds["name"] == exp_dataset["name"]][0]
                download_dataset(dataset)
                conll_df = read_conll(dataset["name"])
                mentions_df_list = []
                for i, mentions_url in enumerate(dataset["mentions_url"]):
                    filename = mentions_url.split("/")[-1]
                    mentions_df = read_mentions(dataset["name"], filename)
                    mentions_df_list.append(mentions_df)

                # ---> Decide on using papers split or default random split
                if exp_dataset["given_split"] == False:
                    LOGGER.info(
                        "Don't use a given split for this dataset. The whole dataset will be assigned to this split.")
                    train_conll_df = conll_df
                    train_mentions_df_list = mentions_df_list
                else:
                    LOGGER.info("Use the given split for this dataset. Using the split provided in datasets_config.json.")
                    train_conll_df, _, _, train_mentions_df_list, _, _ = split_dataset(dataset, conll_df,
                                                                                       mentions_df_list, next(
                            iter(dataset["split_topics"].keys())))

                # ---> Concat dataset split to total splits
                total_train_conll_df = pd.concat([total_train_conll_df, train_conll_df])
                for train_mentions_df in train_mentions_df_list:
                    total_train_mentions_df = pd.concat([total_train_mentions_df, train_mentions_df])

            LOGGER.info("Processing test split...")
            for exp_dataset in experiment_config['test_datasets']:
                LOGGER.info(f"-> {exp_dataset['name']}")

                # ---> Reading Data
                dataset = [ds for ds in config["datasets"] if ds["name"] == exp_dataset["name"]][0]
                download_dataset(dataset)
                conll_df = read_conll(dataset["name"])
                mentions_df_list = []
                for i, mentions_url in enumerate(dataset["mentions_url"]):
                    filename = mentions_url.split("/")[-1]
                    mentions_df = read_mentions(dataset["name"], filename)
                    mentions_df_list.append(mentions_df)

                # ---> Decide on using papers split or default random split
                if exp_dataset["given_split"] == False:
                    LOGGER.info(
                        "Don't use a given split for this dataset. The whole dataset will be assigned to this split.")
                    test_conll_df = conll_df
                    test_mentions_df_list = mentions_df_list
                else:
                    LOGGER.info("Use the given split for this dataset. Using the default split ratio.")
                    _, test_conll_df, _, _, test_mentions_df_list, _ = split_dataset(dataset, conll_df,
                                                                                     mentions_df_list, next(
                            iter(dataset["split_topics"].keys())))

                # ---> Concat dataset split to total splits
                total_test_conll_df = pd.concat([total_test_conll_df, test_conll_df])
                for test_mentions_df in test_mentions_df_list:
                    total_test_mentions_df = pd.concat([total_test_mentions_df, test_mentions_df])

            
            LOGGER.info("Processing dev split...")
            for exp_dataset in experiment_config['dev_datasets']:
                LOGGER.info(f"-> {exp_dataset['name']}")

                # ---> Reading Data
                dataset = [ds for ds in config["datasets"] if ds["name"] == exp_dataset["name"]][0]
                download_dataset(dataset)
                conll_df = read_conll(dataset["name"])
                mentions_df_list = []
                for i, mentions_url in enumerate(dataset["mentions_url"]):
                    filename = mentions_url.split("/")[-1]
                    mentions_df = read_mentions(dataset["name"], filename)
                    mentions_df_list.append(mentions_df)

                # ---> Decide on using papers split or default random split
                if exp_dataset["given_split"] == False:
                    LOGGER.info(
                        "Don't use a given split for this dataset. The whole dataset will be assigned to this split.")
                    dev_conll_df = conll_df
                    dev_mentions_df_list = mentions_df_list
                else:
                    LOGGER.info("Use the given split for this dataset. Using the default split ratio.")
                    _, _, dev_conll_df, _, _, dev_mentions_df_list = split_dataset(dataset, conll_df,
                                                                                   mentions_df_list, next(
                            iter(dataset["split_topics"].keys())))

                # ---> Concat dataset split to total splits
                total_dev_conll_df = pd.concat([total_dev_conll_df, dev_conll_df])
                for dev_mentions_df in dev_mentions_df_list:
                    total_dev_mentions_df = pd.concat([total_dev_mentions_df, test_mentions_df])




    # ----> Converting the data to string conll, and list of JSON objets for the mentions
            #       Reason: The format that our APIs accept
    
    """   Orlin: Commented this for now
    if is_custom_dataset == False:   # the user doesn't provide conll file and mentions file
        train_conll = conll_df_to_conll_str(all_datasets_dict["train_conll_df"])
        dev_conll = conll_df_to_conll_str(all_datasets_dict["dev_conll_df"])
        test_conll = conll_df_to_conll_str(all_datasets_dict["test_conll_df"])

        # ----> Converting the entities and event mentions to lists of JSON objects
        train_event_mentions = mentions_df_to_json_list(all_datasets_dict["train_event_mentions_df"])
        train_entity_mentions = mentions_df_to_json_list(all_datasets_dict["train_entity_mentions_df"])

        dev_event_mentions = mentions_df_to_json_list(all_datasets_dict["dev_event_mentions_df"])
        dev_entity_mentions = mentions_df_to_json_list(all_datasets_dict["dev_entity_mentions_df"])

        test_event_mentions = mentions_df_to_json_list(all_datasets_dict["test_event_mentions_df"])
        test_entity_mentions = mentions_df_to_json_list(all_datasets_dict["test_entity_mentions_df"])
        

    """ 
    
    
    if is_custom_dataset == False:  # If the user does not provide its own conll and mentions files
        LOGGER.debug("Continue with this experiment config: ")
        LOGGER.debug(json.dumps(experiment_config, indent=4))

        # ---> Splitting Finished
        total_train_conll_df[DATASET_TOPIC_ID] = total_train_conll_df[DATASET] + "/" + total_train_conll_df[TOPIC_ID]
        total_test_conll_df[DATASET_TOPIC_ID] = total_test_conll_df[DATASET] + "/" + total_test_conll_df[TOPIC_ID]
        total_dev_conll_df[DATASET_TOPIC_ID] = total_dev_conll_df[DATASET] + "/" + total_dev_conll_df[TOPIC_ID]
        LOGGER.info("Finished splitting. Distribution:")
        LOGGER.info(f"Train: {total_train_conll_df[DATASET_TOPIC_ID].unique()}")
        LOGGER.info(f"Test: {total_test_conll_df[DATASET_TOPIC_ID].unique()}")
        LOGGER.info(f"Dev: {total_dev_conll_df[DATASET_TOPIC_ID].unique()}")

        # --> Filter singletons per split as stated in experiments config
        singletons_config = experiment_config["singletons"]
        total_train_mentions_df, total_test_mentions_df, total_dev_mentions_df, \
        total_train_conll_df, total_test_conll_df, total_dev_conll_df \
            = remove_singletons(singletons_config, total_train_mentions_df, total_test_mentions_df, total_dev_mentions_df,
                                total_train_conll_df, total_test_conll_df, total_dev_conll_df)
        

        # Separate the mentions (train/dev/test) into event and entity
        # total_train_mentions_df: contains all the mentions after removing the singletons
        # train_event_mentions_df: contains the train event mentions which are not singletons
        # The same is done for other mentions categories ...
        
        total_train_mentions_id = total_train_mentions_df.loc[:,"mention_id"].unique()
        total_dev_mentions_id = total_dev_mentions_df.loc[:,"mention_id"].unique()
        total_test_mentions_id = total_test_mentions_df.loc[:,"mention_id"].unique()
        
        train_event_mentions_df = train_event_mentions_df_w_singletons.loc[train_event_mentions_df_w_singletons["mention_id"].isin(total_train_mentions_id)]
        train_entity_mentions_df = train_entity_mentions_df_w_singletons.loc[train_entity_mentions_df_w_singletons["mention_id"].isin(total_train_mentions_id)]

        dev_event_mentions_df = dev_event_mentions_df_w_singletons.loc[dev_event_mentions_df_w_singletons["mention_id"].isin(total_dev_mentions_id)]
        dev_entity_mentions_df = dev_entity_mentions_df_w_singletons.loc[dev_entity_mentions_df_w_singletons["mention_id"].isin(total_dev_mentions_id)]

        test_event_mentions_df = test_event_mentions_df_w_singletons.loc[test_event_mentions_df_w_singletons["mention_id"].isin(total_test_mentions_id)]
        test_entity_mentions_df = test_entity_mentions_df_w_singletons.loc[test_entity_mentions_df_w_singletons["mention_id"].isin(total_test_mentions_id)]

    
        # convert mentions df (train/dev/test) to list of JSON mentions

        train_event_mentions = mentions_df_to_json_list(train_event_mentions_df)
        train_entity_mentions = mentions_df_to_json_list(train_entity_mentions_df)

        dev_event_mentions = mentions_df_to_json_list(dev_event_mentions_df)
        dev_entity_mentions = mentions_df_to_json_list(dev_entity_mentions_df)

        test_event_mentions = mentions_df_to_json_list(test_event_mentions_df)
        test_entity_mentions = mentions_df_to_json_list(test_entity_mentions_df)


        # convert conll df (train, dev, test) into conll string

        train_conll = conll_df_to_conll_str(total_train_conll_df)
        dev_conll = conll_df_to_conll_str(total_dev_conll_df)
        test_conll = conll_df_to_conll_str(total_test_conll_df)
    
    experiment_config = ""
    model_name = "CDCoref_M4"
    # train and test the model using the API
    train_model_dir = train_model(experiment_config, train_conll, train_event_mentions, train_entity_mentions , dev_conll, dev_event_mentions, dev_entity_mentions, model_name)
    print("Finished the training")

    ## Note: THe test data (conll and the mentions files are the same as the dev one): For illustration purposes
    #train_model_dir = "resources/eecdcr_models/self_trained/CDCoref_custom_dataset_T1"    # string
    document_clustering = False   # default value: False
    test_model(experiment_config, test_conll, test_entity_mentions, test_event_mentions, train_model_dir, document_clustering)


    