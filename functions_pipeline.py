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

pd.set_option('display.max_columns', None)

WORKING_DIR = os.path.join(os.getcwd())
DOWNLOAD_DIR = os.path.join(WORKING_DIR, DOWNLOAD_DIR_NAME)


### --------------

from shared.CONSTANTS import paths_own_files
from functions import conll_df_to_conll_str,  mentions_df_to_json_list



### --------------

def read_conll(dataset_name):

    '''
    Read a tokenized conll file.
    :param dataset_name: the name of the dataset
    :return: dataframe containing the conll data
    '''
    LOGGER.info("Reading conll file...")
    conll_df = pd.DataFrame()

    with open(os.path.join(os.path.join(DOWNLOAD_DIR), f'{dataset_name}.conll'), encoding="utf-8") as f:
        conll_str = f.read()
        conll_lines = conll_str.split("\n")
        for i, conll_line in tqdm(enumerate(conll_lines), total=len(conll_lines)):
            if i + 1 == len(conll_lines):
                break
            if "#begin document" in conll_line or "#end document" in conll_line or len(conll_line) <= 1:
                continue

            try:
                topic_subtopic = conll_line.split("\t")[0]
                sent_id = int(conll_line.split("\t")[1])
                token_id = int(conll_line.split("\t")[2])
                doc_id = conll_line.split("\t")[0].split("/")[-1]
                token_str = conll_line.split("\t")[3]
                reference = conll_line.split("\t")[4]
            except IndexError:
                LOGGER.debug(conll_line)
                LOGGER.debug(conll_line.split("\t"))

            conll_df = pd.concat([conll_df, pd.DataFrame({
                DATASET: dataset_name,
                CONLL_DOC_KEY: topic_subtopic,
                TOPIC_SUBTOPIC: topic_subtopic.split("/")[0] + "/" + topic_subtopic.split("/")[1],
                TOPIC_ID: topic_subtopic.split("/")[0],
                SUBTOPIC: topic_subtopic.split("/")[1],
                DOC_ID: doc_id,
                SENT_ID: sent_id,
                TOKEN_ID: token_id,
                TOKEN: token_str,
                REFERENCE: reference
            }, index=[0])])

            # For Testing
            if i > 8000:
               break

    return conll_df



def read_mentions(dataset_name, filename):
    '''
    Read a mentions file.
    :param dataset_name: the name of the dataset
    :param filename: the filename for the specific mentions file
    :return: a dataframe containing the mentions data
    '''
    LOGGER.info("Reading mentions file (" + filename + ")...")
    mentions_df = pd.read_json(os.path.join(DOWNLOAD_DIR, filename))
    mentions_df.reset_index(drop=True, inplace=True)
    mentions_df[DATASET] = dataset_name
    return mentions_df


def split_dataset(dataset, conll_df, mentions_df_list, autdef):
    '''
    Splitting a dataset. The dataset config includes information about which topics/docs to include in what split (train/test/dev).
    :param dataset: the dataset config containing information about the topics that should be included in each split
    :param conll_df: the conll dataframe
    :param mentions_df_list: a list containing all dataframes from the mentions (e.g. events, entities)
    :param autdef: the variable to define whether to do topic-level or doc-level splitting
    :return: the conll_df and mentions_dfs split into train/test/dev
    '''

    if dataset["name"] == "MEANTIME_en":
        topic_id_col = conll_df["topic_id"].replace(['corpus_airbus', 'corpus_apple', 'corpus_gm'], ["1", "2", "3"])
        conll_df["topic_id"] = topic_id_col
    

    if autdef == "topic_level":
        split_topics = dataset["split_topics"][autdef]
        train_topics = split_topics["train"]
        test_topics = split_topics["test"]
        dev_topics = split_topics["dev"]


        LOGGER.debug("Splitting conll data...")
        train_conll_df = conll_df[conll_df[TOPIC_ID].isin(train_topics)]
        test_conll_df = conll_df[conll_df[TOPIC_ID].isin(test_topics)]
        dev_conll_df = conll_df[conll_df[TOPIC_ID].isin(dev_topics)]

        train_mentions_df_list, test_mentions_df_list, dev_mentions_df_list = [], [], []
        for i, mentions_df in enumerate(mentions_df_list):
            LOGGER.debug(f"Splitting mentions file {i} of {len(mentions_df_list) - 1}...")
            train_mentions_df_list.append(mentions_df[mentions_df[TOPIC_ID].astype(str).isin(train_topics)])
            test_mentions_df_list.append(mentions_df[mentions_df[TOPIC_ID].astype(str).isin(test_topics)])
            dev_mentions_df_list.append(mentions_df[mentions_df[TOPIC_ID].astype(str).isin(dev_topics)])

    elif autdef == "doc_level":
        split_docs = dataset["split_topics"][autdef]
        train_docs = split_docs["train"]
        test_docs = split_docs["test"]
        dev_docs = split_docs["dev"]

        LOGGER.debug("Splitting conll data...")
        train_conll_df = conll_df[conll_df[DOC_ID].isin(train_docs)]
        test_conll_df = conll_df[conll_df[DOC_ID].isin(test_docs)]
        dev_conll_df = conll_df[conll_df[DOC_ID].isin(dev_docs)]

        train_mentions_df_list, test_mentions_df_list, dev_mentions_df_list = [], [], []
        for i, mentions_df in enumerate(mentions_df_list):
            LOGGER.debug(f"Splitting mentions file {i} of {len(mentions_df_list) - 1}...")
            # LOGGER.debug(mentions_df[CONLL_DOC_KEY].str.split(pat="/").str[-1])
            train_mentions_df_list.append(
                mentions_df[mentions_df[CONLL_DOC_KEY].str.split(pat="/").str[-1].isin(train_docs)])
            test_mentions_df_list.append(
                mentions_df[mentions_df[CONLL_DOC_KEY].str.split(pat="/").str[-1].isin(test_docs)])
            dev_mentions_df_list.append(
                mentions_df[mentions_df[CONLL_DOC_KEY].str.split(pat="/").str[-1].isin(dev_docs)])

    return train_conll_df, test_conll_df, dev_conll_df, \
           train_mentions_df_list, test_mentions_df_list, dev_mentions_df_list


def randomized_topic_splitting(dataset, split_ratio, conll_df):
    '''
    Splitting a datasets topics by taking the specified ratio into account.
    :param dataset: the dataset config
    :param split_ratio: the ratio to split data with as dict
    :param conll_df: the dataframe containing the conll data
    :return: the dict that specifies which topic belongs to which split
    '''
    topics = conll_df[TOPIC_ID].unique()
    if len(topics) > 10:
        LOGGER.info(f"Splitting {len(topics)} topics with the following default ratio: {str(split_ratio)}")
        topics_train, topics_test, topics_dev = torch.utils.data.random_split(topics, [
            split_ratio["train"],
            split_ratio["test"],
            split_ratio["dev"]
        ], generator=torch.Generator().manual_seed(RAND_SEED))
        topics_train, topics_test, topics_dev = list(topics_train), list(topics_test), list(topics_dev)
    else:
        LOGGER.info(f"Found 10 or less topics ({len(topics)}). Default splitting does not work for that amounts.")
        LOGGER.info(
            f"The dataset will only have a test split. You may restart the program and choose custom splitting.")
        topics_train, topics_test, topics_dev = [], list(conll_df[TOPIC_ID].unique()), []

    LOGGER.debug("Train topics: " + str(topics_train))
    LOGGER.debug("Test topics: " + str(topics_test))
    LOGGER.debug("Dev topics: " + str(topics_dev))

    split_topics = {"topic_level": {
        "train": list(map(str, topics_train)),
        "test": list(map(str, topics_test)),
        "dev": list(map(str, topics_dev))
    }
    }

    return split_topics


def randomized_document_splitting(dataset, split_ratio, conll_df):
    '''
    Splitting a datasets documents by taking the specified ratio into account.
    :param dataset: the dataset config
    :param split_ratio: the ratio to split data with as dict
    :param conll_df: the dataframe containing the conll data
    :return: the dict that specifies which docs belong to which split
    '''
    docs = conll_df[DOC_ID].unique()
    if len(docs) > 50:
        LOGGER.info(f"Splitting {len(docs)} topics with the following default ratio: {str(split_ratio)}")
        docs_train, docs_test, docs_dev = torch.utils.data.random_split(docs, [
            split_ratio["train"],
            split_ratio["test"],
            split_ratio["dev"]
        ], generator=torch.Generator().manual_seed(RAND_SEED))
        docs_train, docs_test, docs_dev = list(docs_train), list(docs_test), list(docs_dev)
    else:
        LOGGER.info(f"Found 50 or less documents ({len(docs)}). Default splitting does not work for that amounts.")
        LOGGER.info(
            f"The dataset will only have a train split. You may restart the program and choose custom splitting.")
        docs_train, docs_test, docs_dev = [], list(conll_df[TOPIC_ID].unique()), []

    LOGGER.debug("Train topics: " + str(docs_train))
    LOGGER.debug("Test topics: " + str(docs_test))
    LOGGER.debug("Dev topics: " + str(docs_dev))

    split_topics = {"doc_level": {
        "train": list(map(str, docs_train)),
        "test": list(map(str, docs_test)),
        "dev": list(map(str, docs_dev))
    }
    }

    return split_topics


def import_dataset():

    '''
    Read the custom train, dev, test conll, event and entity mentions file
    :return: a list of 3 file contents (conll, entity and event mentions files) for each split: In total the content of 9 files as indicated in paths_own_files
    '''

    data = {}

    i = 0
    for file_type in paths_own_files:
        with open(paths_own_files[file_type], "r", encoding="utf8") as f:
            if i %3 ==0:
                data[file_type] = f.read()
            else:
                data[file_type] = json.load(f)
            i +=1

    file_types = list(data.keys())
    return [data[file_types[i]] for i in range(len(file_types))]


    
def choose_dataset(config):
    '''
    Choose a dataset from the datasets_config.json file.
    :return: dataset config, corresponding index in the config file
    '''
    while True:
        try:
            LOGGER.info(
                "You chose to select a dataset for the pipeline. Please select one from this list to download:")
            prompt_str = "\n"
            # make list of available datasets:
            for i, dataset in enumerate(config["datasets"]):
                prompt_str = prompt_str + str(i) + ": " + dataset["name"] + "\n"
            LOGGER.info(prompt_str)
            a = input()
            assert 0 <= int(a) < len(config["datasets"])
            break
        except (ValueError, AssertionError) as e:
            LOGGER.info("Oops! That input was not correct. Please retry by typing a number from the list.")

    a = int(a)
    dataset = config["datasets"][a]

    download_dataset(dataset)

    return dataset, a


def download_dataset(dataset):
    '''
    Downloads a dataset.
    :param dataset: The provided dataset dict from datasets_config
    '''
    LOGGER.info("Downloading this dataset: " + dataset["name"])
    LOGGER.info("Downloading the dataset and saving to: " + str(DOWNLOAD_DIR))
    LOGGER.info("Downloading conll file...")
    r = requests.get(dataset["conll_url"])
    filename = dataset["name"] + ".conll"
    with open(os.path.join(DOWNLOAD_DIR, filename), 'wb') as f:
        f.write(r.content)
    for i, file_link in enumerate(dataset["mentions_url"]):
        LOGGER.info("Downloading mentions file " + str(i + 1) + " of " + str(
            len(dataset["mentions_url"])) + "...\n(Link: " + file_link + ")")
        r = requests.get(file_link)
        filename = file_link.split("/")[-1]
        with open(os.path.join(DOWNLOAD_DIR, filename), 'wb') as f:
            f.write(r.content)


def split_menu(dataset, conll_df, mentions_df_list, dataset_config_index, config):
    '''
    A user menu to handle dataset splitting.
    :param dataset: the dataset config
    :param conll_df: dataset conll data
    :param mentions_df_list: dataset mentions dataframes in a list
    :param dataset_config_index: the index of the dataset in the config file
    :return: dataset config, train/test/dev splits for conll & mentions files
    '''
    # Splitting dataset
    autdef = None
    if "doc_level" in dataset["split_topics"].keys():
        LOGGER.info("Splitting will be done on document level (automatic)...")
        autdef = "doc_level"

        if dataset["split_topics"][autdef] is None:
            # do default splitting
            LOGGER.info("No splitting was provided for " + dataset["name"] + " in the config.")

            # ----> INPUT RATIO
            while True:
                try:
                    LOGGER.info(f"Do you want to use the default split ratio for documents? (y/n)")
                    a = input().lower()
                    assert a == "y" or a == "n"
                    break
                except (ValueError, AssertionError) as e:
                    LOGGER.info("Oops! That input was not correct (y/n). Please retry.")
            if a == "y":
                split_ratio = DEFAULT_SPLIT
            elif a == "n":
                while True:
                    try:
                        LOGGER.info("Please choose a train/test/dev split. Input format example:  0.8,0.1,0.1")
                        b = input().split(",")
                        b = list(map(float, b))
                        LOGGER.info(math.fsum(b))
                        assert math.fsum(b) == 1.0 and 0.0 <= b[0] and 0.0 <= b[1] and 0.0 <= b[2]
                        break
                    except (ValueError, AssertionError) as e:
                        LOGGER.info(b)
                        LOGGER.warning(
                            "Oops! That input was not correct. Make sure all values sum up to 1. Please retry.")
                split_ratio = {
                    "train": b[0],
                    "test": b[1],
                    "dev": b[2]
                }

            LOGGER.info("Proceeding with the following topic split ratio: ")
            LOGGER.info(split_ratio)

            dataset["split_topics"] = randomized_document_splitting(dataset, split_ratio, conll_df)

            # save updated config
            config["datasets"][dataset_config_index]["split_topics"] = dataset["split_topics"]
            with open(DATASET_CONFIG_FILE.replace(".json", "_updated.json"), 'w') as f:
                json.dump(dict(config), f)
            LOGGER.info("Saved the updated config to: " + str(
                os.path.join(os.getcwd(), DATASET_CONFIG_FILE.replace(".json", "_updated.json"))))

    elif "topic_level" in dataset["split_topics"].keys():
        LOGGER.info("Splitting will be done on topic level (default)...")
        autdef = "topic_level"

        if dataset["split_topics"][autdef] is None:
            # do default splitting
            LOGGER.info("No splitting was provided for " + dataset["name"] + " in the config.")

            # ----> INPUT RATIO
            while True:
                try:
                    LOGGER.info(f"Do you want to use the default split ratio for topics? (y/n)")
                    a = input().lower()
                    assert a == "y" or a == "n"
                    break
                except (ValueError, AssertionError) as e:
                    LOGGER.info("Oops! That input was not correct (y/n). Please retry.")
            if a == "y":
                split_ratio = DEFAULT_SPLIT
            elif a == "n":
                while True:
                    try:
                        LOGGER.info("Please choose a train/test/dev split. Input format example:  0.8,0.1,0.1")
                        b = input().split(",")
                        b = list(map(float, b))
                        LOGGER.info(math.fsum(b))
                        assert math.fsum(b) == 1.0 and 0.0 <= b[0] and 0.0 <= b[1] and 0.0 <= b[2]
                        break
                    except (ValueError, AssertionError) as e:
                        LOGGER.warning(
                            "Oops! That input was not correct. Make sure all values sum up to 1. Please retry.")
                split_ratio = {
                    "train": b[0],
                    "test": b[1],
                    "dev": b[2]
                }

            LOGGER.info("Proceeding with the following topic split ratio: ")
            LOGGER.info(split_ratio)

            dataset["split_topics"] = randomized_topic_splitting(dataset, split_ratio, conll_df)

            # save updated config
            config["datasets"][dataset_config_index]["split_topics"] = dataset["split_topics"]
            with open(DATASET_CONFIG_FILE.replace(".json", "_updated.json"), 'w') as f:
                json.dump(dict(config), f)
            LOGGER.info("Saved the updated config to: " + str(
                os.path.join(os.getcwd(), DATASET_CONFIG_FILE.replace(".json", "_updated.json"))))

    train_conll_df, test_conll_df, dev_conll_df, train_mentions_df_list, test_mentions_df_list, dev_mentions_df_list = split_dataset(
        dataset, conll_df, mentions_df_list, autdef)

    return dataset, \
           train_conll_df, test_conll_df, dev_conll_df, \
           train_mentions_df_list, test_mentions_df_list, dev_mentions_df_list


def remove_singletons(singletons_config, total_train_mentions_df, total_test_mentions_df, total_dev_mentions_df,
                      total_train_conll_df, total_test_conll_df, total_dev_conll_df):
    '''
    Removes singletons (coreferences that were only mentioned once) from the data.
    Removes mention entries and updates the conll to not include singletons anymore.
    :param singletons_config: The config dict that states what splits to remove singletons from
    :param total_train_mentions_df: the train mentions
    :param total_test_mentions_df: the test mentions
    :param total_dev_mentions_df: the dev mentions
    :param total_train_conll_df: the train conll data
    :param total_test_conll_df: the test conll data
    :param total_dev_conll_df: the dev conll data
    :return: the updated data
    '''

    # Remove singletons (references only mentioned once) in mentions_df and conll_df
    LOGGER.info(f"Initial amount of mentions in the train split: {str(total_train_mentions_df.shape[0])}")
    if singletons_config["train"]:
        mentions_df_unique = total_train_mentions_df.drop_duplicates([COREF_CHAIN, DOC_ID, SENT_ID, MENTION_HEAD_ID],
                                                                     keep="first")
        chain_df = mentions_df_unique[[DOC_ID, COREF_CHAIN]].groupby(COREF_CHAIN).count()
        singleton_chains = chain_df[(chain_df[DOC_ID] == 1)].index.tolist()
        total_train_mentions_df = total_train_mentions_df[
            ~(total_train_mentions_df[COREF_CHAIN].isin(singleton_chains))]
        # update conll:
        total_train_conll_df = total_train_conll_df.replace(to_replace='\(([^\)]+)\)\s\|', value='',
                                                            regex=True)  # match "(coref_id) |"
        total_train_conll_df = total_train_conll_df.replace(to_replace='\(([^\)]+)\)', value='',
                                                            regex=True)  # match "(coref_id)"
        total_train_conll_df.loc[total_train_conll_df[REFERENCE] == "", REFERENCE] = "-"
        LOGGER.info(
            f"Removed singletons for the train split. New amount of mentions: {str(total_train_mentions_df.shape[0])}")
    LOGGER.info(f"Initial amount of mentions in the test split: {str(total_test_mentions_df.shape[0])}")
    if singletons_config["test"]:
        mentions_df_unique = total_test_mentions_df.drop_duplicates([COREF_CHAIN, DOC_ID, SENT_ID, MENTION_HEAD_ID],
                                                                    keep="first")
        chain_df = mentions_df_unique[[DOC_ID, COREF_CHAIN]].groupby(COREF_CHAIN).count()
        singleton_chains = chain_df[(chain_df[DOC_ID] == 1)].index.tolist()
        total_test_mentions_df = total_test_mentions_df[
            ~(total_test_mentions_df[COREF_CHAIN].isin(singleton_chains))]
        # update conll:
        total_test_conll_df = total_test_conll_df.replace(to_replace='\(([^\)]+)\)\s\|', value='',
                                                          regex=True)  # match "(coref_id) |"
        total_test_conll_df = total_test_conll_df.replace(to_replace='\(([^\)]+)\)', value='',
                                                          regex=True)  # match "(coref_id)"
        total_test_conll_df.loc[total_test_conll_df[REFERENCE] == "", REFERENCE] = "-"
        LOGGER.info(
            f"Removed singletons for the test split. New amount of mentions: {str(total_test_mentions_df.shape[0])}")
    LOGGER.info(f"Initial amount of mentions in the dev split: {str(total_dev_mentions_df.shape[0])}")
    if singletons_config["dev"]:
        mentions_df_unique = total_dev_mentions_df.drop_duplicates([COREF_CHAIN, DOC_ID, SENT_ID, MENTION_HEAD_ID],
                                                                   keep="first")
        chain_df = mentions_df_unique[[DOC_ID, COREF_CHAIN]].groupby(COREF_CHAIN).count()
        singleton_chains = chain_df[(chain_df[DOC_ID] == 1)].index.tolist()
        total_dev_mentions_df = total_dev_mentions_df[
            ~(total_dev_mentions_df[COREF_CHAIN].isin(singleton_chains))]
        # update conll:
        total_dev_conll_df = total_dev_conll_df.replace(to_replace='\(([^\)]+)\)\s\|', value='',
                                                        regex=True)  # match "(coref_id) |"
        total_dev_conll_df = total_dev_conll_df.replace(to_replace='\(([^\)]+)\)', value='',
                                                        regex=True)  # match "(coref_id)"
        total_dev_conll_df.loc[total_dev_conll_df[REFERENCE] == "", REFERENCE] = "-"
        LOGGER.info(
            f"Removed singletons for the dev split. New amount of mentions: {str(total_dev_mentions_df.shape[0])}")

    return total_train_mentions_df, total_test_mentions_df, total_dev_mentions_df, total_train_conll_df, total_test_conll_df, total_dev_conll_df