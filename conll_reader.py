import json

from shared.CONSTANTS import dataset_path, meantimeNameConverter, CONFIG, train_test_path
from shared.classes import Token, Sentence, Document, Topic, Corpus
from typing import Union

"""
    To no real surprise the topic_id/document_id is not consistent between meantime & ecb+
    or between the conll format and the .json files to be more precise.
    So there was a need to change the topic / documents id in both cases differently to match
    the format of the .json file with all the mentions and I had to differentiate between the
    two datasets. If more data sets will be added in the future, this might look even worse.
"""


def read_CoNLL(topic_level=False, specific_document="", split: Union[None, str] = None) -> Corpus:

    number_of_seen_topics = 1
    dataset_name = CONFIG['dataset_name']

    # train_test_dict = json.load(open(train_test_path[dataset_name], 'r'))
    # mode = 'test' if CONFIG['test'] else 'train'
    # necessary_topics = train_test_dict[mode]

    with open(dataset_path[dataset_name], "r", encoding="utf8") as f:
        data = f.read()

    data = data.split("\n")
    

    if topic_level and specific_document != "" and split is None:
        data = [x for x in data if x.split("\t")[0].startswith(specific_document + "/")]
    # Question:
    #  - In the same ECBplus conll file, do we have the topics belonging to the train and test data ? Previously this was the idea and for the test set we just pick the topics belonging to it..

    print("split", split)
    #  Read the data of Test set - the topics belonging to the test set :  Done
    if split in ['test', 'train', 'dev']:
        train_test_dict = json.load(open(train_test_path[dataset_name], 'r'))
        necessary_topics = train_test_dict[split]

        # Filter the test data
        test_data = []
        for datum in data:          # datum represents each line in the data
            if datum.startswith("#begin") or datum.startswith("#end"):
                test_data.append(datum)
                continue
            if datum =="":
                continue

            split_line = datum.split("\t")

            if len(split_line) ==5:
                token_txt = split_line[3]

            elif len(split_line) == 3:
                token_txt="\n"

            else:
                continue

            topic_document = split_line[0]
            topic_name = topic_document.split("/")[0]
            sub_topic = topic_document.split("/")[1]
            topic_name = topic_name + "/" + sub_topic
            if topic_name in necessary_topics:
                test_data.append(datum)

        data = test_data      # Now data consists of the test_data

        #data = [datum for datum in data if datum.split("\t")[0].split("/")[0] in necessary_topics]  # we will have the data of the "split" in this case of test set (All data of different documents are merged)

    #print("Test data",data)
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

        print()

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

        # Reading the topic, document id, sentence and token
        topic_document = split_line[0]

        topic_name, sub_topic, doc_id = topic_document.split("/")
        topic_name = topic_name + "/" + sub_topic

        sentence_id = split_line[1]
        token_id = split_line[2]



        """  Orlin: Commented below lines for now, we may need to modify them later
        document_name = topic_and_document.split("/")[1]
        if dataset_name == "ECB+":
            document_name = topic_name.replace("ecb", f"_{document_name}ecb")
        if dataset_name == "MEANTime":
            topic_name = meantimeNameConverter[topic_name]
        """

        # if we start a new sentence add the old sentence to the document
        # Problem: When we have two different documents starting with same sentence id (document 7,8),
        # the tokens add to the same sentence -- Fixed
        # ----

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
