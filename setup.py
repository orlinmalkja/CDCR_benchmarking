# PARAMS
import os
import json

DOWNLOAD_DIR_NAME = "data"
CONLL_SUFFIX = ".conll"
ENTITIES_MENTIONS_SUFFIX = "entities_mentions.json"
EVENT_MENTIONS_SUFFIX = "event_mentions.json"
DATASET_CONFIG_FILE = "datasets_config.json"
EXPERIMENT_CONFIG_FILE = "experiment_config.json"
RAND_SEED = 42

DEFAULT_SPLIT = {
    "train": 0.8,
    "test":  0.1,
    "dev": 0.1
}

# DATASETS
NEWSWCL50 = "NewsWCL50"
ECB_PLUS = "ECBplus"
MEANTIME_EN = "MEANTIME_en"
NIDENT = "NiDENT"
NP4E = "NP4E"
GVC = "GVC"
FCC = "FCC"

# coref types
STRICT = "STRICT"
NEAR_IDENTITY = "NEAR_IDENTITY"

# doc.json fields (from news-please)
TITLE = "title"
DESCRIPTION = "description"
TEXT = "text"
SOURCE_DOMAIN = "source_domain"

# NewsWCL50 original column names in annotated mentions)
CODE = "Code"
SEGMENT = "Segment"
DOCUMENT_NAME = "Document name"
BEGINNING = "Beginning"
TYPE = "type"

# mentions.json fields
DATASET = "dataset"
TOPIC_ID = "topic_id"
DATASET_TOPIC_ID = "dataset_topic_id"
TOPIC = "topic"
COREF_CHAIN = "coref_chain"
MENTION_FULL_TYPE = "mention_full_type"
MENTION_TYPE = "mention_type"
MENTION_NER = "mention_ner"
MENTION_HEAD_POS = "mention_head_pos"
MENTION_HEAD_LEMMA = "mention_head_lemma"
MENTION_HEAD = "mention_head"
MENTION_HEAD_ID = "mention_head_id"
DOC_ID_FULL = "doc_id_full"
DOC_ID = "doc_id"
IS_CONTINIOUS = "is_continuous"
IS_SINGLETON = "is_singleton"
SENTENCE = "sentence"
MENTION_ID = "mention_id"
SCORE = "score"
SENT_ID = "sent_id"
MENTION_CONTEXT = "mention_context"
TOKENS_NUMBER = "tokens_number"
TOKENS_TEXT = "tokens_text"
TOKENS_STR = "tokens_str"
TOKEN_ID = "token_id"
COREF_TYPE = "coref_type"
SUBTOPIC = "subtopic"
CONLL_DOC_KEY = "conll_doc_key"

# conll fields
REFERENCE = "reference"
DOC_IDENTIFIER = "doc_identifier"
TOKEN = "token"
TOPIC_SUBTOPIC = "topic/subtopic_name"

# summary fields
DATASET_NAME = "dataset"
TOPICS = "topics"
ENTITY = "entity"
EVENT = "event"
MENTIONS = "mentions"
PHRASING_DIVERSITY = "phrasing_diversity"
UNIQUE_LEMMAS = "unique_lemmas"
WEIGHTED = "_weighted"
MEAN = "_mean"
ALL = "_all"
WO_SINGL = "_wo_singl"
ARTICLES = "articles"
TOKENS = "tokens"
SINGLETONS = "singletons"
AVERAGE_SIZE = "average_size"
