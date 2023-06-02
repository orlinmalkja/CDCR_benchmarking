import os

from EECDCR.all_models.eval_utils import write_span_based_cd_coref_clusters, write_mention_based_cd_clusters, \
    write_mention_based_wd_clusters
from shared.CONSTANTS import CONFIG
from shared.classes import Corpus


def create_gold_files_for_corpus(test_data: Corpus):
    if not os.path.exists(CONFIG['gold_files_dir'].format(CONFIG['dataset_name'])):
        os.makedirs(CONFIG['gold_files_dir'].format(CONFIG['dataset_name']))

    if not CONFIG['mention_based_key_file']:                          # Its value in the config dict is true
        print('Creating span-based event mentions key file')
        out_file = os.path.join(CONFIG['gold_files_dir'].format(CONFIG['dataset_name']),
                                'CD_test_event_span_based.key_conll')
        write_span_based_cd_coref_clusters(test_data, out_file, is_event=True, is_gold=True, use_gold_mentions=True)
        print('Creating span-based entity mentions key file')
        out_file = os.path.join(CONFIG['gold_files_dir'].format(CONFIG['dataset_name']),
                                'CD_test_entity_span_based.key_conll')
        write_span_based_cd_coref_clusters(test_data, out_file, is_event=False, is_gold=True, use_gold_mentions=True)
    else:
        print('Creating mention-based event mentions key file')
        out_file = os.path.join(CONFIG['gold_files_dir'].format(CONFIG['dataset_name']),
                                'CD_test_event_mention_based.key_conll')
        write_mention_based_cd_clusters(test_data, is_event=True, is_gold=True, out_file=out_file)

        out_file = os.path.join(CONFIG['gold_files_dir'].format(CONFIG['dataset_name']),
                                'WD_test_event_mention_based.key_conll')
        write_mention_based_wd_clusters(test_data, is_event=True, is_gold=True, out_file=out_file)

        print('Creating mention-based entity mentions key file')
        out_file = os.path.join(CONFIG['gold_files_dir'].format(CONFIG['dataset_name']),
                                'CD_test_entity_mention_based.key_conll')
        write_mention_based_cd_clusters(test_data, is_event=False, is_gold=True, out_file=out_file)

        out_file = os.path.join(CONFIG['gold_files_dir'].format(CONFIG['dataset_name']),
                                'WD_test_entity_mention_based.key_conll')
        write_mention_based_wd_clusters(test_data, is_event=False, is_gold=True, out_file=out_file)


    print("exiting from function: create_gold_files_for_corpus")
