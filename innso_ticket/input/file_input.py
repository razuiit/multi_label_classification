import os

import pandas as pd

import config
from util import logutil, utils

logger = logutil.logger_run


def get_data() -> pd.DataFrame:
    df1 = pd.read_csv(os.path.join(config.DATA_DIR, 'data/AppGallery_done.csv'), skipinitialspace=True)
    df1 = df1[config.INPUT_COLS]

    df2 = pd.read_csv(os.path.join(config.DATA_DIR, 'data/Cloud_done.csv'), skipinitialspace=True)
    df2 = df2[config.INPUT_COLS]

    # df3 = pd.read_csv('../filestore/data/Huawei ID_done.csv')

    df4 = pd.read_csv(os.path.join(config.DATA_DIR, 'data/IAP_done.csv'), skipinitialspace=True)
    df4 = df4[config.INPUT_COLS]

    df5 = pd.read_csv(os.path.join(config.DATA_DIR, 'data/octdatade_22Nov.csv'), skipinitialspace=True)
    df5.rename(columns={'Ticket': 'Ticket id',
                        'Interaction': 'Interaction id',
                        'Typology 1': 'Type 1',
                        'Typology 2': 'Type 2',
                        'Typology 3': 'Type 3',
                        'Typology 4': 'Type 4',
                        config.CONTENT_COL: 'non_english',
                        'English Translation': config.CONTENT_COL}, inplace=True)
    df5[config.SUMMARY_COL] = ''
    df5 = df5[config.INPUT_COLS]

    df6 = pd.read_csv(os.path.join(config.DATA_DIR, 'data/task_IAP_10_11Nov_done.csv'), skipinitialspace=True)
    df6.rename(columns={'originId': 'Ticket id',
                        'id': 'Interaction id',
                        'type1': 'Type 1',
                        'type2': 'Type 2',
                        'type3': 'Type 3',
                        'type4': 'Type 4',
                        'content_en': config.CONTENT_COL,
                        'processing_en': config.SUMMARY_COL}, inplace=True)
    df6 = df6[config.INPUT_COLS]

    df7 = pd.read_csv(os.path.join(config.DATA_DIR, 'data/task_map_01072022_done.csv'), skipinitialspace=True)
    df7.rename(columns={'originId': 'Ticket id',
                        'id': 'Interaction id',
                        'type1': 'Type 1',
                        'type2': 'Type 2',
                        'type3': 'Type 3',
                        'type4': 'Type 4',
                        'content_en': config.CONTENT_COL,
                        'processing_en': config.SUMMARY_COL}, inplace=True)
    df7 = df7[config.INPUT_COLS]

    df8 = pd.read_csv(os.path.join(config.DATA_DIR, 'data/task_cloud_22062022_done.csv'), skipinitialspace=True)
    df8.rename(columns={'originId': 'Ticket id',
                        'id': 'Interaction id',
                        'Level-1 Classification (Business)': 'Type 1',
                        'Level-2 Classification': 'Type 2',
                        'Level 3 Classification': 'Type 3',
                        'Level-4 Classification': 'Type 4',
                        'English translation of content': config.CONTENT_COL,
                        'Processing process English translation': config.SUMMARY_COL}, inplace=True)
    df8 = df8[config.INPUT_COLS]

    df = pd.concat([df1, df2, df4, df5, df6, df7, df8])

    df = utils.str_type_cols(df, [config.CONTENT_COL, config.SUMMARY_COL])

    return df
