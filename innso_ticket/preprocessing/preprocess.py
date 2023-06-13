from typing import List, Tuple

import pandas as pd

import config
from input import file_input
from preprocessing import noise_cancellation, deduplication
from preprocessing.translation import Translator
from util import logutil, utils

logger = logutil.logger_run


def __clean_text(text: str) -> str:
    return text.strip().lower()


def __get_classes(df: pd.DataFrame) -> List[str]:
    classes = df[config.CLASS_COL].unique().tolist()
    logger.info(f"`{df[config.FORMATTED_TYPE_COLS[0]].iloc[0]}` has {len(classes)} classes: {classes}")
    return classes


def preprocess_text_data(df: pd.DataFrame) -> pd.DataFrame:
    df[config.SUMMARY_COL] = df[config.SUMMARY_COL].fillna('')

    # clean texts
    df[config.SUMMARY_COL] = df[config.SUMMARY_COL].astype(str).apply(__clean_text)
    df[config.CONTENT_COL] = df[config.CONTENT_COL].astype(str).apply(__clean_text)

    # translation
    df[config.SUMMARY_COL] = Translator().translate(df[config.SUMMARY_COL].tolist())
    df[config.CONTENT_COL] = Translator().translate(df[config.CONTENT_COL].tolist())

    # add `TEXT_COL` by joining `TICKET_SUMMARY` and `INTERACTION_CONTENT`
    df[config.TEXT_COL] = df[config.SUMMARY_COL] + ' ' + df[config.CONTENT_COL]

    # noise removal
    df = noise_cancellation.cancel_noise(df)

    return df


def __preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # remove rows with duplicated `TICKET_ID_COL` and `INTERACTION_ID_COL`
    df = df.groupby([config.TICKET_ID_COL, config.INTERACTION_ID_COL]).apply(lambda x: x.iloc[0]).reset_index(drop=True)

    df = preprocess_text_data(df)

    df = deduplication.deduplicate(df)

    # remove empty content data
    df = df[df[config.CONTENT_COL].str.len() > 0]

    # preprocess type columns
    for col in config.TYPE_COLS:
        df[col] = df[col].str.lower().str.strip()

    # add `formatted_` type columns
    df[config.FORMATTED_TYPE_COLS] = df[config.TYPE_COLS].apply(utils.format_long_types, axis=1)
    # add `CLASS_COL`
    df[config.CLASS_COL] = df[config.FORMATTED_TYPE_COLS[-1]].copy()

    # remove unlabelled data
    df = df.loc[(df[config.CLASS_COL] != '') & df[config.CLASS_COL].notna()]

    # remove small scopes
    scope_counts = df[config.TYPE_1_COL].value_counts()
    ignore_scopes = scope_counts[scope_counts < config.SCOPE_MIN_RECORDS]
    logger.info(f"ignore scopes: {ignore_scopes}")
    df = df[~df[config.TYPE_1_COL].isin(ignore_scopes.keys())]

    return df


def __load_preprocessed_df() -> pd.DataFrame:
    df = pd.read_csv(config.PREPROCESSED_FP)
    df = utils.str_type_cols(df, [config.CONTENT_COL, config.SUMMARY_COL])
    return df


def load_data(type1_list: List[str],
              preprocessed: bool) -> Tuple[str, pd.DataFrame, List[str]]:
    if preprocessed:
        # when reading from already preprocessed file
        df = __load_preprocessed_df()
        if type1_list:
            df = df[df[config.TYPE_1_COL].isin(type1_list)]
        logger.info('Preprocessed DF is loaded.')
    else:
        # when reading from multiple raw data files
        df = file_input.get_data()
        if type1_list:
            df = df[df[config.TYPE_1_COL].str.strip().str.lower().isin(type1_list)]
        df = __preprocess_data(df)
        df.to_csv(config.PREPROCESSED_FP)
        logger.info('Data is preprocessed and saved.')

    logger.info(f"dataset size: {df.shape}")

    data = [(scope, sdf, __get_classes(sdf)) for scope, sdf in df.groupby(config.FORMATTED_TYPE_COLS[0])]

    stats_df = pd.DataFrame(
        [[len(sdf)] + [sdf[col].nunique() for col in config.FORMATTED_TYPE_COLS] for _, sdf, _ in data],
        index=list(zip(*data))[0], columns=['num_texts'] + [f"{col}_class_count" for col in config.TYPE_COLS])
    logger.info(stats_df)

    return data
