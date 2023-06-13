import os
import shutil
from typing import Dict, List

import pandas as pd

import config


def str_type_cols(df: pd.DataFrame,
                  cols: List[str]) -> pd.DataFrame:
    return df.astype({col: str for col in cols})


def filter_empty_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df[config.CURRENT_TYPE] != '') & (~df[config.CURRENT_TYPE].isna()),]


def concat_types(types) -> str:
    return config.JOIN_CHAR.join(types)


def format_long_types(types: pd.Series) -> pd.Series:
    types = types.str.strip().str.lower().fillna(config.EMPTY_TYPE)
    return pd.Series([concat_types(types[:i + 1]) for i in range(len(types))])


def parse_full_type(full_type: str) -> pd.Series:
    return format_long_types(pd.Series(full_type.split(config.JOIN_CHAR)))


def parse_full_type_to_short_types(full_type: str) -> pd.Series:
    return pd.Series(full_type.split(config.JOIN_CHAR)).str.strip().str.lower().fillna(config.EMPTY_TYPE)


def df_to_json(df: pd.DataFrame) -> Dict:
    """
    Convert DF into JSON by row
    """
    return df.apply(lambda x: x.to_dict(), axis=1).to_list()


def remove_file_or_dir(fp: str) -> None:
    if os.path.isfile(fp):
        os.remove(fp)
    elif os.path.isdir(fp):
        shutil.rmtree(fp)


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        print(f"Create a new directory {dir_path}")
        os.makedirs(dir_path)


def model_from_pretrained(model_class, model_name, **args):
    model_dir = os.path.join(config.PRETRAINED_MODEL_DIR, model_name)
    try:
        model = model_class.from_pretrained(model_dir, **args)
        print(f"`{model_class.__name__}` model `{model_name}` is loaded locally.")
    except:
        mkdir(model_dir)

        model = model_class.from_pretrained(model_name, **args)
        model.save_pretrained(model_dir)
        print(f"`{model_class.__name__}` model `{model_name}` is loaded online.")

    return model


def get_model_fp(scope_name: str) -> str:
    return os.path.join(config.MODEL_DIR, f"model_{scope_name}")
