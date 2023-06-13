import json
import os
import pickle
from typing import Tuple, List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer

import config
from util import logutil

logger = logutil.logger_run


def save_tfidf_converter(scope_name: str,
                         tfidf_converter: TfidfVectorizer) -> None:
    fp = os.path.join(config.MODEL_DIR, f"tf_idf_{scope_name}")
    pickle.dump(tfidf_converter, open(fp, 'wb'))
    logger.info(f"TfidfVectorizer [{scope_name}] saved at {fp}")


def load_tfidf_converter(scope_name: str) -> TfidfVectorizer:
    fp = os.path.join(config.MODEL_DIR, f"tf_idf_{scope_name}")
    tfidf_converter = pickle.load(open(os.path.join(config.MODEL_DIR, f"tf_idf_{scope_name}"), 'rb'))
    logger.info(f"TfidfVectorizer [{scope_name}] loaded from {fp}")
    return tfidf_converter


def save_result_file(results: Tuple[str, float]) -> None:
    result_str = '; '.join([f'"{scope_name}": {overall_acc:.2%}' for scope_name, overall_acc in results]) + '\n'

    with open(config.RESULT_FILE, 'a') as f:
        f.write(result_str)


def save_model_info(model_info: List[Tuple]) -> None:
    info = {
        scope_name: {'model_module': model_module, 'model_name': model_name, 'types': types}
        for scope_name, types, model_module, model_name in model_info
    }

    with open(config.MODEL_INFO_FILE, 'w') as f:
        json.dump(info, f)


def load_model_info() -> Dict[str, Dict]:
    with open(config.MODEL_INFO_FILE, 'r') as f:
        model_info = json.load(f)
    return model_info
