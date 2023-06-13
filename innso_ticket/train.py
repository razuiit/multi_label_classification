import random
from typing import List, Tuple

import numpy as np
import pandas as pd

import config
from model.base import BaseModel
from model.bert_pretrained import BERT
from model.randomforest import RandomForest
from model.unsupervised.unsupervised import UnsupervisedModel
from innso_ticket.model.bert_hq import BERT_HQ
from model.voting import Voting
from preprocessing.embedding import Embedder
from preprocessing.preprocess import load_data
from unit.data import Data, split_data
from util import logutil
from util.io import save_result_file, save_model_info

logger = logutil.logger_run

random.seed(config.SEED)
np.random.seed(config.SEED)

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


def __train_scope(scope_name: str,
                  train_data: Data,
                  test_data: Data,
                  full_types: List[str]) -> Tuple[BaseModel, float, pd.DataFrame]:
    model_classes = [
        # RandomForest,
        # Voting,
        # BERT,
        BERT_HQ,
        # UnsupervisedModel,
    ]

    best_acc, best_model, results = 0., None, []

    for model_class in model_classes:
        logger.info(f"{model_class.__name__} is training...")
        model = model_class(scope_name, full_types)
        overall_acc, accuracies = model.train(train_data, test_data)
        if overall_acc > best_acc:
            best_acc = overall_acc
            best_model = model
        results.append([scope_name, model_class.__name__, overall_acc] + accuracies)

    best_model.save_model()
    logger.info(f"best model for {scope_name} is {best_model.__class__.__name__}")

    result_df = pd.DataFrame(results,
                             columns=['scope_name', 'model', 'overall_acc'] + [f"{t}_acc" for t in config.TYPE_COLS])
    return best_model, best_acc, result_df


def train_model(type1_list: List[str],
                preprocessed: bool) -> None:
    data = load_data(type1_list, preprocessed)

    model_info, overall_results, result_dfs = [], [], []
    for scope_name, sdf, classes in data:
        logger.info(f"Business Scope:    {scope_name}")
        embedder = Embedder(scope_name)
        embedding = embedder.get_embedding(sdf, fit=True)
        train_data, test_data, full_types = split_data(embedding, sdf)
        if train_data is None:
            continue
        best_model, best_acc, result_df = __train_scope(scope_name, train_data, test_data, full_types)
        model_info.append((scope_name, full_types, best_model.__class__.__module__, best_model.__class__.__name__))
        overall_results.append((scope_name, best_acc))
        result_dfs.append(result_df)

    save_model_info(model_info)
    save_result_file(overall_results)

    result_df = pd.concat(result_dfs)
    result_df.to_csv(config.FULL_RESULT_FILE, index=False)
    logger.info(result_df)
