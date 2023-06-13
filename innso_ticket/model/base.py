import os
import pickle
from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import config
from unit.data import Data
from util import logutil
from util.utils import parse_full_type, remove_file_or_dir

logger = logutil.logger_run


class BaseModel(ABC):
    def __init__(self,
                 scope_name: str,
                 full_types: List[str]) -> None:
        self.scope_name = scope_name
        self.full_types = full_types
        self.model_fp = os.path.join(config.MODEL_DIR, f"model_{self.scope_name}")

    @abstractmethod
    def train(self,
              train_data: Data,
              test_data: Data) -> Tuple[float, List[float]]:
        ...

    @abstractmethod
    def predict(self) -> List[str]:
        ...

    def _calc_accuracies(self,
                         y_true: List[str],
                         y_pred: List[str]) -> Tuple[float, List[float]]:
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df[config.FORMATTED_TYPE_COLS] = df['y_true'].apply(parse_full_type)
        df[config.PRED_TYPE_COLS] = df['y_pred'].apply(parse_full_type)
        accuracies = [accuracy_score(df[true_col], df[pred_col])
                      for true_col, pred_col in zip(config.FORMATTED_TYPE_COLS, config.PRED_TYPE_COLS)]
        overall_acc = np.mean(accuracies[1:])
        return overall_acc, accuracies

    def save_model(self) -> None:
        remove_file_or_dir(self.model_fp)

        pickle.dump(self.model, open(self.model_fp, 'wb'))
        logger.info(f"Model has been saved at {self.model_fp}")

    def load_model(self) -> None:
        self.model = pickle.load(open(self.model_fp, 'rb'))
        logger.info(f"Model is loaded from {self.model_fp}")
