from typing import List

from sklearn.ensemble import RandomForestClassifier

import config
from model.ml_model import MLModel
from util import logutil

logger = logutil.logger_run


class RandomForest(MLModel):
    def __init__(self,
                 scope_name: str,
                 full_types: List[str]) -> None:
        super(RandomForest, self).__init__(scope_name, full_types)

        self.model = RandomForestClassifier(n_estimators=1000, random_state=config.SEED,
                                            class_weight='balanced_subsample')
