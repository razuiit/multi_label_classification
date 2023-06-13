from typing import Tuple, List

from imblearn.over_sampling import RandomOverSampler

import config
from model.base import BaseModel
from unit.data import Data
from util import logutil

logger = logutil.logger_run


class MLModel(BaseModel):

    def train(self,
              train_data: Data,
              test_data: Data) -> Tuple[float, List[float]]:
        X_train, y_train = RandomOverSampler(random_state=config.SEED).fit_resample(train_data.X_embedding,
                                                                                    train_data.y_type)

        self.model = self.model.fit(X_train, y_train)

        y_pred = self.model.predict(test_data.X_embedding).tolist()
        y_true = test_data.y_type.tolist()

        return self._calc_accuracies(y_true, y_pred)

    def predict(self,
                data: Data) -> List[str]:
        return self.model.predict(data.X_embedding).tolist()
