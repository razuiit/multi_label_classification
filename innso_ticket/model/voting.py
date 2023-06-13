from typing import List

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import config
from model.ml_model import MLModel
from util import logutil

logger = logutil.logger_run

ESTIMATORS = [
    LogisticRegression(multi_class='multinomial', random_state=config.SEED),
    RandomForestClassifier(n_estimators=1000, random_state=config.SEED, class_weight='balanced_subsample'),
    GaussianNB(),
]


class Voting(MLModel):
    def __init__(self,
                 scope_name: str,
                 full_types: List[str]) -> None:
        super(Voting, self).__init__(scope_name, full_types)

        self.model = VotingClassifier(estimators=[(e.__class__.__name__, e) for e in ESTIMATORS], voting='hard')
