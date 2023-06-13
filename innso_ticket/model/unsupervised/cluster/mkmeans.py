from typing import Tuple, List

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import config
from model.unsupervised.cluster.base_clustering import BaseClustering


class MKMeans(BaseClustering):

    def __init__(self,
                 class_reprs: np.ndarray):
        self.model = MiniBatchKMeans(n_clusters=class_reprs.shape[0], init=class_reprs,
                                     random_state=config.RANDOM_STATE, batch_size=400)

    def fit_and_predict(self,
                        text_reprs: np.ndarray) -> Tuple[List[int], np.ndarray]:
        self.model.fit(text_reprs)

        prediction = self.model.predict(text_reprs)
        centers = self.model.cluster_centers_
        return prediction, centers
