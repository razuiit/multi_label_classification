from typing import List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

import config
from model.unsupervised.cluster.base_clustering import BaseClustering
from model.unsupervised.embeddings import cosine_similarity_embeddings


class GMM(BaseClustering):

    def __init__(self,
                 class_reprs: np.ndarray):
        self.model = GaussianMixture(n_components=class_reprs.shape[0], covariance_type='tied',
                                     random_state=config.RANDOM_STATE, n_init=999, warm_start=True)
        self.model.converged_ = 'HACK'
        self.class_reprs = class_reprs

    def fit_and_predict(self,
                        text_reprs: np.ndarray) -> Tuple[List[int], np.ndarray]:
        cosine_similarities = cosine_similarity_embeddings(text_reprs, self.class_reprs)
        class_initialization = np.argmax(cosine_similarities, axis=1)
        class_initialization_matrix = np.zeros((text_reprs.shape[0], self.class_reprs.shape[0]))
        for i in range(text_reprs.shape[0]):
            class_initialization_matrix[i][class_initialization[i]] = 1.0

        self.model._initialize(text_reprs, class_initialization_matrix)
        self.model.lower_bound_ = -np.infty
        self.model.fit(text_reprs)

        prediction = self.model.predict(text_reprs)
        centers = self.model.means_
        # distance = -gmm.predict_proba(sentence_representations) + 1
        # pp = gmm.predict_proba(sentence_representations)

        return prediction, centers
