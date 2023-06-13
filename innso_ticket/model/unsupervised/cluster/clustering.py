import argparse
import os
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA

import config
from util.file_io import load_doc_reprs, save_clustering
from model.unsupervised.cluster.base_clustering import BaseClustering
from model.unsupervised.cluster.gmm import GMM
from model.unsupervised.cluster.mkmeans import MKMeans
from model.unsupervised.embeddings import cosine_similarity_embedding

CLUSTERINGS = {
    'gmm': GMM,
    'mkmeans': MKMeans,
}


def __reduce_dimensions(text_reprs: np.ndarray,
                        class_reprs: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    # use pca
    pca_n_components = min(config.PCA_N_COMPONENTS, len(text_reprs))
    _pca = PCA(n_components=pca_n_components, random_state=config.RANDOM_STATE)
    text_reprs = _pca.fit_transform(text_reprs)
    class_reprs = _pca.transform(class_reprs)
    return text_reprs, class_reprs


def run(text_reprs: np.ndarray,
        class_reprs: List[np.ndarray],
        cluster_method: str) -> Tuple[
    BaseClustering,  # clustering
    np.ndarray,  # prediction
    np.ndarray,  # centers
    np.ndarray,  # distance
    np.ndarray  # distance2
]:
    text_reprs, class_reprs = __reduce_dimensions(text_reprs, class_reprs)

    clustering = CLUSTERINGS[cluster_method](class_reprs)
    prediction, centers = clustering.fit_and_predict(text_reprs)

    distance = np.array([[cosine_similarity_embedding(s, c) for c in centers] for s in text_reprs])
    distance2 = np.array([[np.linalg.norm(s - c) for c in centers] for s in text_reprs])

    print(f"{os.path.splitext(os.path.basename(__file__))[0]} is done.")

    return clustering, prediction, centers, distance, distance2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cluster_method", type=str, required=True)

    args = parser.parse_args()

    text_reprs, class_reprs, _, _ = load_doc_reprs(args.data_dir)

    clustering_model, prediction, centers, distance, distance2 = run(text_reprs, class_reprs, args.cluster_method)

    save_clustering(args.data_dir, clustering_model, prediction, centers, distance, distance2)
