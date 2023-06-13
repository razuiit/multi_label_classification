import numpy as np


def cosine_similarity_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))


# def dot_product_embeddings(emb_a, emb_b):
#     return np.dot(emb_a, np.transpose(emb_b))


def cosine_similarity_embedding(emb_a, emb_b):
    return np.dot(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)


# def pairwise_distances(x, y):
#     return cdist(x, y, 'euclidean')


def average_with_harmonic_series(representations):
    weights = [0.0] * len(representations)
    for i in range(len(representations)):
        weights[i] = 1. / (i + 1)
    return np.average(representations, weights=weights, axis=0)


def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
