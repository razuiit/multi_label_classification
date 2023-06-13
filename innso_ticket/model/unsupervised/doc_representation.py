import argparse
import os
from typing import Tuple, List, Set, Dict

import numpy as np
from scipy.special import softmax
from tqdm import tqdm

from util.file_io import load_static, save_doc_reprs, load_tokenization
from model.unsupervised.embeddings import cosine_similarity_embeddings


# todo:
#  - `similarities2`, `candidate_classes2` not used.
#  - rename function variables: text_repr, class_reprs ...

def __weight_sentence(word_to_index: Tuple[str, int],
                      text_tokenization_info: Tuple[List[str], List[Tuple[int, int, int]], List[List[int]]],
                      text_repr: np.ndarray,
                      class_reprs: List[np.ndarray]
                      ) -> Tuple[
    np.ndarray,  # weighted_text_repr, shape: (embedding_len, )
    np.ndarray,  # similarities, shape: (num_tokens, num_classes)
    Set[int]  # candidate_classes
]:
    tokenized_text, tokenized_to_id_indicies, tokenids_chunks = text_tokenization_info

    """
    weight_sentence_with_attention
    """
    assert len(tokenized_text) == len(text_repr)

    text_repr = np.array([word_repr
                          for token, word_repr in zip(tokenized_text, text_repr)
                          if token in word_to_index])

    # todo: potential bug in `__rank_by_significance()` when len(text_repr) == 0
    if len(text_repr) == 0:
        print("Empty Sentence (or sentence with no words that have enough frequency)")
        return np.average(text_repr, axis=0), np.empty((0, 0)), set()

    """
    rank_by_significance
    """
    # print('inside rank_by_significance..')
    similarities = np.array([cosine_similarity_embeddings(text_repr, class_repr).mean(axis=1)
                             for class_repr in class_reprs]).T
    # print('similarities: ', similarities)

    sim_thres = similarities.max() * 0.9
    candidate_classes = set([np.argmax(similarity) for similarity in similarities if np.max(similarity) > sim_thres])

    significance_scores = [np.max(softmax(similarity)) for similarity in similarities]
    ranking = np.argsort(np.argsort(-np.array(significance_scores)))
    weights = [1. / (r + 1) for r in ranking]

    weighted_text_repr = np.average(text_repr, weights=weights, axis=0)

    return weighted_text_repr, similarities, candidate_classes


# sentence representation
def run(word_to_index: Dict[str, int],
        text_words_reprs: List[np.ndarray],
        class_words_reprs: List[np.ndarray],
        text_tokenization_info: List[Tuple[List[str], List[Tuple[int, int, int]], List[List[int]]]],
        class_tokenization_info: List[Tuple[List[str], List[Tuple[int, int, int]], List[List[int]]]]
        ) -> Tuple[
    np.ndarray,  # text_reprs, shape: (num_texts, embedding_len)
    np.ndarray,  # class_reprs, shape: (num_classes, embedding_len)
    Tuple[np.ndarray],  # similarities, shape: num_texts * (num_tokens, num_classes)
    Tuple[set]  # candidate_classes, length: num_texts
]:
    text_reprs, similarities, candidate_classes = list(zip(*[__weight_sentence(word_to_index,
                                                                               token_info,
                                                                               text_repr,
                                                                               class_words_reprs)
                                                             for token_info, text_repr in
                                                             tqdm(zip(text_tokenization_info,
                                                                      text_words_reprs))]))

    class_reprs, similarities2, candidate_classes2 = list(zip(*[__weight_sentence(word_to_index,
                                                                                  token_info,
                                                                                  class_repr,
                                                                                  text_words_reprs)
                                                                for token_info, class_repr in
                                                                tqdm(zip(class_tokenization_info,
                                                                         class_words_reprs))]))

    print(f"{os.path.splitext(os.path.basename(__file__))[0]} is done.")

    return np.array(text_reprs), np.array(class_reprs), similarities, candidate_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)

    args = parser.parse_args()

    _, _, word_to_index, _, text_words_reprs, class_words_reprs = load_static(args.data_dir)
    text_tokenization_info, class_tokenization_info = load_tokenization(args.data_dir)

    text_reprs, class_reprs, similarities, candidate_classes = run(word_to_index,
                                                                   text_words_reprs,
                                                                   class_words_reprs,
                                                                   text_tokenization_info,
                                                                   class_tokenization_info)

    save_doc_reprs(args.data_dir, text_reprs, class_reprs, similarities, candidate_classes)
