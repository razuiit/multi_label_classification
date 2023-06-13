import os
import pickle
from typing import List, Tuple, Dict, Set

import numpy as np
import pandas as pd

import config
from model.unsupervised.cluster.base_clustering import BaseClustering

PREPROCESSED_FN = 'preprocessed.pk'
TOKENIZATION_FN = 'tokenization.pk'
STATIC_REPRS_FN = 'static-reprs.pk'
CLASS_REPRS_FN = 'class-reprs.pk'
DOC_REPRS_FN = 'doc-reprs.pk'
CLUSTERING_FN = 'clustering.pk'
RESULT_FN = 'result.csv'
EVALUATIONS_FN = 'evaluations.csv'


def load_raw_dataset(data_dir: str,
                     dataset: str) -> pd.DataFrame:
    fp = os.path.join(data_dir, f"{dataset}.csv")
    df = pd.read_csv(fp)[config.INPUT_COLS]
    print(f"`{dataset}` DF shape: {df.shape}")
    return df


def load_raw_datasets(data_dir: str,
                      datasets: List[str]) -> pd.DataFrame:
    return pd.concat([load_raw_dataset(data_dir, dataset) for dataset in datasets])


def load_dataset(data_dir: str,
                 dataset: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, f"{dataset}.csv"))


def load_classes(data_dir: str,
                 classes_fn: str) -> List[str]:
    with open(os.path.join(data_dir, classes_fn), mode='r', encoding='utf-8') as f:
        classes = list(map(str.lower, ''.join(f.readlines()).strip().split('\n')))
    return classes


def save_preprocessed(data_dir: str,
                      data: Tuple[str, pd.DataFrame, List[str]]) -> None:
    with open(os.path.join(data_dir, PREPROCESSED_FN), "wb") as f:
        pickle.dump(data, f)


def load_preprocessed(data_dir: str) -> Tuple[str, pd.DataFrame, List[str]]:
    with open(os.path.join(data_dir, PREPROCESSED_FN), "rb") as f:
        data = pickle.load(f)

    return data


def save_tokenization(data_dir: str,
                      text_tokenization_info: List[Tuple[List[str], List[Tuple[int, int, int]], List[List[int]]]],
                      class_tokenization_info: List[
                          Tuple[List[str], List[Tuple[int, int, int]], List[List[int]]]]) -> None:
    with open(os.path.join(data_dir, TOKENIZATION_FN), 'wb') as f:
        pickle.dump({
            'text_tokenization_info': text_tokenization_info,
            'class_tokenization_info': class_tokenization_info,
        }, f, protocol=4)


def load_tokenization(data_dir: str) -> List[Tuple[List[str], List[Tuple[int, int, int]], List[List[int]]]]:
    with open(os.path.join(data_dir, TOKENIZATION_FN), 'rb') as f:
        data = pickle.load(f)

        text_tokenization_info = data['text_tokenization_info']
        class_tokenization_info = data['class_tokenization_info']

    return text_tokenization_info, class_tokenization_info


def save_static(data_dir: str,
                word_rep: np.ndarray,
                vocab_words: List[str],
                word_to_index: Dict[str, int],
                vocab_occurrence: List[int],
                text_words_reprs: List[np.ndarray],
                class_words_reprs: List[np.ndarray]) -> None:
    with open(os.path.join(data_dir, STATIC_REPRS_FN), 'wb') as f:
        pickle.dump({
            'word_rep': word_rep,
            'vocab_words': vocab_words,
            'word_to_index': word_to_index,
            'vocab_occurrence': vocab_occurrence,
            'text_words_reprs': text_words_reprs,
            'class_words_reprs': class_words_reprs,
        }, f, protocol=4)


def load_static(data_dir: str) -> Tuple[
    np.ndarray,  # word_rep
    List[str],  # vocab_words
    Dict[str, int],  # word_to_index
    List[int],  # vocab_occurrence
    List[np.ndarray],  # text_words_reprs
    List[np.ndarray],  # class_words_reprs
]:
    with open(os.path.join(data_dir, STATIC_REPRS_FN), 'rb') as f:
        data = pickle.load(f)

        word_rep = data['word_rep']
        vocab_words = data['vocab_words']
        word_to_index = data['word_to_index']
        vocab_occurrence = data['vocab_occurrence']
        text_words_reprs = data['text_words_reprs']
        class_words_reprs = data['class_words_reprs']

    return word_rep, vocab_words, word_to_index, vocab_occurrence, text_words_reprs, class_words_reprs


def save_class_repr(data_dir: str,
                    classes: List[str],
                    class_representations: np.ndarray,
                    texts: List[str],
                    num_texts: int) -> None:
    with open(os.path.join(data_dir, CLASS_REPRS_FN), 'wb') as f:
        pickle.dump({
            'classes': classes,
            'class_representations': class_representations,
            'data_text': texts,
            'num_texts': num_texts,
        }, f, protocol=4)


def load_class_repr(data_dir: str) -> Tuple[
    List[str],  # classes
    np.ndarray,  # class_representations
    List[str],  # texts
    int  # num_texts
]:
    with open(os.path.join(data_dir, CLASS_REPRS_FN), 'rb') as f:
        data = pickle.load(f)

        classes = data['classes']
        class_representations = data['class_representations']
        data_text = data['data_text']
        num_texts = data['num_texts']

    return classes, class_representations, data_text, num_texts


def save_doc_reprs(data_dir: str,
                   text_reprs: np.ndarray,
                   class_reprs: np.ndarray,
                   similarities: Tuple[np.ndarray],
                   candidate_classes: Tuple[Set[int]]) -> None:
    with open(os.path.join(data_dir, DOC_REPRS_FN), 'wb') as f:
        pickle.dump({
            'text_reprs': text_reprs,
            'class_reprs': class_reprs,
            'similarities': similarities,
            'candidate_classes': candidate_classes
        }, f, protocol=4)


def load_doc_reprs(data_dir: str) -> Tuple[
    np.ndarray,  # text_reprs
    np.ndarray,  # class_reprs
    Tuple[np.ndarray],  # similarities
    Tuple[Set[int]],  # candidate_classes
]:
    with open(os.path.join(data_dir, DOC_REPRS_FN), 'rb') as f:
        data = pickle.load(f)

        text_reprs = data['text_reprs']
        class_reprs = data['class_reprs']
        similarities = data['similarities']
        candidate_classes = data['candidate_classes']

    return text_reprs, class_reprs, similarities, candidate_classes


def save_clustering(data_dir: str,
                    clustering_model: BaseClustering,
                    prediction: np.ndarray,
                    centers: np.ndarray,
                    distance: np.ndarray,
                    distance2: np.ndarray) -> None:
    with open(os.path.join(data_dir, CLUSTERING_FN), "wb") as f:
        pickle.dump({
            'clustering_model': clustering_model,
            'prediction': prediction,
            'centers': centers,
            'distance': distance,
            'distance2': distance2,
        }, f)


def load_clustering(data_dir: str) -> Tuple[
    BaseClustering,  # clustering_model
    np.ndarray,  # prediction
    np.ndarray,  # centers
    np.ndarray,  # distance
    np.ndarray  # distance2
]:
    with open(os.path.join(data_dir, CLUSTERING_FN), 'rb') as f:
        data = pickle.load(f)

        clustering_model = data['clustering_model']
        prediction = data['prediction']
        centers = data['centers']
        distance = data['distance']
        distance2 = data['distance2']

    return clustering_model, prediction, centers, distance, distance2


def save_result(data_dir: str,
                df: pd.DataFrame):
    df.to_csv(os.path.join(data_dir, RESULT_FN), index=False)


def load_result(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, RESULT_FN))


def save_evaluations(data_dir: str,
                     df: pd.DataFrame):
    df.to_csv(os.path.join(data_dir, EVALUATIONS_FN), index=False)
