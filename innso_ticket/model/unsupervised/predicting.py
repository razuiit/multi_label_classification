import argparse
import os
import re
from typing import List, Tuple

import nltk
import numpy as np
import pandas as pd
import stanza
from stanza.pipeline.core import DownloadMethod
from stanza.resources.common import DEFAULT_MODEL_DIR

from util.file_io import load_preprocessed, save_result, load_clustering


def __predict_1(prediction: np.ndarray,
                classes: List[str]) -> List[str]:
    single_pred = [classes[pred] for pred in prediction]
    return single_pred


def __predict_2(texts: List[str],
                classes: List[str],
                text_tokenization_info: List[Tuple[List[str], List[Tuple[int, int, int]], List[List[int]]]],
                similarities: Tuple[np.ndarray],
                cluster_predictions: np.ndarray,
                cluster_similarities: np.ndarray) -> Tuple[
    List[str],  # pred_classes
    np.ndarray,  # sims
]:
    download_method = DownloadMethod.REUSE_RESOURCES if os.path.exists(
        os.path.join(DEFAULT_MODEL_DIR, "resources.json")) else DownloadMethod.DOWNLOAD_RESOURCES
    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', download_method=download_method)

    pred_classes, sims = [], []

    for pred_cluster, cluster_sim, text, (tokenized_text, tokenized_to_id_indicies, tokenids_chunks), word_sim in zip(
            cluster_predictions,
            cluster_similarities,
            texts,
            text_tokenization_info,
            similarities):

        pred_class = classes[pred_cluster]
        sim = cluster_sim

        pos_tag = nltk.pos_tag(tokenized_text)
        tokenized_text = [token for token, pos in pos_tag if re.search('[0-9a-zA-Z]', token)]

        # dependency parser
        doc = nlp(text)

        head_words = set()
        for sent in doc.sentences:
            for word in sent.words:
                if word.head > 0:
                    if word.upos == 'NOUN':
                        head_words.add(word.text)
                    elif word.upos == 'ADJ':
                        head_words.add(sent.words[word.head - 1].text)

        scored_classes = {}

        for head_word in head_words:
            if head_word not in tokenized_text:
                break

            word_idx = tokenized_text.index(head_word)
            class_sim = word_sim[word_idx]
            class_idx = np.argmax(class_sim)  # get most similar class to key(aspect)
            max_sim = np.max(word_sim[word_idx])

            if max_sim > 0.49:  # vary the threshold accordingly
                if (class_idx not in scored_classes) or (max_sim > scored_classes[class_idx][0]):
                    scored_classes[class_idx] = (max_sim, class_sim)

        else:
            # If dependency Parser based method works to detect desired aspects
            if len(scored_classes) > 1:
                sorted_classes = sorted(list(scored_classes.items()), key=lambda x: x[1][0], reverse=True)
                pred_class = classes[sorted_classes[0][0]]
                sim = sorted_classes[0][1][1]

        pred_classes.append(pred_class)
        sims.append(sim)

    return pred_classes, np.array(sims)


def run(df: pd.DataFrame,
        classes: List[str],
        prediction: np.ndarray) -> pd.DataFrame:
    # texts = df[config.PREPROCESSED_TEXT_COL].to_list()
    #
    # (word_rep, word_to_index, vocab_words, vocab_occurrence, selected_words, text_words_reprs,
    #  class_words_reprs) = load_static(data_dir)
    #
    # text_tokenization_info, class_tokenization_info = load_tokenization(data_dir)
    #
    # text_reprs, class_reprs, similarities, candidate_classes = load_doc_reprs(data_dir)

    df['pred_full_type_1'] = __predict_1(prediction, classes)
    # df['pred_full_type_2'], sims_2 = __predict_2(texts,
    #                                              classes,
    #                                              text_tokenization_info,
    #                                              similarities,
    #                                              prediction,
    #                                              distance)

    # df = pd.concat([df, pd.DataFrame(sims_2, columns=classes)], axis=1)

    print(f"{os.path.splitext(os.path.basename(__file__))[0]} is done.")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    data = load_preprocessed(args.data_dir)
    _, df, classes = data[0]
    _, prediction, _, _, _ = load_clustering(args.data_dir)

    df = run(df, classes, prediction)

    save_result(args.data_dir, df)
