import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

import config
from util import logutil
from util.io import save_tfidf_converter, load_tfidf_converter
from util.utils import model_from_pretrained

logger = logutil.logger_run

warnings.filterwarnings('ignore')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(config.SEED)


class Embedder:
    def __init__(self,
                 scope_name: str,
                 sentence_transformer_model_name: str = config.SENTENCE_TRANSFORMER_MODEL_NAME,
                 bert_model_name: str = config.BERT_MODEL_NAME):
        self.scope_name = scope_name
        self.bert_model_name = bert_model_name
        self.sentence_transformer_model_name = sentence_transformer_model_name

    def __get_sentence_transformer_embedding(self,
                                             *texts_list: List[List[str]]):
        st_model = SentenceTransformer(self.sentence_transformer_model_name)
        embedding = np.concatenate([st_model.encode(texts) for texts in texts_list], axis=1)
        return embedding

    def __get_tfidf_embedding(self,
                              texts: List[str],
                              fit: bool = False) -> np.ndarray:
        if fit:
            tfidf_converter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
            tfidf_converter.fit(texts)
            save_tfidf_converter(self.scope_name, tfidf_converter)
        else:
            tfidf_converter = load_tfidf_converter(self.scope_name)

        embedding = tfidf_converter.transform(texts).toarray()

        return embedding

    def __get_bert_pretrained_embedding(self,
                                        texts: List[str]) -> np.ndarray:

        tokenizer = model_from_pretrained(BertTokenizer, self.bert_model_name)
        model = model_from_pretrained(BertModel, self.bert_model_name, output_hidden_states=True)

        batch_size = 32
        batch_embeddings = []
        for idx in range(0, len(texts), batch_size):
            batch = texts[idx: idx + batch_size]
            encoded = tokenizer.batch_encode_plus(batch, max_length=50, padding='max_length', truncation=True)
            encoded = {key: torch.LongTensor(value) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
                batch_embeddings.append(outputs['pooler_output'])
        embedding = np.concatenate(batch_embeddings, axis=0)

        return embedding

    def get_embedding(self,
                      df: pd.DataFrame,
                      fit: bool = False) -> np.ndarray:
        texts, summary_texts, content_texts = [df[col].tolist() for col in [config.TEXT_COL,
                                                                            config.SUMMARY_COL,
                                                                            config.CONTENT_COL]]
        embedding = np.concatenate((self.__get_bert_pretrained_embedding(texts),
                                    self.__get_sentence_transformer_embedding(summary_texts, content_texts),
                                    self.__get_tfidf_embedding(texts, fit)), axis=1)
        return embedding
