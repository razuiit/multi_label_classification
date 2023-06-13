from typing import List, Tuple

import pandas as pd

import config
from model.base import BaseModel
from model.unsupervised import doc_representation, predicting
from model.unsupervised import static_representation
from model.unsupervised.cluster import clustering
from unit.data import Data

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class UnsupervisedModel(BaseModel):

    def train(self,
              train_data: Data,
              test_data: Data) -> Tuple[float, List[float]]:
        # X_train, y_train = RandomOverSampler(random_state=config.SEED).fit_resample(train_data.X_embedding,
        #                                                                             train_data.y_type)
        #
        # self.model = self.model.fit(X_train, y_train)
        #
        # y_pred = self.model.predict(test_data.X_embedding).tolist()
        # y_true = test_data.y_type.tolist()
        #
        # return self._calc_accuracies(y_true, y_pred)
        (
            _, _, word_to_index, _, text_tokenization_info, class_tokenization_info,
            text_words_reprs, class_words_reprs
        ) = static_representation.run(train_data.X_text.to_list(),
                                      train_data.y_type.unique().tolist(),
                                      config.UNSUPERVISED_MODEL_NAME)

        text_reprs, class_reprs, similarities, candidate_classes = doc_representation.run(word_to_index,
                                                                                          text_words_reprs,
                                                                                          class_words_reprs,
                                                                                          text_tokenization_info,
                                                                                          class_tokenization_info)

        clustering_model, prediction, centers, distance, distance2 = clustering.run(text_reprs,
                                                                                    class_reprs,
                                                                                    config.CLUSTER_METHOD)
        df = predicting.run(test_data.X_text, test_data.y_type, prediction)
        print()

    def predict(self,
                data: Data) -> List[str]:
        return self.model.predict(data.X_embedding).tolist()


def run_single(dataset_name: str,
               df: pd.DataFrame,
               classes: List[str],
               model_name: str,
               cluster_method: str) -> pd.Series:
    print(f"""
    dataset:         {dataset_name}
    model_name:      {model_name}
    cluster_method:  {cluster_method}
    """)

    (
        _, _, word_to_index, _, text_tokenization_info, class_tokenization_info,
        text_words_reprs, class_words_reprs
    ) = static_representation.run(df, classes, model_name)

    text_reprs, class_reprs, similarities, candidate_classes = doc_representation.run(word_to_index,
                                                                                      text_words_reprs,
                                                                                      class_words_reprs,
                                                                                      text_tokenization_info,
                                                                                      class_tokenization_info)

    clustering_model, prediction, centers, distance, distance2 = clustering.run(text_reprs, class_reprs, cluster_method)

    df = predicting.run(df, classes, prediction)

    # evaluation = postprocessing.run(dataset_name, model_name, cluster_method, df)
    # return evaluation
