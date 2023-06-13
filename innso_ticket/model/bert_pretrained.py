import os
import pickle
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFBertModel, BertTokenizerFast

import config
from model.base import BaseModel
from unit.data import Data
from util import logutil
from util.utils import model_from_pretrained, remove_file_or_dir

logger = logutil.logger_run

tf.random.set_seed(config.SEED)

MAX_LENGTH = 140
BATCH_SIZE = 16
VALIDATION_SIZE = 0.15


class BERT(BaseModel):
    def __init__(self,
                 scope_name: str,
                 full_types: List[str] = None) -> None:
        super(BERT, self).__init__(scope_name, full_types)

        self.tokenizer = model_from_pretrained(BertTokenizerFast, config.BERT_MODEL_NAME)

        if self.full_types is not None:
            self.num_types = len(full_types)
            self.model = self.__bert_model(self.num_types)

    def __bert_model(self,
                     num_types: int):
        bert_encoder = model_from_pretrained(TFBertModel, config.BERT_MODEL_NAME, output_attentions=True)

        input_word_ids = tf.keras.layers.Input(
            shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids"
        )
        last_hidden_states = bert_encoder(input_word_ids)[0]
        clf_output = tf.keras.layers.Flatten()(last_hidden_states)
        net = tf.keras.layers.Dense(512, activation="relu")(clf_output)
        net = tf.keras.layers.Dropout(0.3)(net)
        net = tf.keras.layers.Dense(440, activation="relu")(net)
        net = tf.keras.layers.Dropout(0.3)(net)
        output = tf.keras.layers.Dense(num_types, activation="softplus")(net)
        model = tf.keras.models.Model(inputs=input_word_ids, outputs=output)

        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(
            loss="categorical_crossentropy", optimizer=adam_optimizer, metrics=["accuracy"]
        )

        return model

    def __encode(self,
                 texts: np.ndarray) -> tf.Tensor:
        encoding = self.tokenizer.batch_encode_plus(texts.tolist(),
                                                    max_length=MAX_LENGTH,
                                                    padding='max_length',
                                                    truncation=True)
        return tf.constant(encoding["input_ids"])

    def __calc_full_type(self,
                         y_pred: np.ndarray) -> List[str]:
        return [self.full_types[i] for i in np.argmax(y_pred, axis=1)]

    def train(self,
              train_data: Data,
              test_data: Data) -> Tuple[float, List[float]]:
        X_train, X_val, y_train, y_val = train_test_split(train_data.X_text,
                                                          train_data.y_index,
                                                          test_size=VALIDATION_SIZE,
                                                          random_state=config.SEED)
        X_train = self.__encode(X_train)
        X_val = self.__encode(X_val)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.num_types)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=self.num_types)

        train_data = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                      .shuffle(100)
                      .batch(BATCH_SIZE)
                      ).cache()
        val_data = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                    .shuffle(100)
                    .batch(BATCH_SIZE)
                    ).cache()

        self.model.fit(train_data,
                       batch_size=BATCH_SIZE,
                       epochs=1,  # todo: change epochs
                       validation_data=val_data,
                       verbose=1)

        X_test = self.__encode(test_data.X_text)
        test_dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE)
        y_pred = self.model.predict(test_dataset, batch_size=BATCH_SIZE)

        y_pred = self.__calc_full_type(y_pred)
        y_true = test_data.y_type.tolist()

        return self._calc_accuracies(y_true, y_pred)

    def predict(self,
                data: Data) -> List[str]:
        test_encoded = self.__encode(data.X_text)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_encoded).batch(BATCH_SIZE)
        predictions_matrix = self.model.predict(test_dataset, batch_size=BATCH_SIZE)
        predictions = [self.full_types[i] for i in np.argmax(predictions_matrix, axis=1)]
        return predictions

    def save_model(self) -> None:
        remove_file_or_dir(self.model_fp)

        self.model.save_weights(os.path.join(self.model_fp, f"{self.__class__.__name__}_{self.scope_name}"))

        pickle.dump(self.full_types, open(os.path.join(self.model_fp, f"types_{self.scope_name}"), 'wb'))

        logger.info(f"Model has been saved at {self.model_fp}")

    def load_model(self) -> None:
        self.full_types = pickle.load(open(os.path.join(self.model_fp, f"types_{self.scope_name}"), 'rb'))

        self.num_types = len(self.full_types)
        self.model = self.__bert_model(self.num_types)

        self.model.load_weights(os.path.join(self.model_fp, f"{self.__class__.__name__}_{self.scope_name}")
                                ).expect_partial()

        logger.info(f"Model is loaded from {self.model_fp}")
