from typing import List, Tuple

import numpy as np
import torch
from transformers import BertModel

import config
from util.utils import model_from_pretrained


class DocumentEncoder:
    def __init__(self,
                 model_name: str):
        self.model = model_from_pretrained(model_class=BertModel, model_name=model_name, output_hidden_states=True)
        self.model.eval()
        # model.cuda()

    @staticmethod
    def __tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.clone().detach().cpu().numpy()

    @staticmethod
    def __sentence_to_wordtoken_embeddings(layer_embeddings: List[np.ndarray],
                                           tokenized_to_id_indicies: List[Tuple[int, int, int]]) -> np.ndarray:
        word_embeddings = np.array([
            np.average(layer_embeddings[chunk_index][start_index: end_index], axis=0)
            for chunk_index, start_index, end_index in tokenized_to_id_indicies
        ])
        return word_embeddings

    def __encode_sentence(self,
                          tokens_id: List[int]) -> np.ndarray:
        input_ids = torch.tensor([tokens_id], device=self.model.device)

        with torch.no_grad():
            hidden_states = self.model(input_ids)
        all_layer_outputs = hidden_states[2]

        layer_embedding = DocumentEncoder.__tensor_to_numpy(all_layer_outputs[config.NUM_LAYERS].squeeze(0))[1: -1]
        return layer_embedding

    def encode_document(self,
                        tokenized_text: List[str],
                        tokenized_to_id_indicies: List[Tuple[int, int, int]],
                        tokenids_chunks: List[List[int]]) -> np.ndarray:
        layer_embeddings = [self.__encode_sentence(tokenids_chunk) for tokenids_chunk in tokenids_chunks]
        word_embeddings = DocumentEncoder.__sentence_to_wordtoken_embeddings(layer_embeddings,
                                                                             tokenized_to_id_indicies)
        assert len(word_embeddings) == len(tokenized_text)
        return word_embeddings
