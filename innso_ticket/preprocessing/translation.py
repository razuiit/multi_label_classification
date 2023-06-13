from typing import List

import stanza
from stanza.pipeline.core import DownloadMethod
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import config
from util.utils import model_from_pretrained


class Translator:
    def __init__(self):
        self.model = model_from_pretrained(M2M100ForConditionalGeneration, config.TRANSLATION_MODEL_NAME)
        self.tokenizer = model_from_pretrained(M2M100Tokenizer, config.TRANSLATION_MODEL_NAME)

        self.nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid",
                                          download_method=DownloadMethod.REUSE_RESOURCES)

    def translate(self,
                  texts: List[str]) -> List[str]:
        text_en_l = []

        for text in tqdm(texts):
            if text == "":
                text_en_l = text_en_l + [text]
                continue

            doc = self.nlp_stanza(text)
            if doc.lang == "en":
                text_en_l = text_en_l + [text]
            else:
                # convert to model supported language code
                # https://stanfordnlp.github.io/stanza/available_models.html
                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/m2m_100/tokenization_m2m_100.py
                lang = doc.lang
                if lang == "fro":  # fro = Old French
                    lang = "fr"
                elif lang == "la":  # latin
                    lang = "it"
                elif lang == "nn":  # Norwegian (Nynorsk)
                    lang = "no"
                elif lang == "kmr":  # Kurmanji
                    lang = "tr"
                elif lang == "hsb":  # Upper Sorbian
                    lang = "de"
                elif lang == "mt":
                    lang = "en"

                self.tokenizer.src_lang = lang
                encoded_hi = self.tokenizer(text, return_tensors="pt")
                generated_tokens = self.model.generate(**encoded_hi,
                                                       forced_bos_token_id=self.tokenizer.get_lang_id("en"))
                text_en = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                text_en = text_en[0]
                text_en_l = text_en_l + [text_en]

        return text_en_l
