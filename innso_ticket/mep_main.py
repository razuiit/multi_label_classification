import importlib
from typing import Dict, List

import pandas as pd

import config
from model.base import BaseModel
from preprocessing.embedding import Embedder
from preprocessing.preprocess import preprocess_text_data
from unit.data import make_data
from util import logutil
from util.io import load_model_info
from util.utils import parse_full_type_to_short_types, df_to_json

logger = logutil.logger_run

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 2000)
pd.set_option('max_colwidth', -1)


def load_models() -> Dict[str, BaseModel]:
    def load_model(scope_name: str,
                   model_module: str,
                   model_name: str,
                   types: List[str]) -> BaseModel:
        module = importlib.import_module(model_module)
        class_ = getattr(module, model_name)
        model = class_(scope_name, types)
        model.load_model()
        return model

    model_info = load_model_info()
    models = {scope_name: load_model(scope_name, **info) for scope_name, info in model_info.items()}
    return models


class InferenceModel:
    def load(self) -> None:
        self.models = load_models()

    def calc(self,
             request: dict) -> dict:
        logger.info('start')
        df = pd.json_normalize(request, record_path=['data', 'queries'])
        df = preprocess_text_data(df)

        ids, predictions = [], []
        for scope_name, gdf in df.groupby(config.TYPE_1_COL):
            scope_name = scope_name.lower().strip()
            logger.info(f"Business Scope:    {scope_name}")
            embedder = Embedder(scope_name)
            embedding = embedder.get_embedding(gdf, fit=False)
            data = make_data(embedding, gdf, self.models[scope_name].full_types)

            preds = self.models[scope_name].predict(data)
            ids += gdf['id'].to_list()
            predictions += preds

        df = pd.DataFrame({'id': ids, 'prediction': predictions})
        df[config.TYPE_COLS] = df['prediction'].apply(parse_full_type_to_short_types)
        df.drop(columns=['prediction'], inplace=True)

        # wrap result
        result = {
            'version': request['version'],
            'meta': {
                'uuid': request['meta']['uuid']
            },
            'result': {
                'code': 1,
                'des': "XXX",
                'prediction': df_to_json(df)
            }
        }

        logger.info('end')
        return result


if __name__ == '__main__':
    from preprocessing.preprocess import load_data

    scope_name = 'music'.lower()
    data = load_data([scope_name], preprocessed=True)

    sdf = data[0][1].sample(5, random_state=config.SEED)[[config.TYPE_1_COL, config.SUMMARY_COL, config.CONTENT_COL]]
    sdf['id'] = list(range(len(sdf)))
    logger.info(sdf)
    queries = df_to_json(sdf)

    request = {
        'version': '1.0.0',
        'meta': {
            'uuid': 'abc',
        },
        'data': {
            'queries': queries,
        }
    }

    mdl = InferenceModel()
    mdl.load()
    logger.info(mdl.calc(request))
