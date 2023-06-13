import argparse
from typing import List

from train import train_model
from util import logutil

logger = logutil.logger_run


def main(type1_list: List[str],
         preprocessed: bool):
    # todo
    # print_versions()

    logger.info('start')
    train_model(type1_list, preprocessed)
    logger.info('end')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type1_list", type=str, required=True)
    parser.add_argument('--preprocessed', type=str, required=True)
    args, unknown = parser.parse_known_args()
    logger.info(f"args: {args}")

    type1_list = [x.replace('";"', ';') for x in filter(lambda x: len(x) > 0, args.type1_list.lower().split('^'))]
    preprocessed = args.preprocessed.lower() == 'true'
    logger.info(f"type1_list  =  {type1_list}")
    logger.info(f"preprocessed = {preprocessed}")

    main(type1_list, preprocessed)
