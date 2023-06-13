import os

from util import logutil

logger = logutil.logger_run

DATA_DIR = os.getenv('DATA_DIR', '../filestore/')
ENV = os.getenv('ENV', 'LOCAL').upper()

if ENV == 'MTP':
    MODEL_DIR = os.getenv('OUTPUT_DIR', '')
elif ENV == 'MEP':
    MODEL_DIR = '/model-data/model/'
else:
    MODEL_DIR = '../results/'

PREPROCESSED_FP = os.path.join(DATA_DIR, 'data/preprocessed.csv')

PRETRAINED_MODEL_DIR = os.path.join(DATA_DIR, 'pretrained')

MODEL_INFO_FILE = os.path.join(MODEL_DIR, 'model_info.json')
RESULT_FILE = os.path.join(MODEL_DIR, 'result.txt')
FULL_RESULT_FILE = os.path.join(MODEL_DIR, 'result.csv')

# input columns
TICKET_ID_COL = 'Ticket id'
INTERACTION_ID_COL = 'Interaction id'
SUMMARY_COL = 'Ticket Summary'
CONTENT_COL = 'Interaction content'
TYPE_COLS = ['Type 1', 'Type 2', 'Type 3', 'Type 4']
INPUT_COLS = [TICKET_ID_COL, INTERACTION_ID_COL, SUMMARY_COL, CONTENT_COL] + TYPE_COLS

# additional columns
TEXT_COL = 'text'
CLASS_COL = 'full_type'
TYPE_1_COL = TYPE_COLS[0]
FORMATTED_TYPE_COLS = [f"formatted_{col}" for col in TYPE_COLS]
PRED_TYPE_COLS = [f"pred_{col}" for col in TYPE_COLS]

JOIN_CHAR = '^'
EMPTY_TYPE = 'none'

SCOPE_MIN_RECORDS = 10

TRANSLATION_MODEL_NAME = 'facebook/m2m100_418M'
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
SENTENCE_TRANSFORMER_MODEL_NAME = 'all-MiniLM-L6-v2'
BERT_MODEL_HQ = 'bert-base-uncased.gz'
UNSUPERVISED_MODEL_NAME = 'nlptown/bert-base-multilingual-uncased-sentiment'
CLUSTER_METHOD = 'mkmeans'

TEST_SIZE = 0.2
SEED = 7

logger.info(f"os.env:     {os.environ}")
logger.info(f"ENV:        {ENV}")
logger.info(f"MODEL_DIR:  {MODEL_DIR}")
logger.info(f"DATA_DIR:   {DATA_DIR}")

"""
static_representation setting
"""
HAS_SOS_EOS = True
MODEL_MAX_TOKENS = 512
MAX_TOKENS = MODEL_MAX_TOKENS - 2 if HAS_SOS_EOS else MODEL_MAX_TOKENS
SLIDING_WINDOW_SIZE = MODEL_MAX_TOKENS // 2 - int(HAS_SOS_EOS)

logger.info(f"HAS_SOS_EOS = {HAS_SOS_EOS}")
logger.info(f"MODEL_MAX_TOKENS = {MODEL_MAX_TOKENS}")
logger.info(f"SLIDING_WINDOW_SIZE = {SLIDING_WINDOW_SIZE}")

"""
DocumentEncoder setting
"""
NUM_LAYERS = 12

"""
clustering setting
"""
RANDOM_STATE = 42
PCA_N_COMPONENTS = 64
