import json
import os
import re
from collections import Counter

from nltk.corpus import stopwords

INNSO_STOPWORDS = ['huawei']


def load_typology2_map(data_dir):
    with open(os.path.join(data_dir, 'typology_mapping.json'), "r") as f:
        word_to_typology_map = json.load(f)
    return word_to_typology_map


def create_mapping_file(df, data_dir, high_type_col, low_type_col, use_stopwords=True):
    word_to_typology2_map = {}
    for typology1, typologies2 in df.groupby(high_type_col)[low_type_col].unique().iteritems():
        typologies2 = [re.sub('[^0-9a-zA-Z]+', ' ', c).lower() for c in typologies2]
        counter = Counter([t for t in re.sub('[^0-9a-zA-Z]+', ' ', ' '.join(typologies2)).split()])
        ignore_words = [k for k, v in counter.items() if (v > 1) or (len(k) == 1)] + INNSO_STOPWORDS + (
            stopwords.words() if use_stopwords else [])
        word_to_typology2_map[typology1] = {w: t2
                                            for t2 in typologies2
                                            for w in t2.split()
                                            if w not in ignore_words}

    print(json.dumps(word_to_typology2_map, indent=4))

    with open(os.path.join(data_dir, 'typology_mapping.json'), "w") as f:
        json.dump(word_to_typology2_map, f, indent=4)

    return word_to_typology2_map
