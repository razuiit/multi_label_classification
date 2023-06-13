from abc import ABC, abstractmethod
from typing import List
from typing import Tuple

import numpy as np


class BaseClustering(ABC):
    @abstractmethod
    def __init__(self,
                 num_classes: int,
                 class_reprs: np.ndarray):
        ...

    @abstractmethod
    def fit_and_predict(self,
                        text_reprs: np.ndarray) -> Tuple[List[int], np.ndarray]:
        ...
