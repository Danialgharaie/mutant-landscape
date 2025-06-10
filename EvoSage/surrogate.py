from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression

AA_CODES = "ACDEFGHIKLMNPQRSTVWY"
AA_MAP = {aa: i for i, aa in enumerate(AA_CODES)}


class SurrogateModel:
    """Lightweight surrogate model with prediction caching."""

    def __init__(self) -> None:
        self.model = LinearRegression()
        self.cache: Dict[str, float] = {}
        self.fitted = False

    def _featurize(self, seq: str) -> List[int]:
        return [AA_MAP.get(aa, -1) for aa in seq]

    def fit(self, df) -> None:
        if df.empty:
            return
        X = np.array([self._featurize(s) for s in df["seq"]])
        y = df["additive"].astype(float).values
        self.model.fit(X, y)
        self.cache.clear()
        self.fitted = True

    def predict(self, seq: str) -> float:
        if seq in self.cache:
            return self.cache[seq]
        if not self.fitted:
            return 0.0
        x = np.array([self._featurize(seq)])
        pred = float(self.model.predict(x)[0])
        self.cache[seq] = pred
        return pred
