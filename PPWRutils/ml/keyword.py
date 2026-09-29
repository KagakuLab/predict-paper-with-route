import re
from typing import Iterable, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

OPRD_KEYWORDS: List[str] = [

    # General synthesis
    "synthesis", "synthetic", "preparation", "prepared", "preparative", "production",

    # Route-related
    "synthetic route", "synthesis route", "route to", "route for", "process route",
    "improved route", "new route", "alternative route", "efficient route",

    # Process/manufacturing chemistry
    "process development", "process chemistry", "process research", "process optimization",
    "process optimisation", "scale-up", "scale up", "scalable synthesis",
    "large-scale synthesis", "large scale synthesis", "manufacturing process",
    "commercial manufacture", "industrial synthesis", "pilot plant",

    # Route development
    "development of a process", "development of an efficient synthesis",
    "development of a scalable synthesis", "development of a manufacturing process",
    "optimization of the synthesis", "optimisation of the synthesis",
    "optimization of a synthetic route", "optimisation of a synthetic route",

    # Intermediates and building blocks
    "key intermediate", "advanced intermediate", "pharmaceutical intermediate",
    "intermediate for the synthesis",

    # API / drug manufacture
    "active pharmaceutical ingredient", "api synthesis", "drug substance",
    "drug product synthesis",
]


class KeywordClassifier(BaseEstimator, ClassifierMixin):
    """
    Predicts whether a text (e.g. paper title or abstract) is likely to
    describe a multi-step synthesis route, based on presence of any keyword
    from a provided list.
    """

    def __init__(self, keywords = None):
        self.keywords = list(keywords) if keywords is not None else OPRD_KEYWORDS

    def fit(self, X=None, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def _score_one(self, text: str) -> int:
        if not isinstance(text, str):
            return 0
        text = text.lower()
        return int(any(keyword in text for keyword in self.keywords))

    def predict(self, X) -> np.ndarray:
        return np.array([self._score_one(x) for x in X])

    def predict_proba(self, X) -> np.ndarray:
        preds = self.predict(X)
        return np.column_stack([1 - preds, preds])