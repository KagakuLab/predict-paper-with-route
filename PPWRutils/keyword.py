import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


KEYWORDS = [
    # General synthesis
    "synthesis", "syntheses", "synthetic", "preparation", "prepared", "preparative", "production",

    # Route-related
    "synthetic route", "synthesis route", "route to", "route for", "process route",
    "improved route", "new route", "alternative route", "efficient route",

    # Process/manufacturing chemistry
    "process development", "process chemistry", "process research",
    "process optimization", "process optimisation",
    "scale-up", "scale up", "scalable synthesis",
    "large-scale synthesis", "large scale synthesis",
    "manufacturing process", "commercial manufacture",
    "industrial synthesis", "pilot plant",

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

    def __init__(self, keywords=None):
        self.keywords = keywords

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self._keywords = self.keywords if self.keywords is not None else KEYWORDS
        return self

    def predict(self, X):
        check_is_fitted(self)
        return np.array([self._label(text) for text in X])

    def predict_proba(self, X):
        check_is_fitted(self)
        preds = self.predict(X)
        # Hard 0/1 probabilities to satisfy the sklearn interface
        proba = np.zeros((len(preds), 2))
        proba[preds == 0, 0] = 1.0
        proba[preds == 1, 1] = 1.0
        return proba

    # ------------------------------------------------------------------
    def _label(self, text):
        if not isinstance(text, str):
            return 0
        text = text.lower()
        return int(any(kw in text for kw in self._keywords))