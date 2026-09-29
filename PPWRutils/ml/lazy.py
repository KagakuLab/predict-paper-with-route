# ==========================================================
# Imports
# ==========================================================
import os
import gc
import psutil
import time
import shutil
import warnings

import numpy as np
import pandas as pd

from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.utils.multiclass import type_of_target

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

from qsarcons.hopt import StepwiseHopt, DEFAULT_PARAM_GRID_CLASSIFIERS

warnings.filterwarnings("ignore")

VECTORIZERS = {

    # word n-grams, TF-IDF weighted
    "tfidf-word-1_2gram": lambda: TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True,
    ),
    "tfidf-word-1_3gram": lambda: TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 3),
        min_df=2, max_df=0.95, sublinear_tf=True,
    ),

    # character n-grams, TF-IDF weighted (robust to typos/morphology, e.g. "optimisation" vs "optimization")
    "tfidf-char-3_5gram": lambda: TfidfVectorizer(
        lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
        min_df=2, max_df=0.95, sublinear_tf=True,
    ),

    # raw word counts, no IDF reweighting
    "count-word-1_2gram": lambda: CountVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=2, max_df=0.95,
    ),

    # hashed word n-grams (no fit needed, fixed-size output, no vocabulary stored)
    "hashing-word-1_2gram": lambda: HashingVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        n_features=2 ** 15, alternate_sign=False,
    ),

    # TF-IDF followed by SVD (LSA): dense, lower-dimensional projection
    "tfidf-svd100": lambda: make_pipeline(
        TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            min_df=2, max_df=0.95, sublinear_tf=True,
        ),
        TruncatedSVD(n_components=100, random_state=42),
    ),
}

# Kept exactly as in the original lazy.py — StepwiseHopt and the param grids
# are reused unchanged from qsarcons.hopt.
CLASSIFIERS = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
    "MLPClassifier": MLPClassifier,
    "RidgeClassifier": RidgeClassifier,
    "SVC":SVC,
    "LinearSVC":LinearSVC
}

# ==========================================================
# Utility Functions
# ==========================================================
def to_dense(x):
    """Convert vectorizer output (possibly sparse) to a dense float array."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=float)


def vectorize_text(text_train, text_val, text_test, vectorizer):
    """
    Fit the vectorizer on the train split ONLY, then transform train/val/test.

    This replaces calc_descriptors from the molecular version: descriptor
    calculators were stateless (no fit), so no leakage risk existed there.
    Text vectorizers are not stateless, so fit/transform must be split like
    this to keep val/test genuinely held out.
    """
    x_train = to_dense(vectorizer.fit_transform(text_train))
    x_val = to_dense(vectorizer.transform(text_val))
    x_test = to_dense(vectorizer.transform(text_test))
    return x_train, x_val, x_test


def get_predictions(estimator, X):
    return estimator.predict(X).tolist()


def build_model(x_train, x_val, x_test, y_train, y_val, y_test, estimator_class, hopt=True):

    # 1. Optimize hyperparameters
    # NOTE: no scale_descriptors step here (unlike the molecular version) —
    # TF-IDF/count/hashing vectors are already normalized or scale-insensitive
    # for these classifiers. Revisit if you add scale-sensitive methods (e.g. SVR-like).
    if hopt:
        est_name = estimator_class.__name__
        param_grid = DEFAULT_PARAM_GRID_CLASSIFIERS.get(est_name)

        estimator_instance = estimator_class()
        stepwise_hopt = StepwiseHopt(estimator_instance, param_grid, verbose=False)
        stepwise_hopt.fit(x_train, y_train)
        estimator_instance = stepwise_hopt.estimator
    else:
        estimator_instance = estimator_class()

    # 2. Train on train split only (not final training yet)
    estimator_instance.fit(x_train, y_train)
    pred_train = get_predictions(estimator_instance, x_train)
    pred_val = get_predictions(estimator_instance, x_val)

    # 3. Retrain model on full (train + val)
    x_full, y_full = np.vstack([x_train, x_val]), np.hstack([y_train, y_val])

    estimator_instance.fit(x_full, y_full)
    pred_test = get_predictions(estimator_instance, x_test)

    # 4. Release memory
    del estimator_instance
    gc.collect()

    return pred_train, pred_val, pred_test


class LazyML:
    def __init__(self, hopt=True, output_folder=None, verbose=True):
        self.hopt = hopt
        self.output_folder = output_folder
        self.verbose = verbose

        if self.output_folder:
            if os.path.exists(self.output_folder):
                shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder)
        else:
            raise ValueError("output_folder must be specified.")

    def run(self, df_train, df_val, df_test):

        # 1. Get data (text and label). First column = text (title/abstract),
        # second column = label (e.g. HAS_ROUTE), matching the SMILES/property
        # convention of the original.
        result_df_train = pd.DataFrame()
        text_train, y_train = list(df_train.iloc[:, 0]), list(df_train.iloc[:, 1])
        result_df_train["TEXT"], result_df_train["Y_TRUE"] = text_train, y_train

        result_df_val = pd.DataFrame()
        text_val, y_val = list(df_val.iloc[:, 0]), list(df_val.iloc[:, 1])
        result_df_val["TEXT"], result_df_val["Y_TRUE"] = text_val, y_val

        result_df_test = pd.DataFrame()
        text_test, y_test = list(df_test.iloc[:, 0]), list(df_test.iloc[:, 1])
        result_df_test["TEXT"], result_df_test["Y_TRUE"] = text_test, y_test

        # This lazy.py variant only supports binary classification
        # (route vs. no-route) — no REGRESSORS branch, unlike the molecular version.
        task_type = type_of_target(y_train)
        if task_type != "binary":
            raise ValueError(
                "Task type not supported. This lazy.py variant only supports binary classification."
            )
        estimators_dict = CLASSIFIERS

        total_models = len(VECTORIZERS) * len(estimators_dict)
        current_model = 0

        # 2. Vectorize text (fit on train, transform val/test)
        for vec_name, vec_factory in VECTORIZERS.items():

            vectorizer = vec_factory()
            x_train, x_val, x_test = vectorize_text(text_train, text_val, text_test, vectorizer)

            # 3. Train models
            for est_name, estimator in estimators_dict.items():

                model_name = f"{vec_name}|{est_name}"
                current_model += 1
                if self.verbose:
                    print(f"[{current_model}/{total_models}] Running model: {model_name}", flush=True)

                start = time.time()
                pred_train, pred_val, pred_test = build_model(
                    x_train,
                    x_val,
                    x_test,
                    y_train,
                    y_val,
                    y_test,
                    estimator,
                    self.hopt
                )
                elapsed_min = (time.time() - start) / 60

                # 4. Write predictions
                result_df_train[model_name] = pred_train
                result_df_train.to_csv(os.path.join(self.output_folder, "train.csv"), index=False)

                result_df_val[model_name] = pred_val
                result_df_val.to_csv(os.path.join(self.output_folder, "val.csv"), index=False)

                result_df_test[model_name] = pred_test
                result_df_test.to_csv(os.path.join(self.output_folder, "test.csv"), index=False)

                if self.verbose:
                    process = psutil.Process()
                    mem_gb = process.memory_info().rss / (1024 ** 3)
                    print(f"  > Finished in {elapsed_min:.2f} min | Memory usage: {mem_gb:.3f} GB")

        return None