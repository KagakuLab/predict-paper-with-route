import pandas as pd
from sklearn.model_selection import train_test_split

from routehunter.ml.lazy import LazyML
from qsarcons.consensus import SystematicSearch, GeneticSearch


class ConsensusModel:

    def __init__(
        self,
        hopt: bool = True,
        output_folder=None,
        verbose: bool = True,
    ):
        self.output_folder = output_folder
        self.hopt = hopt
        self.verbose = verbose

        self.lazy_ml = LazyML(hopt=hopt, output_folder=output_folder, verbose=verbose)
        # self.cons_search = GeneticSearch(cons_size="auto", n_iter=50, verbose=verbose)
        self.cons_search = SystematicSearch(cons_size="auto")

        self.best_cons = None

    def run(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> "ConsensusModel":
        # 1. Add a placeholder label column if df_test has text only
        # (inference-only usage, no ground truth available yet).
        if len(df_test.columns) == 1:
            df_test = df_test.copy()
            df_test[df_test.shape[1]] = None

        # 2. Train/val split (val used for the genetic search)
        df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)

        # 3. Build every (vectorizer, classifier) model via LazyML
        self.lazy_ml.run(df_train, df_val, df_test)

        # 4. Load val predictions, run genetic search to find best consensus
        res_val = pd.read_csv(f"{self.output_folder}/val.csv")
        x_val, y_val = res_val.iloc[:, 2:], res_val.iloc[:, 1]

        self.best_cons = self.cons_search.run(x_val, y_val)
        if self.verbose:
            print(f"Systematic consensus: {self.best_cons}")

        return self

    def predict(self, df_test: pd.DataFrame) -> pd.DataFrame:
        if self.best_cons is None:
            raise RuntimeError("Call run() before predict().")

        res_test = pd.read_csv(f"{self.output_folder}/test.csv")
        x_test = res_test.iloc[:, 2:]

        df_test = df_test.copy()
        df_test["CONS_PRED"] = self.cons_search.predict(x_test[self.best_cons])
        return df_test

    def predict_proba(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Like predict(), but returns the consensus probability (mean of
        member-model probabilities) instead of a thresholded 0/1 label.
        Useful for routehunter's downstream triage, which will likely
        threshold on probability rather than a hard label.
        """
        if self.best_cons is None:
            raise RuntimeError("Call run() before predict_proba().")

        res_test = pd.read_csv(f"{self.output_folder}/test.csv")
        x_test = res_test.iloc[:, 2:]

        df_test = df_test.copy()
        df_test["CONS_PROBA"] = self.cons_search.predict_proba(x_test[self.best_cons])
        return df_test

    def run_predict(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
        self.run(df_train, df_test)
        return self.predict(df_test)




