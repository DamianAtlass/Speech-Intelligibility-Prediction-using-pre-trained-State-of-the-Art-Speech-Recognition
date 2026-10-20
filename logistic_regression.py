from utils.new_config_dataclass import InferenceConfig, load_config

from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
from utils.plotting_utils import PredictorPlots

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
import json

import logging
logger = logging.getLogger(__name__)

def get_X_and_y(df: pd.DataFrame, metrics: list[str], kw_number: int | None = None):
    is_microscopic = kw_number is not None
    if is_microscopic:
        if kw_number<0 or kw_number>2: raise ValueError("Must be between 0 and 2")

    X = df.copy()
    X = X[metrics]
    if kw_number is not None:
        for metric in metrics:
            X[metric] = X[metric].apply(lambda x: x[kw_number])


    if is_microscopic:
        human_kw = df["human_transcript_kw"].apply(lambda x: x[kw_number])
        ref_kw = df["reference_kw"].apply(lambda x: x[kw_number])
        y = human_kw == ref_kw
    else:
        y = df["wer_human_kw"]

    filter = X.isna().any(axis=1)
    X = X[~filter]
    y = y[~filter]
    return X, y



def train_and_evaluate_predictors(
        df: pd.DataFrame,
        metrics_list: list[list[str]],
        output_path: Path,
) -> list:


    if metrics_list is None:
        metrics_list = [
            ["macroscopic_entropy"],
            ["mtd"],
            ["laptev"],
            ["macroscopic_entropy", "mtd", "laptev"],

            ["entropy_kw_est"],
            ["entropy_kw_from_time_align"],
            ["tad_kw"],
            ["entropy_kw_from_time_align", "tad_kw"],
        ]

    n_splits = 5
    keywords = ["color", "letter", "digit"]
    results_list: list[dict] = []

    for metrics in metrics_list:
        metric_is_micro = "kw" in metrics[0]
        metric_results = {
            "micro/macro": "microscopic" if metric_is_micro else "macroscopic",
            "used metrics": metrics,
            "n splits": n_splits,
            "results": [],
        }

        scoring_metric = 'accuracy' if metric_is_micro else "neg_mean_absolute_error"
        metric_results["scoring metric"] = scoring_metric
        error_fn = accuracy_score if metric_is_micro else mean_absolute_error

        kw_numbers: list = [0, 1 ,2] if metric_is_micro else [None]
        for kw_number in kw_numbers:
            result = {
                "keyword type": keywords[kw_number] if metric_is_micro else "None (macroscopic)",
                "n samples": len(df)
            }
            X, y = get_X_and_y(
                df=df,
                metrics=metrics,
                kw_number=kw_number)
            if metric_is_micro:
                model = LogisticRegression(max_iter=200)
            else:
                model = LinearRegression()

            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            scores = []
            weights = []
            intercepts = []

            for train_idx, test_idx in kfold.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model.fit(X_train, y_train)

                # Save model weights and intercept
                weights.append(model.coef_.tolist())
                intercepts.append(model.intercept_.tolist())

                # Evaluate on the test fold
                y_pred = model.predict(X_test)
                scores.append(abs(error_fn(y_test, y_pred)))

            valid_rows = len(X)
            non_valid_rows = (len(df)-valid_rows)/len(df)
            result["Weights per fold"] = weights
            result["Mean weights"] = np.mean(weights, axis=0).tolist()
            result["Mean intercept"] = np.mean(intercepts).tolist()
            result["non-valid rows (percent)"] = non_valid_rows*100
            result["mean score of valid rows"] = np.mean(scores).tolist()
            result["adjusted score"] = (np.mean(scores)*valid_rows)/len(df) if metric_is_micro else "???"

            metric_results["results"].append(result)
        results_list.append(metric_results)

    with open(output_path / "cross_validation.json", 'w') as f:
        json.dump(results_list, f, indent=4)
    return results_list




def build_predictors(df: pd.DataFrame,
                     metrics_list: list[list[str]],
                     output_path: Path):
    output_path = output_path / "predictors"
    output_path.mkdir(parents=True, exist_ok=True)
    results: list = train_and_evaluate_predictors(df=df, metrics_list=metrics_list, output_path=output_path)

    r = [o for o in results if o["micro/macro"] == "macroscopic"]
    PredictorPlots.plot_macroscopoic(r, output_path)

    r = [o for o in results if o["micro/macro"] == "microscopic"]
    PredictorPlots.plot_microscopic(r, output_path)


if __name__ == '__main__':
    pass
