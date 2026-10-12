from utils.new_config_dataclass import InferenceConfig, load_config
from utils.logging_utils import catch_time
from utils.evaluate_utils import get_data

from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
from utils.plotting_utils import PredictorPlots

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
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
        df: pd.DataFrame | None = None,
        config: InferenceConfig | None = None,
        restrict_vocab: bool | None = None,
        metrics_list: list[list[str]] = None,
        device: torch.device | None = None,) -> list:
    if not device:
        device = torch.device("cpu")

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


    if df is None:
        assert config is not None
        assert restrict_vocab is not None
        with catch_time() as t:
            df = get_data(
                model_name=config.model.name,
                output_path=config.output_path,
                dataset_type=config.data.test_split.dataset_type,
                extract_logprobs=config.extract_logprobs,
                word_timestamps=config.word_timestamps,
                restrict_vocab=restrict_vocab,
                device=device)
        print(f"Reading the generated files took: {t():.4f} s")

    df["model_type"] = config.model.model_type

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
            "scoring metric": None
        }

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
            scoring_metric = 'accuracy' if metric_is_micro else "neg_mean_absolute_error"
            result["scoring metric"] = scoring_metric
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring_metric)
            print(scores)
            valid_rows = len(X)
            non_valid_rows = (len(df)-valid_rows)/len(df)
            result["non-valid rows (percent)"] = non_valid_rows*100
            result["mean score of valid rows"] = np.mean(scores)
            result["adjusted score"] = (np.mean(scores)*valid_rows)/len(df) if metric_is_micro else "???"

            metric_results["results"].append(result)
        results_list.append(metric_results)

    results_dir = "results_restrict_vocab" if restrict_vocab else "results_full_vocab"

    with open(config.output_path / results_dir / "cross_validation.json", 'w') as f:
        json.dump(results_list, f, indent=4)
    return results_list




def build_predictors(
        df: pd.DataFrame | None = None,
        config: InferenceConfig | None = None,
        restrict_vocab: bool | None = None,
        metrics_list: list[list[str]] = None,
        device: torch.device | None = None,):

    results: list = train_and_evaluate_predictors(df=df, config=config, restrict_vocab=restrict_vocab, metrics_list=metrics_list, device=device)
    results_dir = "results_restrict_vocab" if restrict_vocab else "results_full_vocab"

    r = [o for o in results if o["micro/macro"] == "macroscopic"]
    PredictorPlots.plot_macroscopoic(r, config.output_path / results_dir)

    r = [o for o in results if o["micro/macro"] == "microscopic"]
    PredictorPlots.plot_microscopic(r, config.output_path / results_dir)


if __name__ == '__main__':
    #config: InferenceConfig = load_config(Path("inferences/turbo_exp2_bc") / "config.yaml")
    config: InferenceConfig = load_config(Path("inferences/delete_me") / "config.yaml")
    build_predictors(config = config, restrict_vocab=False)
