from utils.config_dataclasses import old_get_config, old_unfold_config, Old_InferenceConfig
from utils.new_config_dataclass import InferenceConfig, load_config, convert_old_config_into_new, save_config
from pathlib import Path
from utils.logging_utils import catch_time

from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
import logging
logger = logging.getLogger(__name__)
from utils.evaluate_utils import get_data, evaluate_individual_run, evaluate_forced_alignment_run

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

import json
def get_X_and_y(df: pd.DataFrame, metrics: list[str], kw_number: int | None = None):
    if kw_number is not None:
        if kw_number<0 or kw_number>2: raise ValueError("Must be between 0 and 2")

    X = df.copy()
    X = X[metrics]
    if kw_number is not None:
        for metric in metrics:
            X[metric] = X[metric].apply(lambda x: x[kw_number])



    human_kw = df["human_transcript_kw"].apply(lambda x: x[kw_number])
    ref_kw = df["reference_kw"].apply(lambda x: x[kw_number])
    y = human_kw == ref_kw

    filter = X.isna().any(axis=1)
    X = X[~filter]
    y = y[~filter]
    return X, y



def regr(path: Path, restrict_vocab: bool, device: torch.device | None = None):
    if not device:
        device = torch.device("cpu")


    config: InferenceConfig = load_config(path/"config.yaml")

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

    metrics_list = [
        ["entropy_kw_from_time_align"],
        ["laptev_kw_from_time_align"],
        ["tad_kw"],
        ["entropy_kw_from_time_align", "laptev_kw_from_time_align", "tad_kw"],
    ]

    n_splits = 5
    keywords = ["color", "letter", "digit"]
    results_list: list[dict] = []

    for metrics in metrics_list:
        metric_is_micro = "kw" in metrics[0]
        metric_results = {
            "micro/macro": "microscopic" if metric_is_micro else "macroscopic",
            "used metrics": metrics,
            "results": []
        }

        kw_numbers: list = [0, 1 ,2] if metric_is_micro else [None]
        for kw_number in kw_numbers:
            result = {
                "keyword number": keywords[kw_number] if metric_is_micro else "None (macroscopic)",
                "n samples": len(df)
            }
            X, y = get_X_and_y(
                df=df,
                metrics=metrics,
                kw_number=kw_number)

            model = LogisticRegression(max_iter=200)
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

            valid_rows = len(X)
            non_valid_rows = (len(df)-valid_rows)/len(df)
            result["non-valid rows"] = f"{non_valid_rows*100:.2f} %"
            result["mean score of valid rows"] = f"{np.mean(scores):.2f} %"
            result["adjusted score"] = f"{(np.mean(scores)*valid_rows)/len(df):.2f} %"

            metric_results["results"].append(result)
        results_list.append(metric_results)

    results_dir = "results_restrict_vocab" if restrict_vocab else "results_full_vocab"

    with open(path / results_dir / "cross_validation.json", 'w') as f:
        json.dump(results_list, f, indent=4)


if __name__ == '__main__':
    regr(Path("inferences/delete_me"), restrict_vocab=False)