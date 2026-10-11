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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score,roc_curve, roc_auc_score

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

    metric_col = "entropy_kw_from_time_align"
    kw_col = "machine_transcript_kw_from_time_align"

    curr_kw_idx = 0
    df[metric_col] = df[metric_col].apply(lambda x: x[curr_kw_idx])
    df[kw_col] = df[kw_col].apply(lambda x: x[curr_kw_idx])
    human_kw = df["human_transcript_kw"].apply(lambda x: x[curr_kw_idx]).values
    ref_kw = df["reference_kw"].apply(lambda x: x[curr_kw_idx]).values

    x = df[metric_col]
    y = human_kw == ref_kw

    filter = x.isna()
    x = x[~filter]
    y = y[~filter]
    #https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=29
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.to_frame())
    X_test = scaler.transform(X_test.to_frame())

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy: {round(accuracy, 2)}")


if __name__ == '__main__':
    regr(Path("inferences/delete_me"), restrict_vocab=False)