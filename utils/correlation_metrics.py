import pandas as pd
import torch
from typing import Literal
import numpy as np
from scipy import stats as stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def corr_or_mut(
        df: pd.DataFrame,
        col_name: str,
        kw_idx: int|None,
        wer_col_or_kw_col: Literal["wer_human_kw" , "human_transcript_kw", "machine_transcript_kw_est", "machine_transcript_kw_from_time_align", "human_transcript_kw"],
        special_metric: Literal["spearman_correlation", "mutual_information"],
) -> dict:
    microscopic = kw_idx is not None
    if microscopic:
        assert wer_col_or_kw_col != "wer_human_kw" # wer_col_or_kw_col is a columns carrying the keywords
        assert wer_col_or_kw_col in list(df), f"Column '{wer_col_or_kw_col}' not in dataframe!"
        assert "reference_kw" in list(df), f"Column 'reference_kw' not in dataframe!"
        assert 0 <= kw_idx <= 2

        ref_kw = df["reference_kw"].map(lambda x: x[kw_idx])
        keywords = df[wer_col_or_kw_col].map(lambda x: x[kw_idx])
        y = (ref_kw != keywords).astype(int)  # basically the WER

        x = df[col_name].map(lambda x: x[kw_idx])
    else:
        x = df[col_name]
        y = df["wer_human_kw"]
    n1 = len(y)
    filter = x.isna()
    x = torch.from_numpy(np.array(x.astype(float))[~filter])
    y = y[~filter]
    n2=len(y)
    if special_metric == "spearman_correlation":

        x_ranked = stats.rankdata(x)
        y_ranked = stats.rankdata(y)
        del y, x

        # spearman corr == pearson corr of ranks
        regr = stats.pearsonr(x=x_ranked, y=y_ranked)

        return {"value": regr.statistic, "p-value": regr.pvalue, "n": n2/n1}

    elif special_metric == "mutual_information":
        if microscopic:
            mi = mutual_info_classif(X=torch.Tensor(x).reshape(-1, 1), y=y)
        else:
            mi = mutual_info_regression(X=torch.Tensor(x).reshape(-1, 1), y=y)
        return {"value": mi[0], "n": n2/n1}
    else:
        raise Exception(f"Unknown metric: {special_metric}")
