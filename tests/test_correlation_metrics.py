import pandas as pd
import torch
from utils.correlation_metrics import corr_or_mut
import pytest

def test_corr_or_mut():
    df = pd.DataFrame({
        "x": torch.linspace(0, 1, 100)+torch.rand(100),
        "wer_human_kw": torch.linspace(0, 1, 100)
    })

    r = corr_or_mut(
        df=df,
        wer_col_or_kw_col="y",
        col_name="x",
        special_metric="spearman_correlation",
        kw_idx=None
    )
    assert r["value"].item() is not None
    assert r["p-value"].item() is not None