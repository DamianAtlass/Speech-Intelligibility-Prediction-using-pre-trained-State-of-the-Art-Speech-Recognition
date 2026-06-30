import pandas as pd
import torch
from pathlib import Path
from utils.plotting_utils import plot_regr_line_for_pearson_corr, plot_regr_line_for_spearman_corr

test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()

def test_plot_regr_line_for_pearson_corr():
    x = torch.linspace(0, 99, 100)

    def func(x: float) -> float:
        return x * 2  # linear

    noise = torch.randn(len(x)) * 15
    y = torch.Tensor(list(map(func, x)))
    y = y + noise

    df = pd.DataFrame({
        "x": x,
        "y": y,
    })

    rvalue, pvalue, _, _ = plot_regr_line_for_pearson_corr(df, name="test pearson", xlabel="x label", ylabel="y label", output_path=test_folder)
    assert rvalue > 0.9
    assert pvalue < 0.05


def test_plot_regr_line_for_spearman_corr():
    x = torch.linspace(0, 99, 100)

    def func(x: float) -> float:
        return x**2 #exponential

    noise = torch.randn(len(x)) * 400
    y = torch.Tensor(list(map(func, x)))
    y = y + noise

    df = pd.DataFrame({
        "x": x,
        "y": y,
    })

    rvalue, pvalue = plot_regr_line_for_spearman_corr(df, name="test spearman", xlabel="x label", ylabel="y label", output_path=test_folder)
    assert rvalue > 0.9
    assert pvalue < 0.05
