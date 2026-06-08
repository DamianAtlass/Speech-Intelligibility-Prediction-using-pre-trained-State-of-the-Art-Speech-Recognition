from utils.evaluate_utils import get_only_keywords, remove_nan, calc_pearson_corr, calc_spearman_corr
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
import pytest


@pytest.mark.parametrize(("string", "output"), [
    ("one two three four five six", "two four five"),
    ("one two three four", "two four"),
    ("place red with j three again", "red j three"),
    ("one", ""),
    ("", ""),
])
def test_get_only_keywords(string, output):
    assert get_only_keywords(string) == output

@pytest.mark.parametrize(("x", "y", "x_exp", "y_exp"), [
    (torch.tensor([1, 2, 3, torch.nan]), torch.tensor([torch.nan, 2, 3, 4]), torch.tensor([2, 3]), torch.tensor([2, 3])),
        (torch.tensor([1, torch.nan]), torch.tensor([torch.nan, 2]), torch.tensor([]), torch.tensor([])),
    (torch.tensor([torch.nan, 2, 3, 4, 5]), torch.tensor([1, 9, 9, 9, 9]), torch.tensor([2, 3, 4, 5]), torch.tensor([9, 9, 9, 9])),

])
def test_remove_nan(x, y, x_exp, y_exp):
    x_out ,y_out = remove_nan(x,y)
    torch.equal(x_out, x_exp)
    assert torch.equal(y_out, y_exp)




def test_calc_pearson_corr():
    x = torch.linspace(0, 99, 500)

    def func(x: float) -> float:
        return x * 2  # linear

    noise = torch.randn(len(x)) * 15
    y = torch.Tensor(list(map(func, x)))
    y = y + noise

    rvalue, pvalue, _, _ = calc_pearson_corr(x, y, name="bla bla bla", xlabel="label", ylabel="label")
    assert rvalue > 0.9
    assert pvalue < 0.05

def test_calc_spearman_corr():
    x = torch.linspace(0, 99, 500)

    def func(x: float) -> float:
        return x**4 #exponential

    noise = torch.randn(len(x)) * 15
    y = torch.Tensor(list(map(func, x)))
    y = y + noise

    rvalue, pvalue = calc_spearman_corr(x, y, name="bla bla bla", xlabel="label", ylabel="label")
    assert rvalue > 0.9
    assert pvalue < 0.05