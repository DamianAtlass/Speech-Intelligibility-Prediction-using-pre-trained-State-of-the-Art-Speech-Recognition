from utils.evaluate_utils import get_only_keywords, remove_nan
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
import pytest
import pandas as pd
from pathlib import Path

test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()

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


