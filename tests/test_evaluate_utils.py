from utils.evaluate_utils import get_only_keywords, remove_nan, get_only_keywords_using_alignments
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
import pytest
from pathlib import Path

test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()

@pytest.mark.parametrize(("string", "output", "exception"), [
    ("one two three four five six", "two four five", False),
    ("one two three four", "two four", True),
    ("place red with j three again", "red j three", False),
])
def test_get_only_keywords(string, output, exception):
    try:
        assert get_only_keywords(string) == output
        assert not exception
    except ValueError:
        assert exception


@pytest.mark.parametrize(("reference", "string", "output"), [
    ("1 2 3 4 5 6", "2 4 5", "2 4 5"),
    ("1 2 3 4 5 6", "", ""),
    ("11 22 33 44 55 66", "22 44 55", "22 44 55"),
    ("this is not a cool riddle", "is not a cool riddle", "is a cool"),
    ("so mean of you to just", "so mean of", "mean"),
    ("so mean of you to just", "so mean of you", "mean you"),
    ("1 2 3 4 5 6", "2 X 5", "2 X 5"),
    ('bin blue at r seven again', "its been blue its r seven again", "blue r seven"),
])
def test_get_only_keywords_using_alignments(reference: str, string: str, output: str):
    keywords = get_only_keywords_using_alignments(reference, string)
    assert keywords == output


@pytest.mark.parametrize(("x", "y", "x_exp", "y_exp"), [
    (torch.tensor([1, 2, 3, torch.nan]), torch.tensor([torch.nan, 2, 3, 4]), torch.tensor([2, 3]), torch.tensor([2, 3])),
        (torch.tensor([1, torch.nan]), torch.tensor([torch.nan, 2]), torch.tensor([]), torch.tensor([])),
    (torch.tensor([torch.nan, 2, 3, 4, 5]), torch.tensor([1, 9, 9, 9, 9]), torch.tensor([2, 3, 4, 5]), torch.tensor([9, 9, 9, 9])),

])
def test_remove_nan(x, y, x_exp, y_exp):
    x_out ,y_out = remove_nan(x,y)
    torch.equal(x_out, x_exp)
    assert torch.equal(y_out, y_exp)


