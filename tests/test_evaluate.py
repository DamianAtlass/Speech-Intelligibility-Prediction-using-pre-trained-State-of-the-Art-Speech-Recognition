from evaluate_run import get_only_keywords
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