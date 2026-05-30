from utils.werpy_utils import numbers_to_words, separate_numbers_from_letter, additional_normalization
import pytest

@pytest.mark.parametrize(("text", "result"),[
    ["2 Fast 2 Furious","two Fast two Furious"],
    ["22", "twenty-two"],
    ["0", "zero"]])
def test_numbers_to_words(text:str, result: str):
    assert numbers_to_words(text) == result

@pytest.mark.parametrize(("text", "result"),[
    ["tree4wood10","tree 4 wood 10"],
    ["sdfa-.,`2**##123sdf", "sdfa-.,` 2 **## 123 sdf"]])
def test_separate_numbers_from_letter(text, result: str):
    assert separate_numbers_from_letter(text) == result

def test_additional_normalization():
    assert additional_normalization(["as42_a!!5"])[0] == "as forty-two _a!! five"