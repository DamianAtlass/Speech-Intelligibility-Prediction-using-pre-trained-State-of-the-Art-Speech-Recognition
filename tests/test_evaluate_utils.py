from utils.evaluate_utils import get_only_keywords_by_index, remove_nan, get_only_keywords_using_alignments, \
    find_ordered_indices, get_only_keywords_by_identity, get_only_keywords_by_phonetic_similarity, get_only_keywords_by_accepting_other_options
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
import pytest
from pathlib import Path

@pytest.mark.parametrize(("transcript", "keywords_to_find", "outcome"), [
    ("a b c d", "b c", [1,2]),
    ("a b c d", "a b c d", [0, 1, 2, 3]),
    ("a b c d", "", []),
    ("a b b b c", "b b", [1,2]),
    ("b b b b", "b b b", [0, 1, 2]),
    ("b b b b", ["b", None, "b"], [0, None, 2]),
    ("b b b b", [None, "b", "b"], [None, 1, 2]),
    ("b b b b", ["b", "b", None], [0, 1, None]),
    ("b b b b", [None, "b", None], [None, 1, None]),
    ("b b b b", [None, None, None], [None, None, None]),
    ("b b b b", [None, None, None], [None, None, None]),
])
def test_find_ordered_indices(transcript, keywords_to_find, outcome):
    keywords_to_find = keywords_to_find.split() if isinstance(keywords_to_find, str) else keywords_to_find

    r = find_ordered_indices(transcript=transcript.split(), keywords_to_find=keywords_to_find)
    assert r == outcome

@pytest.mark.parametrize(("transcript", "keywords_to_find"), [
    ("a b c d", "a b c d E")
])
def test_find_ordered_indices_throw_exception(transcript, keywords_to_find):
    keywords_to_find = keywords_to_find.split() if isinstance(keywords_to_find, str) else keywords_to_find
    try:
        r = find_ordered_indices(transcript=transcript.split(), keywords_to_find=keywords_to_find)
        assert False
    except ValueError:
        assert True



@pytest.mark.parametrize(("string", "output", "exception"), [
    ("one two three four five six", "two four five", False),
    ("one two three four", "two four", True),
    ("place red with j three again", "red j three", False),
])
def test_get_only_keywords(string, output, exception):
    try:
        assert get_only_keywords_by_index(string) == output
        assert not exception
    except ValueError:
        assert exception

@pytest.mark.parametrize(("transcript", "reference_kw", "result"), [
    ("one two three four five six", "two four five", "two four five"),
    ("one two three four five six", "A B C", ""),
    ("one two three four", "two X four", "two four"),
    #("place red with j three again", "red j three", False),
])
def test_get_only_keywords_by_identity(transcript, reference_kw, result: str):
     r = get_only_keywords_by_identity(reference_kw=reference_kw.split(), transcript=transcript.split())
     assert result == " ".join(r)


@pytest.mark.parametrize(("transcript", "reference_kw", "result"), [
    ("one two three four five six", "two four five", [1,3,4]),
    ("one two three four five six", "A B C", [None, None, None]),
    ("one two three four", "two X four", [1, None, 3]),
    ("place red with j three again", "red j three", [1,3,4]),
    ("place red with j four again", "red j three", [1, 3, None]),
])
def test_get_only_keywords_by_identity_with_return_idx(transcript, reference_kw, result: str):
    r = get_only_keywords_by_identity(reference_kw=reference_kw.split(), transcript=transcript.split(), return_idx=True)
    assert result == r


@pytest.mark.parametrize(("reference", "string", "output"), [
    ("1 2 3 4 5 6", "2 4 5", ["2", "4", "5"]),
    ("1 2 3 4 5 6", "", [None, None, None]),
    ("11 22 33 44 55 66", "22 44 55", ["22", "44", "55"]),
    ("this is not a cool riddle", "is not a cool riddle", ["is", "a", "cool"]),
    ("so mean of you to just", "so mean of", ["mean", None, None]),
    ("so mean of you to just", "so mean of you", ["mean", "you", None]),
    ("1 2 3 4 5 6", "2 X 5", ["2", "X", "5"]),
    ('bin blue at r seven again', "its been blue its r seven again", ["blue", "r", "seven"]),
])
def test_get_only_keywords_using_alignments(reference: str, string: str, output: str):

    keywords = get_only_keywords_using_alignments(reference.split(), string.split())
    assert keywords == output

@pytest.mark.parametrize(("reference", "transcript", "output"), [
    ("1 2 3 4 5 6", "a a a a a a 1 2 3 4 5 6", [7, 9, 10]),
    ("1 2 3 4 5 6", "2 4 5", [0, 1, 2]),
    ("1 2 3 4 5 6", [], [None, None, None]),
    ("A C A B B B", "C A B B B", [0, 2, 3]),
    ("A B C D E F", "A B", [1, None, None]),
    ("A B C D E F", "B B", [0, None, None]),
    ("B B C D E F", "B B D E", [0, 2, 3]),
    ("1 2 3 4 5 6", "7 8 9 10 11 12", [1, 3, 4]),
])
def test_get_only_keywords_using_alignments_with_return_idx(reference: str, transcript: str, output: str):
    transcript: list = transcript.split() if isinstance(transcript, str) else transcript

    keywords = get_only_keywords_using_alignments(reference=reference.split(), transcript=transcript, return_idx=True)
    assert keywords == output

@pytest.mark.parametrize(("transcript", "reference_kw", "expected_output"), [
    ("one two three four five six", "two four five", ["two", "four", "five"]),
    ("one to three for five six", "two four five", ["to", "for", "five"]),
    ("one transformers three for five six", "two four five", [None, "for", "five"]),
])
def test_get_only_keywords_by_phonetic_similarity(transcript, reference_kw, expected_output):
    r = get_only_keywords_by_phonetic_similarity(reference_kw=reference_kw.split(), transcript=transcript.split())
    assert expected_output == r

@pytest.mark.parametrize(("transcript", "reference_kw", "expected_output"), [
    ("one transformers three for five six", "two four five", [2, 3, 4]),
])
def test_get_only_keywords_by_phonetic_similarity(transcript, reference_kw, expected_output):
    r = get_only_keywords_by_phonetic_similarity(reference_kw=reference_kw.split(), transcript=transcript.split(), return_idx=True)
    assert expected_output == r

@pytest.mark.parametrize(("transcript", "reference_kw", "expected_output"), [
    ("bin green at x eight now", "green x eight", ["green", "x", "eight"]),
    ("bin green at c eight now", "green x eight", ["green", "c", "eight"]),
    ("bin white at BOX BOX now", "green x eight", ["white", None, None]),
])
def test_get_only_keywords_by_accepting_other_options(transcript, reference_kw, expected_output):
    r = get_only_keywords_by_accepting_other_options(reference_kw=reference_kw.split(), transcript=transcript.split())
    assert expected_output == r

@pytest.mark.parametrize(("transcript", "reference_kw", "expected_output"), [
    ("bin green at c BOX now", "green x eight", [1, 3, None]),
])
def test_get_only_keywords_by_accepting_other_options_with_return_index(transcript, reference_kw, expected_output):
    r = get_only_keywords_by_accepting_other_options(reference_kw=reference_kw.split(), transcript=transcript.split(), return_idx=True)
    assert expected_output == r

@pytest.mark.parametrize(("x", "y", "x_exp", "y_exp"), [
    (torch.tensor([1, 2, 3, torch.nan]), torch.tensor([torch.nan, 2, 3, 4]), torch.tensor([2, 3]), torch.tensor([2, 3])),
        (torch.tensor([1, torch.nan]), torch.tensor([torch.nan, 2]), torch.tensor([]), torch.tensor([])),
    (torch.tensor([torch.nan, 2, 3, 4, 5]), torch.tensor([1, 9, 9, 9, 9]), torch.tensor([2, 3, 4, 5]), torch.tensor([9, 9, 9, 9])),
])
def test_remove_nan(x, y, x_exp, y_exp):
    x_out ,y_out = remove_nan(x,y)
    torch.equal(x_out, x_exp)
    assert torch.equal(y_out, y_exp)


