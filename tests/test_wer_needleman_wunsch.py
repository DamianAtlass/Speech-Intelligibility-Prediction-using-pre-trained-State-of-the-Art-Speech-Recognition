from utils.wer_needleman_wunsch import wer_needleman_wunsch, wer_needleman_wunsch_per_sample
import pytest

@pytest.mark.parametrize(("references", "transcripts", "result"),[
    [["test"], ["text"], 1],
    [["test", "X"], ["test", "X"], 0],
    [["test", "X"], ["test", "Y"], 0.5],
    [["test"], ["test"], 0],
    [["test bingo"], ["test"], 0.5],
    [["X"], ["XX"], 1],
    [["this is an example sentence"], ["this is an example sentence"], 0],
    [["this is an example sentence"], ["this is an example TEST"], 0.2],
    [["this is an example sentence"], ["this is an sentence"], 0.2],
    [["this is an example sentence"], ["is an sentence"], 0.4],
    [["this is an example sentence"], ["this an sentence"], 0.4],
])

def test_wer_needleman_wunsch(references: list, transcripts: list, result: float | int):
    r = wer_needleman_wunsch(references=references, transcripts=transcripts)
    assert result == r

@pytest.mark.parametrize(("references", "transcripts", "result"),[
    [["test"], ["text"], [1]],
    [["test", "X"], ["test", "X"], [0, 0]],
    [["test", "X"], ["test", "Y"], [0, 1]],
])
def test_wer_needleman_wunsch_per_sample(references: list, transcripts: list, result: float | int):
    r = wer_needleman_wunsch_per_sample(references=references, transcripts=transcripts)
    assert result == r