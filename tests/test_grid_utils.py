import pytest
from utils.grid_utils import get_grid, apply_split, get_sentence_and_alignments
from utils.config_dataclasses import Config
from pathlib import Path

@pytest.mark.parametrize(("split", "resulting_size"), [
        ((0.7, 0.2, 0.1), (23_800, 6800, 3400)),
        ((0.5, 0.35, 0.15), (17000, 11900, 5100)),
        ((0.5, 0.1, 0.1), (17000, 3400, 3400)),
        ((1, 2, 3), (1, 2, 3)),
])
def test_apply_split(split: tuple, resulting_size: tuple):
    train_split, test_split, val_split = split

    dataset = get_grid() # len == 34,000
    config = Config(model="",
                    model_type="",
                    model_path=Path(""),
                    output_path=Path("tests/inference_test"),
                    dataset_path=Path("datasets/grid/"),
                    train_split=train_split,
                    test_split=test_split,
                    val_split=val_split)

    dataset_dict = apply_split(dataset, config)

    assert len(dataset_dict["train"]) == resulting_size[0]
    assert len(dataset_dict["test"]) == resulting_size[1]
    assert len(dataset_dict["val"]) == resulting_size[2]

@pytest.mark.parametrize("file_path",[
    Path("datasets/grid/downloaded_grid_files/align/s1/align/bbaf2n.align"),
    Path("datasets/grid/downloaded_grid_files/align/s1/align/bbaf3s.align"),
    Path("datasets/grid/downloaded_grid_files/align/s13/align/bbae1s.align"),
    Path("datasets/grid/downloaded_grid_files/align/s26/align/srwz9n.align"),
    Path("datasets/grid/downloaded_grid_files/align/s34/align/lwwi9p.align"),
    Path("datasets/grid/downloaded_grid_files/align/s4/align/sbim7p.align"),
])
def test_get_sentence_and_alignments(file_path):
    file_path = Path.cwd() / file_path
    sentence, alignment = get_sentence_and_alignments(file_path)

    assert isinstance(sentence, str)
    assert len(sentence.split(" ")) == 6
    assert isinstance(alignment, list)
    for row in alignment:
        assert isinstance(row, tuple)
        assert len(row) == 3