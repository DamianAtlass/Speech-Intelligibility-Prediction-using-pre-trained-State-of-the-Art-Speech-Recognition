import pytest
from utils.grid_utils import get_grid, apply_split, get_sentence_and_alignments, parse_and_save_grid, download_grid
from utils.config_dataclasses import Config
from pathlib import Path
import shutil

downloaded_grid_files = Path.cwd() / "tests" / "grid_downloaded"

def test_download_grid():
    # should take only a couple of seconds with good internet connection

    download_grid(downloaded_grid_files, max_speaker=2)
    assert downloaded_grid_files.is_dir()

def test_parse_and_save_grid():
    #needs output from test above
    save_at = Path.cwd()/"tests"/"grid_parsed"
    if not downloaded_grid_files.exists():
        pytest.skip()
    parse_and_save_grid(grid_folder=downloaded_grid_files,
                        max_speaker=1,
                        max_files_per_speaker=1,
                        save_at=save_at
                        )
    assert save_at.is_dir()
    shutil.rmtree(save_at)

@pytest.mark.parametrize(("split", "resulting_size"), [
        ((0.7, 0.2, 0.1, 1), (23_800, 6800, 3400)),
        ((0.5, 0.35, 0.15, 1), (17000, 11900, 5100)),
        ((0.5, 0.1, 0.1, 1), (17000, 3400, 3400)),
        ((1, 2, 3, 1), (1, 2, 3)),
        ((0.5, 0.1, 0.1, 0.5), (8500, 1700, 1700)),
])

def test_apply_split(split: tuple, resulting_size: tuple):
    dataset = get_grid() # len == 34,000

    dataset_dict = apply_split(dataset, *split)

    assert len(dataset_dict["train"]) == resulting_size[0]
    assert len(dataset_dict["test"]) == resulting_size[1]
    assert len(dataset_dict["val"]) == resulting_size[2]


@pytest.mark.parametrize(("scale", "resulting_size"), [
    (1, 34000),
    (0.5, 17000)
])
def test_apply_split_for_full_val_split(scale: int | float, resulting_size: int):

    dataset = get_grid() # len == 34,000
    config = Config(model="",
                    model_type="",
                    model_path=Path(""),
                    output_path=Path("tests/inference_test"),
                    dataset_type="grid",
                    dataset_path=Path("datasets/grid/"),
                    train_split=0,
                    test_split=0,
                    val_split=1,
                    dataset_scaling=scale)

    dataset_dict = apply_split(dataset, config.train_split, config.test_split, config.val_split, config.dataset_scaling)
    assert len(dataset_dict["val"]) == resulting_size


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