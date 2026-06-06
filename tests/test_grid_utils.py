import pytest
from utils.grid_utils import get_grid, apply_split, get_sentence_and_alignments, parse_and_save_grid, download_grid, convert_short_name_to_ref
from utils.config_dataclasses import Config
from pathlib import Path
import shutil

test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()
grid_folder = test_folder / "grid"
real_grid_folder = test_folder.parent / "datasets" / "grid"

def test_download_grid():
    if grid_folder.exists():
        shutil.rmtree(grid_folder)
    # should take only a couple of seconds with good internet connection

    download_grid(grid_folder, max_speaker=2)
    assert (grid_folder/"downloaded_grid_files").is_dir()

def test_parse_and_save_grid():
    #needs output from test above

    if not grid_folder.exists():
        pytest.skip()
    dataset = parse_and_save_grid(grid_folder=grid_folder,
                        max_speaker=2,
                        max_files_per_speaker=4,
                        )
    assert len(dataset) == 2 * 4
    assert (grid_folder/"saved_dataset").is_dir()
    shutil.rmtree(grid_folder/"saved_dataset")

@pytest.mark.parametrize(("split", "resulting_size"), [
        ((0.7, 0.2, 0.1, 1), (23_800, 6800, 3400)),
        ((0.5, 0.35, 0.15, 1), (17000, 11900, 5100)),
        ((0.5, 0.1, 0.1, 1), (17000, 3400, 3400)),
        ((1, 2, 3, 1), (1, 2, 3)),
        ((0.5, 0.1, 0.1, 0.5), (8500, 1700, 1700)),
])
def test_apply_split(split: tuple, resulting_size: tuple):
    dataset = get_grid(real_grid_folder) # len == 34,000

    dataset_dict = apply_split(dataset, *split)

    assert len(dataset_dict["train"]) == resulting_size[0]
    assert len(dataset_dict["test"]) == resulting_size[1]
    assert len(dataset_dict["val"]) == resulting_size[2]


@pytest.mark.parametrize(("scale", "resulting_size"), [
    (1, 34000),
    (0.5, 17000)
])
def test_apply_split_for_full_val_split(scale: int | float, resulting_size: int):
    dataset = get_grid(real_grid_folder) # len == 34,000
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


@pytest.mark.parametrize(("file_path", "params", "start_expected"),[
    (grid_folder / "downloaded_grid_files/align/s1/align/bbaf2n.align", (25_000, 16_000), (0, 15200, 18880, 21760, 22720, 26240, 30240, 33920)), #25kHz
    (test_folder.parent / "datasets/GridIntelligibilityDatabase/word16kHz/s1/bbaf2n.align", (16_000, 16_000), (0, 15200, 18880, 21760, 22720, 26240, 30240, 33920)),  # 16kHz
])
def test_get_sentence_and_alignments(file_path, params, start_expected):
    sentence, alignment = get_sentence_and_alignments(file_path, *params)

    assert isinstance(sentence, str)
    assert len(sentence.split(" ")) == 6
    assert isinstance(alignment, list)
    for row in alignment:
        assert isinstance(row, tuple)
        assert len(row) == 3
    start = [s for (s, _,_ ) in alignment]

    for a,b in zip(start, start_expected):
        assert int(a) == b

@pytest.mark.parametrize(("file_name", "only_keywords", "expected_result"), [
    ("sraf4p" , False, "set red at f four please"),
    ("srbz8n", False, "set red by z eight now"),
    ("rz8", True, "red z eight"),
    ("be9", True, "blue e nine"),
])
def test_convert_short_name_to_ref(file_name, only_keywords, expected_result):
    assert convert_short_name_to_ref(file_name, input_is_only_keywords=only_keywords) == expected_result