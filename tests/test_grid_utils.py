import pytest
from utils.grid_utils import get_sentence_and_alignments, parse_and_save_grid, download_grid, convert_short_name_to_ref
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