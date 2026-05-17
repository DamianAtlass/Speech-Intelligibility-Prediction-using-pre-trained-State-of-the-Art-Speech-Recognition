from pathlib import Path
import os
import pytest

from utils.config_dataclasses import get_config, unfold_config

TEST_FOLDER = Path.cwd()/"tests/config_dataclasses"

@pytest.mark.parametrize("file_name", [
        "test_inference_config.ini",
        "test_group_inference_config.ini",
        "test_training_config.ini",
#        "test_group_training_config.ini",
])
def test_loading_and_saving(file_name):
    file_path = TEST_FOLDER/file_name
    tmp_file = TEST_FOLDER/"tmp.ini"

    if tmp_file.exists():
        os.remove(tmp_file)

    config = get_config(file_path)

    config.save_to_file(tmp_file)

    new_config = get_config(tmp_file)

    assert config == new_config

    if tmp_file.exists():
        os.remove(tmp_file)

@pytest.mark.parametrize(("file_name", "l"), [
        ("test_inference_config.ini", 1),
        ("test_group_inference_config.ini", 3),
        ("test_training_config.ini", 1),
#        "test_group_training_config.ini",
])
def test_unfold_config(file_name, l):
    file_path = TEST_FOLDER / file_name
    config = get_config(file_path)

    configs = unfold_config(config)

    assert len(configs) == l