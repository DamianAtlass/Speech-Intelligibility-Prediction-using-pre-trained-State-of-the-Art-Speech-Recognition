from pathlib import Path
import os
import pytest
from utils.paths import TEST_FOLDER
from utils.config_dataclasses import get_config, unfold_config, to_int_or_float

CONFIG_TEST_FOLDER = TEST_FOLDER/"config_dataclasses"

@pytest.mark.parametrize("file_name", [
        "test_inference_config.ini",
        "test_group_inference_config.ini",
        "test_training_config.ini",
        "test_training_config2.ini",
])
def test_loading_and_saving(file_name):
    file_path = CONFIG_TEST_FOLDER/file_name
    tmp_file = CONFIG_TEST_FOLDER/"tmp.ini"

    if tmp_file.exists():
        os.remove(tmp_file)

    config = get_config(file_path)

    config.save_to_file(tmp_file)

    new_config = get_config(tmp_file)

    assert config == new_config

    if tmp_file.exists():
        os.remove(tmp_file)


def test_loading_bad_file(file_name = "test_training_config_bad.ini"):

    file_path = CONFIG_TEST_FOLDER/file_name
    try:
        get_config(file_path)
    except ValueError:
        assert True


@pytest.mark.parametrize(("file_name", "length"), [
        ("test_inference_config.ini", 1),
        ("test_group_inference_config.ini", 3),
        ("test_group_inference_config2.ini", 9),
        ("test_training_config.ini", 1),
#        "test_group_training_config.ini",
])
def test_unfold_config(file_name, length):
    file_path = CONFIG_TEST_FOLDER / file_name
    config = get_config(file_path)

    configs = unfold_config(config)

    assert len(configs) == length

@pytest.mark.parametrize(["num_string", "type_"],
                         [("6", int),
                          ("2.1", float)])
def test_to_int_or_float(num_string, type_):
    num = to_int_or_float(num_string)
    assert isinstance(num, type_)