import os
import pytest
from utils.whisper_utils import convert_hf_model_to_openai_whisper
from pathlib import Path
from utils.paths import TEST_FOLDER

def test_convert_hf_model_to_openai_whisper(checkpoint_path = TEST_FOLDER/"tiny_test_checkpoint", ):
    if not Path(checkpoint_path/"model.safetensors").is_file():
        print("Skip test. No model to import.")
        pytest.skip()
    save_file = "whisper_openai_format_test.pt"
    path_tmp = checkpoint_path/save_file
    if path_tmp.exists():
        os.remove(path_tmp)

    path = convert_hf_model_to_openai_whisper(hf_checkpoint_file_path=checkpoint_path,
                                              safe_file=save_file,
                                              model_type="tiny")

    assert path.is_file()
    os.remove(path)


