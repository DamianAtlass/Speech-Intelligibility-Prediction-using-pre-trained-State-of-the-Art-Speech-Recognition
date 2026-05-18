import os
import pytest
from utils.convert_hf_to_openai_format import convert_hf_model_to_openai_whisper
from pathlib import Path

def test_convert_hf_model_to_openai_whisper(checkpoint_path = Path("tests"), ):
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


