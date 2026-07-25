from utils.config_dataclasses import InferenceConfig
from utils.parakeet_utils import load_parakeet_model
from pathlib import Path
test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from utils.cuda_utils import select_device
from dotenv import load_dotenv
from torch.utils.data import DataLoader
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch

def test_load_parakeet_model():
    config = InferenceConfig(
        model="parakeet",
        model_type="ctc-0.6b",
        model_path=None,
        output_path=test_folder / "inference_test",
        dataset_type="grid",
        dataset_path=test_folder.parent / "datasets/grid/",
        train_split=0,
        test_split=0,
        val_split=1,
        extract_logprobs=False,
        word_timestamps=False,
        beam_size=2
    )
    device = select_device()
    model = load_parakeet_model(config, device)
    assert isinstance(model, EncDecCTCModelBPE)