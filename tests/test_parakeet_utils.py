from utils.new_config_dataclass import InferenceConfig, DatasetConfig, ModelConfig, DataSplitConfig
from utils.parakeet_utils import load_parakeet_model
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
from utils.paths import TEST_FOLDER

def test_load_parakeet_model():


    config = InferenceConfig(
        output_path=TEST_FOLDER,
        task_type='inference',
        data=DatasetConfig(
            val_split=DataSplitConfig(dataset_type='grid', path=None, start=0, end=1, noise=False, scaling=1)),
        debug=False,
        extract_logprobs=False,
        word_timestamps=False,  # has no effect here
        beam_size=2,
        model=ModelConfig(name="parakeet", model_type="ctc-0.6b", path=None),
    )
    #dont load cudo here with select_device function. GPU could still be a bit busy
    model = load_parakeet_model(config, torch.device("cpu"))
    assert isinstance(model, EncDecCTCModelBPE)