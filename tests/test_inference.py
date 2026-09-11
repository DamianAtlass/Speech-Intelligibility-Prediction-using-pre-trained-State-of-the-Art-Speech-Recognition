from inference import inference
import pytest
from utils.new_config_dataclass import InferenceConfig, ModelConfig, DataSplitConfig, DatasetConfig
from utils.dataset_utils import get_dataset_dict
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch
from utils.cuda_utils import select_device
import shutil
from utils.paths import GRID_FOLDER, TEST_FOLDER

@pytest.mark.parametrize("time_stamps", [False,True])
@pytest.mark.parametrize("extract_logprobs", [False,True])
def test_batch_inference_whisper(time_stamps, extract_logprobs):

    config = InferenceConfig(
        output_path=TEST_FOLDER / "inference_test",
        task_type='inference',
        data=DatasetConfig(
            test_split=DataSplitConfig(dataset_type='grid', path=None, start=0, end=1, noise=False, scaling=1)),
        debug=False,
        extract_logprobs=extract_logprobs,
        word_timestamps=time_stamps,
        beam_size=2,
        model=ModelConfig(name="whisper", model_type="tiny", path=None),
    )
    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset_dict = get_dataset_dict(config.data)
    device = select_device()
    config.output_path.mkdir(exist_ok=config.debug)
    inference(config, dataset_dict, device)

    assert (TEST_FOLDER / "inference_test/data/s26_pwwizs.json").exists()

    if extract_logprobs:
        assert (TEST_FOLDER/"inference_test/logprobs/s26_pwwizs.pt").exists()

    torch.cuda.empty_cache()

def test_inference_with_multiple_runs():
    config = InferenceConfig(
        output_path=TEST_FOLDER / "inference_test",
        task_type='inference',
        data=DatasetConfig(
            test_split=DataSplitConfig(dataset_type='grid', path=None, start=0, end=2, noise=False, scaling=1)),
        debug=False,
        temperature=0,
        extract_logprobs=True,
        word_timestamps=False,
        beam_size=1,
        runs_per_sample=2,
        model=ModelConfig(name="whisper", model_type="tiny", path=None),
    )
    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset_dict = get_dataset_dict(config.data)
    device = select_device()
    config.output_path.mkdir(exist_ok=config.debug)
    inference(config, dataset_dict, device)

    assert sum([1 for f in (config.output_path/"data").iterdir() if f.is_file()]) == 4
    assert sum([1 for f in (config.output_path/"logprobs").iterdir() if f.is_file()]) == 4
    torch.cuda.empty_cache()

def test_inference_expected_exception():
    config = InferenceConfig(
        output_path=TEST_FOLDER / "inference_test",
        task_type='inference',
        data=DatasetConfig(
            test_split=DataSplitConfig(dataset_type='grid', path=None, start=0, end=2, noise=False, scaling=1)),
        debug=False,
        temperature=0.25,
        extract_logprobs=True,
        word_timestamps=False,
        beam_size=2,
        model=ModelConfig(name="whisper", model_type="tiny", path=None),
    ) #todo add run argument here
    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset_dict = get_dataset_dict(config.data)
    device = select_device()
    config.output_path.mkdir(exist_ok=config.debug)
    try:
        inference(config, dataset_dict, device)
        assert False
    except ValueError:
        assert True
    torch.cuda.empty_cache()

def test_forced_alignment_inference_whisper():
    config = InferenceConfig(
        output_path=TEST_FOLDER / "forced_alignment_test",
        task_type='inference',
        data=DatasetConfig(
            test_split=DataSplitConfig(dataset_type='grid', path=None, start=0, end=1, noise=False, scaling=1)),
        debug=False,
        extract_logprobs=False,
        word_timestamps=True,
        beam_size=5,
        model=ModelConfig(name="whisper", model_type="large-v3-turbo", path=None),
        forced_alignment=True,
    )
    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset_dict = get_dataset_dict(config.data)
    device = select_device()
    config.output_path.mkdir(exist_ok=config.debug)
    inference(config, dataset_dict, device)
    assert sum(1 for _ in (config.output_path/"data").iterdir()) == 37
    torch.cuda.empty_cache()

    if config.output_path.exists():
        shutil.rmtree(config.output_path)