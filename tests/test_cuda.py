import pytest
import whisper
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch
from utils.cuda_utils import get_gpu_index, select_device
import os

def test_get_gpu_index():
    if not torch.cuda.is_available():
        pytest.skip("Cuda not available for testing.")
    gpu_index = get_gpu_index()

    assert isinstance(gpu_index, int)

def test_select_device():
    device: torch.device = select_device()

    if os.environ["DEVICE_HAS_USABLE_GPU"]=="True":
        assert device.type == "cuda"

    else:
        assert device.type == "cpu"


def test_cuda_with_whisper_module():
    if not torch.cuda.is_available():
        pytest.skip("Cuda not available for testing.")

    model = whisper.load_model("tiny", device=select_device())

    audio = whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = whisper.pad_or_trim(audio)

    whisper.transcribe(model, audio, word_timestamps=True, language="en")

    assert True