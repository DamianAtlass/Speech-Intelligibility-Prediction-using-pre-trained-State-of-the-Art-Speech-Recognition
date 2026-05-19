import pytest
import os
import whisper
from dotenv import load_dotenv
from utils.cuda_utils import select_gpu

def test_cuda_with_whisper_module():
    load_dotenv()
    device = select_gpu(int(os.getenv("GPU_DEVICE")))

    model = whisper.load_model("tiny", device=device)

    audio = whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = whisper.pad_or_trim(audio)

    whisper.transcribe(model, audio, word_timestamps=True, language="en")

    assert True