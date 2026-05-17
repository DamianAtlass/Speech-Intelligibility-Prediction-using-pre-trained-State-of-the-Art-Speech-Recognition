import pytest
import os
import whisper

def test_whisper_module():
    model = whisper.load_model("tiny")
    audio = whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = whisper.pad_or_trim(audio)

    whisper.transcribe(model, audio, word_timestamps=False, language="en")


def test_whisper_module_with_timestamps():
    model = whisper.load_model("tiny")
    audio = whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = whisper.pad_or_trim(audio)

    whisper.transcribe(model, audio, word_timestamps=True, language="en")