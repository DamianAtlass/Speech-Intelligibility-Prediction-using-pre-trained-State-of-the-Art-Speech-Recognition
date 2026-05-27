import pytest
import whisper

@pytest.mark.parametrize("time_stamps", [False,True])
def test_whisper_module(time_stamps):
    model = whisper.load_model("tiny", device="cpu")
    audio = whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = whisper.pad_or_trim(audio)

    whisper.transcribe(model,
                       audio,
                       fp16=False,
                       beam_size=2,
                       temperature=0,
                       word_timestamps=time_stamps,
                       condition_on_previous_text=False
                       )