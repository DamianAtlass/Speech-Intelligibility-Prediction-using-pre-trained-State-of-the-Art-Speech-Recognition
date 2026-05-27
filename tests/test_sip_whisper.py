import pytest
import whisper
import sip_whisper
import torch
@pytest.mark.parametrize("time_stamps", [False,True])
def test_sip_whisper_module(time_stamps):
    model = sip_whisper.load_model("tiny", device="cpu")
    audio = sip_whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    result = sip_whisper.transcribe(model,
                       audio,
                       fp16=False,
                       beam_size=2,
                       temperature=0,
                       word_timestamps=time_stamps,
                       condition_on_previous_text=False)

    result["extracted_logprobs"]
    print()

def test_mixing_functions():
    model = sip_whisper.load_model("tiny", device="cpu")
    audio = sip_whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio = sip_whisper.pad_or_trim(audio)

    result_1 = sip_whisper.transcribe(model,
                                    audio,
                                    fp16=False,
                                    beam_size=2,
                                    temperature=0,
                                    word_timestamps=True,
                                    condition_on_previous_text=False)
    tensors_1 = result_1.pop("extracted_logprobs")


    #use regular whisper functions
    audio_2 = whisper.load_audio("tests/sample_audio_small.mp3", 16_000)
    audio_2 = whisper.pad_or_trim(audio)

    result_2 = sip_whisper.transcribe(model,
                                    audio_2,
                                    fp16=False,
                                    beam_size=2,
                                    temperature=0,
                                    word_timestamps=True,
                                    condition_on_previous_text=False)
    tensors_2 = result_2.pop("extracted_logprobs")

    assert result_1 == result_2
    assert torch.equal(tensors_1, tensors_2)
    print()