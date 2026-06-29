import whisper
import numpy as np

from utils.wer_needleman_wunsch import wer_needleman_wunsch
from utils.grid_utils import get_grid
from utils.grid_bc_utils import get_grid_bc
from utils.cuda_utils import select_device
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch

from utils.grid_utils import convert_short_name_to_ref


downloaded_grid_file_25kHz = "datasets/grid/downloaded_grid_files/audio/s23/s23/sraf4p.wav"
def foo():

    model = whisper.load_model("tiny")
    dataset = get_grid()



    audio = whisper.load_audio(downloaded_grid_file_25kHz)

    other_audio = whisper.load_audio(downloaded_grid_file_25kHz)
    import soundfile as sf
    sf.write("file1.wav", audio, 16_000)
    sf.write("file2.wav", other_audio, 16_000)

    audio_long = np.concat([audio, audio, audio])

    audio2 = whisper.pad_or_trim(audio_long)

    result = whisper.transcribe(model, audio2, condition_on_previous_text=False, word_timestamps=True, beam_size=5, temperature=0)

def compare_transcripts():
    model = whisper.load_model("small")
    dataset = get_grid_bc()
    refs = []
    array_list = []
    loaded_list = []

    for i, sample in tqdm(enumerate(dataset.select(range(100))), total=100):
        ref = sample["sentence"]
        audio_array = sample["audio"]["array"]

        # load audio and pad/trim it to fit 30 seconds
        audio_loaded = whisper.load_audio(sample["audio_path"], )

        corr = np.correlate(audio_array, audio_loaded, mode="full")
        lag = np.argmax(corr) - (len(audio_loaded) - 1)
        if lag != 0:
            print(f"audio not correlating for {i =}")

        audio_array = whisper.pad_or_trim(audio_array)
        audio_loaded = whisper.pad_or_trim(audio_loaded)

        result_array = whisper.transcribe(model, audio_array, condition_on_previous_text=False, word_timestamps=False,
                                     beam_size=5, temperature=0)

        result_loaded = whisper.transcribe(model, audio_loaded, condition_on_previous_text=False, word_timestamps=False,
                                     beam_size=5, temperature=0)

        if result_array["text"] != result_loaded["text"]:
            refs.append(ref)
            array_list.append(result_array["text"])
            loaded_list.append(result_loaded["text"])
            print(f"text unequal for {i=} ({result_array["text"]}  =/= {result_loaded["text"]} )")

    print(f"array: {wer_needleman_wunsch(reference=refs, transcript=array_list)}")
    print(f"loaded: {wer_needleman_wunsch(reference=refs, transcript=loaded_list)}")

    # mean arr: 5.413363646233904e-10
    # min arr: -1.52587890625e-05
    # mean arr: 1.52587890625e-05

def find_weird_file():
    dataset = get_grid()
    device = select_device()
    model = whisper.load_model("tiny", device=device)


    for sample in tqdm(dataset):
        audio = sample["audio"]["array"]



        result = whisper.transcribe(model, audio, condition_on_previous_text=False, word_timestamps=False,
                                     beam_size=5, temperature=0)

        if len(result["text"]) > 50:
            print(sample["audio_path"])
            print(result["text"])
            break

def inspect_file():
    file_path = "datasets/grid/downloaded_grid_files/audio/s27/s27/lbihzn.wav"
    device = select_device()
    model = whisper.load_model("tiny", device=device)

    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)

    print("reference: ", convert_short_name_to_ref("lbihzn", False))
    result = whisper.transcribe(model, audio,
                                condition_on_previous_text=False,
                                fp16=False,
                                word_timestamps=True,
                                beam_size=5,
                                temperature=0.2,
                                language="en",
                                )

    print(result)
    pass



if __name__ == '__main__':
    compare_transcripts()

