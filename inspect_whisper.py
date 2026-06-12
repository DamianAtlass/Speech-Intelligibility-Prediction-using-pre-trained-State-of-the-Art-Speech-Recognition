import whisper
import numpy as np
from utils.grid_utils import get_grid
from tqdm import tqdm
import torch


downloaded_grid_file_25kHz = "datasets/grid/downloaded_grid_files/audio/s23/s23/sraf4p.wav"
def foo():

    from tqdm import tqdm

    model = whisper.load_model("tiny")
    dataset = get_grid()



    audio = whisper.load_audio(downloaded_grid_file_25kHz)
    #audio = np.pad(audio, (0, 160_000 - len(audio)))

    other_audio = whisper.load_audio(downloaded_grid_file_25kHz)
    import soundfile as sf
    sf.write("file1.wav", audio, 16_000)
    sf.write("file2.wav", other_audio, 16_000)

    audio_long = np.concat([audio, audio, audio])

    audio2 = whisper.pad_or_trim(audio_long)

    result = whisper.transcribe(model, audio2, condition_on_previous_text=False, word_timestamps=True, beam_size=5, temperature=0)

def compare_transcripts():
    model = whisper.load_model("tiny")
    dataset = get_grid()

    for i, sample in tqdm(enumerate(dataset.select(range(100)))):
        audio1 = sample["audio"]["array"]

        # load audio and pad/trim it to fit 30 seconds
        audio2 = whisper.load_audio(sample["audio_path"])

        corr = np.correlate(audio1, audio2, mode="full")
        lag = np.argmax(corr) - (len(audio2) - 1)
        if lag != 0:
            print(f"audio not correlating for {i =}")

        audio1 = whisper.pad_or_trim(audio1)
        audio2 = whisper.pad_or_trim(audio2)

        result1 = whisper.transcribe(model, audio1, condition_on_previous_text=False, word_timestamps=False,
                                     beam_size=5, temperature=0)

        result2 = whisper.transcribe(model, audio2, condition_on_previous_text=False, word_timestamps=False,
                                     beam_size=5, temperature=0)

        if result1["text"] != result2["text"]:
            print(f"text unequal for {i =}")

    # mean arr: 5.413363646233904e-10
    # min arr: -1.52587890625e-05
    # mean arr: 1.52587890625e-05

def foo():
    print(whisper.available_models())
    #for n in ["tiny", "large-v1", "large-v2", "large-v3", "large", "large-v3-turbo", "turbo"]:
    model1 = whisper.load_model("large")
    model2 = whisper.load_model("large-v3")

    same = True

    for k in model1.state_dict():
        if not torch.equal(model1.state_dict()[k], model1.state_dict()[k]):
            same = False
            print("Different at:", k)
            break

    print("Same weights:", same)

def find_weird_file():
    model = whisper.load_model("tiny")
    dataset = get_grid()

    sample = dataset[2396]
    audio = sample["audio"]["array"]

    audio = whisper.pad_or_trim(audio)

    result = whisper.transcribe(model, audio, condition_on_previous_text=False, word_timestamps=False,
                                 beam_size=5, temperature=0)


    print(result["text"])




if __name__ == '__main__':
    find_weird_file()
