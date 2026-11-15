from pystoi import stoi
import librosa
from utils.paths import PROJECT_ROOT

def calculate_stoi(clean_path:str, noisy_path:str) -> float:
    clean_audio, fs_clean = librosa.load(PROJECT_ROOT/clean_path, sr=16_000)
    noisy_audio, fs_noisy = librosa.load(PROJECT_ROOT/noisy_path, sr=16_000)

    assert fs_noisy == fs_clean, "Unequal sample rate!"
    assert len(clean_audio) == len(noisy_audio), "Unequal length!"

    stoi_value = stoi(x=clean_audio, y=noisy_audio, fs_sig=fs_clean, extended=False)
    return stoi_value