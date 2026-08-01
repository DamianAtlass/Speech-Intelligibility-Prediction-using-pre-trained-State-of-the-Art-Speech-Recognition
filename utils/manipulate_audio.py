import numpy as np
import pickle as pkl
import scipy
import logging
logger = logging.getLogger(__name__)
from pathlib import Path

#def get_power(signal: np.ndarray):
#    return np.mean(signal ** 2)

def get_rms(signal: np.ndarray):
    return np.sqrt(np.mean(signal ** 2))

def calculate_snr(signal: np.ndarray, noise: np.ndarray) ->  np.ndarray:
    return 20 * np.log10(get_rms(signal) / get_rms(noise))

def add_gaussian_noise(signal: np.ndarray, target_snr_db: int | float | None) -> np.ndarray:
    """
    Add noise to a signal to resulting in a specific SNR of the returned signal.

    signal, np.ndarray: the original signal
    target_snr_db, float: the SNR which the returned signal should have
    :return:
    """
    if target_snr_db is None:
        return signal

    signal_rms = get_rms(signal)
    noise_rms = signal_rms / (10 ** (target_snr_db / 20))

    noise = np.random.normal(loc=0, scale=noise_rms, size=len(signal))
    assert np.abs(target_snr_db - calculate_snr(signal, noise)) < 1

    noised_signal = signal + noise

    assert len(signal) == len(noised_signal)

    return noised_signal.astype(np.float32)

def add_speech_shaped_noise(signal: np.ndarray,
                            target_snr_db: float | None,
                            filter_path: Path = Path.cwd() / "speechshaped_filter.pkl" # for testing
                            ) -> np.float32:
    """
    Add speech-shaped noise to a signal to resulting in a specific SNR of the returned signal. The speech-shaped noise
    is equivalent to the noise of the grid_bc dataset.

    signal, np.ndarray: the original signal
    target_snr_db, float: the SNR which the returned signal should have
    :return:
    noised_signal, np.float32: the noised signal. Needs to be of that dtype
    """
    if target_snr_db is None:
        return signal

    with open(filter_path, 'rb') as f:
        speechshaped_filter = pkl.load(f)

    signal_rms = get_rms(signal)
    noise_rms = signal_rms / (10 ** (target_snr_db / 20))

    white_noise = np.random.normal(loc=0, scale=1, size=len(signal))

    speech_shaped_noise = scipy.signal.lfilter(
        b=speechshaped_filter['coeffs']['b'],
        a=speechshaped_filter['coeffs']['a'],
        x=white_noise)

    speech_shaped_noise/=get_rms(speech_shaped_noise)
    speech_shaped_noise*=noise_rms

    noised_signal = signal + speech_shaped_noise

    assert np.abs(target_snr_db - calculate_snr(signal, speech_shaped_noise)) < 1
    assert len(signal) == len(noised_signal)

    return noised_signal.astype(np.float32)

def add_noise_transformation(batch: dict):
    batch["audio"] = [
        {"array": add_speech_shaped_noise(sample["array"], target_snr),
         "sampling_rate": sample["sampling_rate"]}
        for sample, target_snr in zip(batch["audio"], batch["snr"])]
    return batch